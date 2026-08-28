use std::collections::VecDeque;

use rand::{Rng, RngCore};

use crate::battle::command::Command;
use crate::battle::engine::end_turn_resolution::{resolve_residual, resolve_turn_end};
use crate::battle::engine::execute_move::execute_move;
use crate::battle::event::Event;
use crate::battle::hooks::{HookTable, Trigger};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::roster::RosterId;
use crate::battle::state::Outcome;
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::{SpeciesDatum, Stat};

const SWITCHING_PRIO: i8 = 9;
const MIN_PRIORITY: i8 = -7;
const MIN_SPEED: u32 = 0;

fn get_next_command(commands: &mut Vec<Command>, registry: &Registry, battle_state: &BattleState, rng: &mut dyn RngCore) -> Option<Command> {
	if commands.is_empty() {
		//println!("empty commands....");
		return None;
	}
	// we need the thing with the highest priority and the highest speed!
	let mut highest_prio = MIN_PRIORITY;
	let mut highest_speed = MIN_SPEED;
	let mut highest_command_index: Option<usize> = None;
	for (index, command) in commands.iter().enumerate() {
		//println!("doing command: {:?}", command);
		let (prio, speed) = match command {
			Command::MoveAction(move_command) => {
				let mv = registry.get_move(move_command.move_id);
				let creature = battle_state.get_mon(move_command.user)?;
				if creature.current_hp <= 0 {
					continue;
				}
				let prio = mv.base_prio;
				let speed = creature.get_stat(Stat::Speed, &registry);
				//println!("prio, speed: {}, {}", prio, speed);
				(prio, speed)
			}
			Command::Switch {current, new: _} => {
				let speed = battle_state.get_mon(*current)?.get_stat(Stat::Speed, &registry);
				(SWITCHING_PRIO, speed)
			}
		};
		if prio > highest_prio {
			highest_prio = prio;
			highest_speed = speed;
			highest_command_index = Some(index);
		} else if prio == highest_prio && speed > highest_speed {
			highest_prio = prio;
			highest_speed = speed;
			highest_command_index = Some(index);
		} else if prio == highest_prio && speed == highest_speed && rng.random::<bool>() {
			// exact speed tie -> break it with a coin flip (favours neither side)
			// note this method gives a higher chance to later moves in the vector with same speed to go first...
			highest_command_index = Some(index);
		}
	}

	if let Some(i) = highest_command_index {
		Some(commands.remove(i))
	} else {
		//println!("returning none....");
		None
	}
}

/// Subtract HP and report how much was actually lost.
///
/// The amount lost can be less than the amount dealt (a creature with 3 HP left
/// only loses 3 to a 40-damage hit), and hooks that react to damage want the
/// real figure, so this returns it rather than a faint flag.
fn apply_damage(amount: u32, target: PositionId, battle_state: &mut BattleState) -> u32 {
	match battle_state.get_mut_mon(target) {
		Some(target_state) => {
			let before = target_state.current_hp;
			target_state.current_hp = before.saturating_sub(amount);
			before - target_state.current_hp
		}
		None => 0,
	}
}

fn apply_healing(amount: u32, target: PositionId, battle_state: &mut BattleState) -> u32 {
	match battle_state.get_mut_mon(target) {
		Some(target_state) => {
			let before = target_state.current_hp;
			target_state.current_hp = before.saturating_add(amount).min(target_state.max_hp);
			target_state.current_hp - before
		}
		None => 0,
	}
}

fn handle_switch_event(current: PositionId, new: RosterId, battle_state: &mut BattleState) {
	battle_state.field[current] = new;
}

#[derive(Debug, PartialEq)]
pub enum StepRequest {
	NeedsActions,
	NeedsReplacements(Vec<PositionId>),
	Finished(Outcome),
}

pub struct StepResult {
	pub battle_state: BattleState,
	pub step_request: StepRequest,
}

/// Where the turn currently is.
///
/// Replaces the old `non_vol_status_handled` boolean. Once the phase moves past
/// `Actions` the engine stops pulling commands, so a command belonging to a
/// creature that fainted mid-turn is dropped rather than retried forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
	/// Players' chosen moves and switches.
	Actions,
	/// Chip damage, healing, delayed attacks.
	Residual,
	/// Countdowns and expiry.
	TurnEnd,
	Done,
}

/// Resolve one turn.
///
/// The signature is unchanged, so nothing downstream (`rl::battle_playout`,
/// `game_window`) needed touching.
///
/// The loop is: drain the event queue, and when it is empty advance the turn one
/// step — take the next command, or move to the next phase. Hooks feed back into
/// the same queue, so an ability that queues damage is resolved by exactly the
/// same machinery as the move that triggered it.
pub fn step(mut battle_state: BattleState, mut commands: Vec<Command>, registry: &Registry, rng: &mut dyn RngCore) -> StepResult {
	// FIFO, not LIFO. The old code used `Vec::pop`, which resolved a move's
	// secondary effects *before* its damage; with hooks queueing follow-up
	// events that ordering would compound.
	let mut events: VecDeque<Event> = VecDeque::new();
	let mut fainted: Vec<PositionId> = Vec::new();
	let mut hooks = HookTable::new();
	let mut phase = Phase::Actions;

	loop {
		// Cheap: returns immediately unless something invalidated the table.
		hooks.refresh(&battle_state, registry);

		if let Some(event) = events.pop_front() {
			execute_event(
				&mut battle_state,
				event,
				&mut fainted,
				&mut events,
				&mut hooks,
				registry,
				rng,
			);
			continue;
		}

		// Event queue is empty, so advance the turn.
		match phase {
			Phase::Actions => match get_next_command(&mut commands, registry, &battle_state, rng) {
				Some(Command::MoveAction(move_command)) => {
					//println!("executing move: {:?}", move_command);
					execute_move(move_command, registry, &battle_state, &mut events, rng, &hooks);
				}
				Some(Command::Switch { current, new }) => {
					events.push_back(Event::Switch { current, new });
				}
				None => phase = Phase::Residual,
			},
			Phase::Residual => {
				resolve_residual(&battle_state, registry, &hooks, &mut events, rng);
				phase = Phase::TurnEnd;
			}
			Phase::TurnEnd => {
				resolve_turn_end(&battle_state, registry, &hooks, &mut events, rng);
				phase = Phase::Done;
			}
			Phase::Done => break,
		}
	}

	//println!("fainted: {:?}", fainted);
	let step_request =
		if let Some(outcome) = battle_state.outcome() {
			StepRequest::Finished(outcome)
		} else if fainted.is_empty() {
			StepRequest::NeedsActions
		} else {
			StepRequest::NeedsReplacements(fainted)
		};
	StepResult {
		battle_state,
		step_request
	}
}

/// Apply one event, then announce it.
///
/// Every arm has the same shape: mutate, invalidate the hook table if the
/// mutation changed *who is subscribed*, refresh, broadcast. Hooks respond by
/// appending to `queue`, which this same loop will drain.
fn execute_event(
	battle_state: &mut BattleState,
	event: Event,
	fainted: &mut Vec<PositionId>,
	queue: &mut VecDeque<Event>,
	hooks: &mut HookTable,
	registry: &Registry,
	rng: &mut dyn RngCore,
) {
	match event {
		Event::DealDamage { amount, target, source } => {
			//println!("dealing {} to {:?}", amount, target);
			let dealt = apply_damage(amount, target, battle_state);

			hooks.refresh(battle_state, registry);
			hooks.dispatch(
				Trigger::AfterDamage { target, amount: dealt, source },
				battle_state,
				registry,
				queue,
				rng,
			);

			if has_fainted(battle_state, target) && !fainted.contains(&target) {
				queue.push_back(Event::Faint { target });
			}
		}

		Event::Heal { amount, target } => {
			apply_healing(amount, target, battle_state);
		}

		Event::Switch { current, new } => {
			hooks.refresh(battle_state, registry);
			hooks.dispatch(Trigger::SwitchOut { pos: current }, battle_state, registry, queue, rng);

			handle_switch_event(current, new, battle_state);

			// A different creature is standing there now, so its hooks replace
			// the old one's.
			hooks.invalidate();
			hooks.refresh(battle_state, registry);
			hooks.dispatch(Trigger::SwitchIn { pos: current }, battle_state, registry, queue, rng);
		}

		Event::ApplyNonVolStatus { status, target } => {
			// Re-checked here, not just where the event was queued. A move can
			// queue a status and then something in between — another hit, a
			// faint, an ability landing a status first — can make it invalid.
			// This is the check that actually counts; the one in `execute_move`
			// is just an early-out that avoids queueing a doomed event.
			hooks.refresh(battle_state, registry);
			if has_fainted(battle_state, target)
				|| !hooks.allows_status(battle_state, registry, target, status)
			{
				return;
			}

			if let Some(mon) = battle_state.get_mut_mon(target) {
				mon.non_vol_status = status;
			}

			// The new status brings its own hooks.
			hooks.invalidate();
			hooks.refresh(battle_state, registry);
			hooks.dispatch(Trigger::StatusApplied { target, status }, battle_state, registry, queue, rng);
		}

		Event::SetWeather { weather } => {
			battle_state.weather = weather;

			hooks.invalidate();
			hooks.refresh(battle_state, registry);
			hooks.dispatch(Trigger::WeatherChanged, battle_state, registry, queue, rng);
		}

		Event::Faint { target } => {
			// Guard against a second lethal hit queueing a duplicate faint.
			if fainted.contains(&target) {
				return;
			}
			fainted.push(target);

			// Broadcast BEFORE invalidating, deliberately. `providers::collect`
			// skips creatures at 0 HP, so rebuilding first would drop the
			// fainting creature's own hooks and an on-faint effect (Aftermath,
			// a berry) would never fire.
			hooks.dispatch(Trigger::AfterFaint { pos: target }, battle_state, registry, queue, rng);
			hooks.invalidate();
		}
	}
}

fn has_fainted(battle_state: &BattleState, pos: PositionId) -> bool {
	match battle_state.get_mon(pos) {
		Some(mon) => mon.current_hp == 0,
		None => false,
	}
}

pub fn log_move_usage(battle_state: &BattleState, registry: &Registry, user: PositionId, target: PositionId, mv: MoveId) {
	let user_name = get_species_data(battle_state, registry, user).name;
	let target_name = get_species_data(battle_state, registry, target).name;

	let move_name = &registry.get_move(mv).name;
	println!("{} used {} on {}", user_name, move_name, target_name);
}

fn get_species_data(battle_state: &BattleState, registry: &Registry, pos: PositionId) -> SpeciesDatum {
	registry.get_species_data(battle_state.get_mon(pos).unwrap().species_id).clone()
}

/********************************
 *
 * TESTS BEGIN HERE:
 *
 */

#[cfg(test)]
mod tests {
	use crate::battle::hooks::{QueryKind, TriggerKind};
	use crate::battle::state::creature_state::CreatureState;
	use crate::battle::state::non_volatile_status::NonVolatileStatus;
	use crate::battle::state::weather::{TimedWeather, Weather};
	use crate::{battle::{command::MoveCommand, state::Team}, model::{pmove::{MoveId, MoveTargeting, MoveType, PMove}, speciesdata::SpeciesId}};
	use crate::battle::engine::calculate_damage::calculate_damage;
	// maybe i should define my own moves here that aren't actual moves in the
	// registry for more independent testing...
	use super::*;

	fn frail_attacker() -> SpeciesDatum {
		SpeciesDatum {
			name: String::from("frail_attacker"),
			base_hp: 80,
			species_id: SpeciesId(0),
			attack: 100,
			defense: 100,
			speed: 90,
		}
	}

	fn fat_defender() -> SpeciesDatum {
		SpeciesDatum {
			name: String::from("fat_defender"),
			base_hp: 120,
			species_id: SpeciesId(1),
			attack: 80,
			defense: 80,
			speed: 30,
		}
	}

	fn big_damage_attack() -> PMove {
		PMove {
			name: String::from("big_damage"),
			move_id: MoveId(2),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 999999,
			effects: vec![],
			base_prio: 0,
		}
	}

	fn tackle() -> PMove {
		PMove {
			name: String::from("tackle"),
			move_id: MoveId(0),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 0,
		}
	}

	fn quick_attack() -> PMove {
		PMove {
			name: String::from("quick-attack"),
			move_id: MoveId(1),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 1,
		}
	}

	fn test_registry() -> Registry {
		Registry {
			species_data: vec![frail_attacker(), fat_defender()],
			moves: vec![tackle(), quick_attack(), big_damage_attack()],
		}
	}

	fn test_battle_state() -> BattleState {
		let team0: Vec<CreatureState> = vec![
			CreatureState::from_species_data(&frail_attacker(), vec![MoveId(0), MoveId(1), MoveId(2)]),
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1), MoveId(2)]),
		];
		let team1: Vec<CreatureState> = vec![
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1), MoveId(2)]),
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1), MoveId(2)]),
		];
		BattleState::from(
			team0,
			team1,
			vec![0, 1],
		)
	}

	fn frail_uses_tackle() -> Command {
		Command::MoveAction(MoveCommand {
			move_id: tackle().move_id,
			user: PositionId(0),
			targets: vec![PositionId(1)]
		})
	}

	fn fat_uses_quick_attack() -> Command {
		Command::MoveAction(MoveCommand {
			move_id: quick_attack().move_id,
			user: PositionId(1),
			targets: vec![PositionId(0)]
		})
	}

	fn fat_uses_tackle() -> Command {
		Command::MoveAction(MoveCommand {
			move_id: tackle().move_id,
			user: PositionId(1),
			targets: vec![PositionId(0)]
		})
	}

	// this test will need changes when it comes to the actual damage formula
	#[test]
	fn uses_correct_stats_for_damage_calc() {
		let registry = test_registry();
		let mut rng = rand::rng();

		let battle_state = test_battle_state();
		let fat_hp = battle_state.get_mon(PositionId(1)).unwrap().current_hp;

		let next_battle_state = step(battle_state, vec![frail_uses_tackle()], &registry, &mut rng).battle_state;
		let fat_hp_after = next_battle_state.get_mon(PositionId(1)).unwrap().current_hp;

		let tackle_power = tackle().base_power;
		let frail_attack = frail_attacker().attack;
		let fat_defense = fat_defender().defense;
		let actual_damage = calculate_damage(frail_attack as u32, fat_defense as u32, tackle_power);


		assert!(actual_damage == fat_hp - fat_hp_after);
	}

	#[test]
	fn higher_priority_goes_first() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let battle_state = test_battle_state();


		let mut v = vec![frail_uses_tackle(), fat_uses_quick_attack()];

		// fat using quick attack comes first because of priority
		let x = get_next_command(&mut v, &registry, &battle_state, &mut rng);
		let y = get_next_command(&mut v, &registry, &battle_state, &mut rng);
		assert!(x == Some(fat_uses_quick_attack()));
		assert!(y == Some(frail_uses_tackle()));

	}

	#[test]
	fn faster_speed_goes_first() {
		let registry = test_registry();
		let battle_state = test_battle_state();
		let mut rng = rand::rng();

		let mut v = vec![fat_uses_tackle(), frail_uses_tackle()];

		// get_next_command should fetch the frail's tackle before the fat's, because it's faster
		assert!(get_next_command(&mut v, &registry, &battle_state, &mut rng) == Some(frail_uses_tackle()));
		assert!(get_next_command(&mut v, &registry, &battle_state, &mut rng) == Some(fat_uses_tackle()));

	}

	// the example step is more of an integration test...
	#[test]
	fn example_step() {
		let registry = Registry::load();
		let mut rng = rand::rng();

		let battle_state = test_battle_state();

		let command1 = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(0),
			targets: vec![PositionId(1)]
		});

		let new_state = step(battle_state, vec![command1], &registry, &mut rng).battle_state;

		println!("{:?}", new_state);
	}

	#[test]
	fn switching_changes_the_active_mon() {

		let registry = test_registry();
		let mut rng = rand::rng();

		// team0 has TWO mons: frail_attacker (roster 0) and fat_defender (roster 2).
		// team1 has one creature (roster 1). field starts pointing position 0 -> roster 0.
		let team0 = vec![
			CreatureState::from_species_data(&frail_attacker(), vec![MoveId(0), MoveId(1)]),
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1)]),
		];
		let team1 = vec![
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1)]),
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1)]),
		];
		let battle_state = BattleState::from(team0, team1, vec![0, 1]);

		// before switching, position 0's active creature is the frail_attacker (species 0)
		assert_eq!(battle_state.get_mon(PositionId(0)).unwrap().species_id.0, 0);

		// switch position 0 to the benched fat_defender at roster index 2
		let switch = Command::Switch {
			current: PositionId(0),
			new: RosterId(2),
		};
		let next = step(battle_state, vec![switch], &registry, &mut rng).battle_state;

		// the active creature at position 0 is now the fat_defender (species 1)
		assert_eq!(next.get_mon(PositionId(0)).unwrap().species_id.0, 1);
		// the field mapping now points position 0 at roster index 2
		assert_eq!(next.field[PositionId(0)], RosterId(2));
		// and the switched-OUT creature still exists in the roster (data preserved, not moved)
		assert_eq!(next.roster.get_mon(RosterId(0)).as_ref().unwrap().species_id.0, 0);
	}

	#[test]
	fn requests_replacement_when_mon_faints() {
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();

		// set the position 0 creature's hp to 1
		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 1;
		// hit the position 0 creature with a really strong attack
		let attack_command = Command::MoveAction(MoveCommand {
			move_id: MoveId(2),
			targets: vec![PositionId(0)],
			user: PositionId(1),
		});
		// since there is still one creature left on team 0, we should request replacements rather
		// than ending the game...
		let StepResult {
			battle_state,
			step_request
		} = step(battle_state, vec![attack_command], &test_registry(), &mut rng);

		assert_eq!(battle_state.get_mon(PositionId(0)).unwrap().current_hp, 0); // it should be ko'd
		let v = vec![PositionId(0)];
		assert_eq!(step_request, StepRequest::NeedsReplacements(v)); // we should need replacement
	}

	#[test]
	fn poison_ko_requests_replacement() {
		let mut rng = rand::rng();
		let registry = test_registry();
		let mut battle_state = test_battle_state();

		// poison position 0's creature (the frail_attacker, roster 0) and drop it low
		// enough that end-of-turn poison will KO it.
		// frail base_hp = 80, so poison ticks max_hp/8 = 10; hp = 1 guarantees a KO.
		let mon = battle_state.get_mut_mon(PositionId(0)).unwrap();
		mon.non_vol_status = NonVolatileStatus::Poison;
		mon.current_hp = 1;

		// No attacks — pass empty commands so the turn goes straight to the
		// end-of-turn phase, and the KO is caused purely by poison, nothing else.
		let StepResult { battle_state, step_request } =
			step(battle_state, vec![], &registry, &mut rng);

		// The poisoned creature should have fainted from the poison tick...
		assert_eq!(battle_state.get_mon(PositionId(0)).unwrap().current_hp, 0);

		// ...and since team0 still has a live benched creature (roster 2), the engine
		// must ask for a replacement rather than declaring the battle finished.
		assert!(matches!(step_request, StepRequest::NeedsReplacements(_)));
	}

	#[test]
	fn poison_ko_of_last_creature_finishes_battle() {
		let mut rng = rand::rng();
		let registry = test_registry();

		// team0 has ONLY ONE creature — so when it dies, there's nobody to replace it.
		let team0 = vec![
			CreatureState::from_species_data(&frail_attacker(), vec![MoveId(0), MoveId(1), MoveId(2)]),
		];
		let team1 = vec![
			CreatureState::from_species_data(&fat_defender(), vec![MoveId(0), MoveId(1), MoveId(2)]),
		];
		let mut battle_state = BattleState::from(team0, team1, vec![0, 1]);

		// Poison team0's only creature and weaken it so end-of-turn poison KOs it.
		{
			let mon = battle_state.get_mut_mon(PositionId(0)).unwrap();
			mon.non_vol_status = NonVolatileStatus::Poison;
			mon.current_hp = 1;
		}

		// No attacks — the end-of-turn poison tick is the only thing that happens.
		let StepResult { battle_state, step_request } =
			step(battle_state, vec![], &registry, &mut rng);

		// The creature fainted from poison...
		assert_eq!(battle_state.get_mon(PositionId(0)).unwrap().current_hp, 0);

		// ...and because it was team0's LAST creature, the battle must be FINISHED (team1 wins),
		// NOT NeedsReplacements. This is the key guard: outcome() has to be checked before the
		// faint list, otherwise the engine would try to request a replacement for a wiped side.
		assert_eq!(step_request, StepRequest::Finished(Outcome::Win { team: Team::One }));
	}

	/*******************************
	 *
	 * HOOK SYSTEM TESTS:
	 *
	 */

	/// The whole point of the table: a moment nobody subscribed to costs nothing,
	/// and a status installs exactly the hooks it declares — not a scan of the field.
	#[test]
	fn table_only_indexes_what_actually_subscribed() {
		let registry = test_registry();
		let mut battle_state = test_battle_state();
		let mut hooks = HookTable::new();

		// Clean field: nothing is poisoned, burned, or standing in weather.
		hooks.refresh(&battle_state, &registry);
		assert_eq!(hooks.reactive_count(TriggerKind::Residual), 0);
		assert_eq!(hooks.query_count(QueryKind::ModifyStat), 0);
		assert_eq!(hooks.reactive_count(TriggerKind::SwitchIn), 0);

		// Burning one creature adds exactly its three declared hooks and nothing else.
		battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Burn;
		hooks.invalidate();
		hooks.refresh(&battle_state, &registry);
		assert_eq!(hooks.reactive_count(TriggerKind::Residual), 1);
		assert_eq!(hooks.query_count(QueryKind::ModifyStat), 1);
		assert_eq!(hooks.query_count(QueryKind::TryApplyStatus), 1);
		// Still nothing listening to switch-in, so switching stays free.
		assert_eq!(hooks.reactive_count(TriggerKind::SwitchIn), 0);
	}

	/// A fainted creature stops being a subscriber, so it cannot keep taking
	/// residual damage or modifying anything.
	#[test]
	fn fainted_creatures_stop_subscribing() {
		let registry = test_registry();
		let mut battle_state = test_battle_state();
		let mut hooks = HookTable::new();

		{
			let mon = battle_state.get_mut_mon(PositionId(0)).unwrap();
			mon.non_vol_status = NonVolatileStatus::Poison;
		}
		hooks.refresh(&battle_state, &registry);
		assert_eq!(hooks.reactive_count(TriggerKind::Residual), 1);

		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 0;
		hooks.invalidate();
		hooks.refresh(&battle_state, &registry);
		assert_eq!(hooks.reactive_count(TriggerKind::Residual), 0);
	}

	/// A modifier hook changing a number the engine was about to use.
	#[test]
	fn burn_halves_attack_through_a_query_hook() {
		let registry = test_registry();
		let mut rng = rand::rng();

		let healthy_damage = {
			let battle_state = test_battle_state();
			let before = battle_state.get_mon(PositionId(1)).unwrap().current_hp;
			let after = step(battle_state, vec![frail_uses_tackle()], &registry, &mut rng)
				.battle_state
				.get_mon(PositionId(1))
				.unwrap()
				.current_hp;
			before - after
		};

		let burned_damage = {
			let mut battle_state = test_battle_state();
			battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Burn;
			let before = battle_state.get_mon(PositionId(1)).unwrap().current_hp;
			let after = step(battle_state, vec![frail_uses_tackle()], &registry, &mut rng)
				.battle_state
				.get_mon(PositionId(1))
				.unwrap()
				.current_hp;
			before - after
		};

		// attack 100 -> 50, so 100*40/80 = 50 becomes 50*40/80 = 25.
		assert_eq!(healthy_damage, 50);
		assert_eq!(burned_damage, 25);
	}

	/// Burn's residual hook still fires in the same turn it is halving Attack —
	/// one status, two hooks, two different moments.
	#[test]
	fn burn_also_chips_its_owner() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();

		battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Burn;
		let before = battle_state.get_mon(PositionId(0)).unwrap().current_hp;
		let max_hp = battle_state.get_mon(PositionId(0)).unwrap().max_hp;

		let after = step(battle_state, vec![], &registry, &mut rng)
			.battle_state
			.get_mon(PositionId(0))
			.unwrap()
			.current_hp;

		assert_eq!(before - after, max_hp / 16);
	}

	/// A veto hook. The rule lives on the status that already exists, and the
	/// engine never asks "does this creature already have a status?".
	#[test]
	fn an_existing_status_blocks_a_new_one() {
		let registry = test_registry();
		let battle_state = {
			let mut bs = test_battle_state();
			bs.get_mut_mon(PositionId(1)).unwrap().non_vol_status = NonVolatileStatus::Burn;
			bs
		};

		let mut hooks = HookTable::new();
		hooks.refresh(&battle_state, &registry);

		// The burned creature refuses poison...
		assert!(!hooks.allows_status(
			&battle_state,
			&registry,
			PositionId(1),
			NonVolatileStatus::Poison
		));
		// ...but its healthy opponent accepts it.
		assert!(hooks.allows_status(
			&battle_state,
			&registry,
			PositionId(0),
			NonVolatileStatus::Poison
		));
	}

	/// Weather is a field-level hook owner: one subscription, damage to everyone.
	#[test]
	fn sandstorm_chips_every_creature_on_the_field() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();
		battle_state.weather = Some(TimedWeather {
			weather: Weather::Sandstorm,
			turns_left: 5,
		});

		let before: Vec<u32> = vec![
			battle_state.get_mon(PositionId(0)).unwrap().current_hp,
			battle_state.get_mon(PositionId(1)).unwrap().current_hp,
		];
		let max_hps: Vec<u32> = vec![
			battle_state.get_mon(PositionId(0)).unwrap().max_hp,
			battle_state.get_mon(PositionId(1)).unwrap().max_hp,
		];

		let next = step(battle_state, vec![], &registry, &mut rng).battle_state;

		assert_eq!(
			before[0] - next.get_mon(PositionId(0)).unwrap().current_hp,
			max_hps[0] / 16
		);
		assert_eq!(
			before[1] - next.get_mon(PositionId(1)).unwrap().current_hp,
			max_hps[1] / 16
		);
		// and it ticked down by one
		assert_eq!(next.weather.unwrap().turns_left, 4);
	}

	/// The countdown hook clears the weather when it runs out, and the residual
	/// damage still happens on that final turn (order::WEATHER_DAMAGE runs in the
	/// Residual phase, the countdown in TurnEnd).
	#[test]
	fn weather_expires_after_its_last_turn() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();
		battle_state.weather = Some(TimedWeather {
			weather: Weather::Sandstorm,
			turns_left: 1,
		});
		let before = battle_state.get_mon(PositionId(0)).unwrap().current_hp;
		let max_hp = battle_state.get_mon(PositionId(0)).unwrap().max_hp;

		let next = step(battle_state, vec![], &registry, &mut rng).battle_state;

		assert_eq!(next.weather, None);
		assert_eq!(before - next.get_mon(PositionId(0)).unwrap().current_hp, max_hp / 16);
	}

	/// Two residual sources on the same creature resolve in the declared order
	/// (weather damage, then poison) and both land.
	#[test]
	fn residual_sources_stack_in_declared_order() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();
		battle_state.weather = Some(TimedWeather {
			weather: Weather::Sandstorm,
			turns_left: 5,
		});
		battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Poison;

		let before = battle_state.get_mon(PositionId(0)).unwrap().current_hp;
		let max_hp = battle_state.get_mon(PositionId(0)).unwrap().max_hp;

		let next = step(battle_state, vec![], &registry, &mut rng).battle_state;

		assert_eq!(
			before - next.get_mon(PositionId(0)).unwrap().current_hp,
			max_hp / 16 + max_hp / 8
		);
	}

	/// The status veto is enforced where the state is actually mutated, not only
	/// where the event was queued.
	#[test]
	fn status_veto_is_enforced_at_apply_time() {
		let registry = Registry {
			species_data: vec![frail_attacker(), fat_defender()],
			// a 0-power move that always poisons
			moves: vec![PMove {
				name: String::from("always_poison"),
				move_id: MoveId(0),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![crate::model::effect::Effect::PoisonChance { chance: 100 }],
				base_prio: 0,
			}],
		};
		let mut rng = rand::rng();

		// The target is already burned, so poison must not overwrite it.
		let mut battle_state = test_battle_state();
		battle_state.get_mut_mon(PositionId(1)).unwrap().non_vol_status = NonVolatileStatus::Burn;

		let poison_them = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(0),
			targets: vec![PositionId(1)],
		});

		let next = step(battle_state, vec![poison_them], &registry, &mut rng).battle_state;

		assert_eq!(
			next.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::Burn
		);
	}

	/// A status lands normally on an unafflicted target — the veto above is a
	/// rule, not the hook system failing to apply anything at all.
	#[test]
	fn status_lands_on_a_clean_target() {
		let registry = Registry {
			species_data: vec![frail_attacker(), fat_defender()],
			moves: vec![PMove {
				name: String::from("always_poison"),
				move_id: MoveId(0),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![crate::model::effect::Effect::PoisonChance { chance: 100 }],
				base_prio: 0,
			}],
		};
		let mut rng = rand::rng();
		let battle_state = test_battle_state();

		let poison_them = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(0),
			targets: vec![PositionId(1)],
		});

		let next = step(battle_state, vec![poison_them], &registry, &mut rng).battle_state;

		assert_eq!(
			next.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::Poison
		);
	}

	/// Residual damage never over-kills into a second faint entry.
	#[test]
	fn a_creature_only_faints_once() {
		let registry = test_registry();
		let mut rng = rand::rng();
		let mut battle_state = test_battle_state();
		battle_state.weather = Some(TimedWeather {
			weather: Weather::Sandstorm,
			turns_left: 5,
		});
		{
			let mon = battle_state.get_mut_mon(PositionId(0)).unwrap();
			mon.non_vol_status = NonVolatileStatus::Poison;
			mon.current_hp = 1;
		}

		let StepResult { step_request, .. } = step(battle_state, vec![], &registry, &mut rng);

		match step_request {
			StepRequest::NeedsReplacements(positions) => {
				assert_eq!(positions, vec![PositionId(0)]);
			}
			other => panic!("expected a single replacement request, got {:?}", other),
		}
	}
}
