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
use crate::battle::state::volatile::VolatileKind;
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

/// Restore HP, but never to a creature that has already fainted.
///
/// A fainted creature stays fainted, as in the games. Without this guard a
/// creature could be knocked to 0 during the action phase and then healed back
/// above 0 by an event queued earlier in the same step — most easily a Leech
/// Seed drain, which heals the *seeder*, who carries no volatile of their own
/// and so is easy to miss. The engine had already recorded the faint, so the
/// step ended by asking for a replacement for a creature that was visibly alive
/// and still on the field. Roughly 8% of battles hit it.
fn apply_healing(amount: u32, target: PositionId, battle_state: &mut BattleState) -> u32 {
	match battle_state.get_mut_mon(target) {
		Some(target_state) if target_state.current_hp > 0 => {
			let before = target_state.current_hp;
			target_state.current_hp = before.saturating_add(amount).min(target_state.max_hp);
			target_state.current_hp - before
		}
		_ => 0,
	}
}

fn handle_switch_event(current: PositionId, new: RosterId, battle_state: &mut BattleState) {
	battle_state.field[current] = new;
}

/// Route damage into a Substitute if the target has one.
///
/// Returns whether the decoy took the hit. The Substitute breaks when its own HP
/// runs out, and the excess is *not* carried through to the creature behind it.
fn absorb_with_substitute(amount: u32, target: PositionId, battle_state: &mut BattleState) -> bool {
	let mon = match battle_state.get_mut_mon(target) {
		Some(mon) => mon,
		None => return false,
	};
	if !mon.volatiles.has(VolatileKind::Substitute) {
		return false;
	}
	let remaining = mon.volatiles.value(VolatileKind::Substitute);
	if amount >= remaining {
		mon.volatiles.remove(VolatileKind::Substitute);
	} else {
		mon.volatiles.set_value(VolatileKind::Substitute, remaining - amount);
	}
	true
}

/// Drop the volatiles that only last the turn they were applied on.
fn clear_turn_scoped_volatiles(battle_state: &mut BattleState) {
	for pos in battle_state.field.all_field_positions() {
		if let Some(mon) = battle_state.get_mut_mon(pos) {
			mon.volatiles.clear_turn_scoped();
		}
	}
}

/// Count every timed volatile down by one and drop the expired ones.
fn tick_volatiles(battle_state: &mut BattleState) {
	for pos in battle_state.field.all_field_positions() {
		if let Some(mon) = battle_state.get_mut_mon(pos) {
			mon.volatiles.tick();
		}
	}
}

/// `Clone` so a UI can hold the pending phase between frames — the game window
/// has to remember "someone still owes me a replacement" across redraws.
#[derive(Debug, PartialEq, Clone)]
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
	/// Before anyone acts: sweep last turn's one-turn volatiles, then let the
	/// things that randomly cancel a move roll for it.
	TurnStart,
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
	let mut phase = Phase::TurnStart;

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
			Phase::TurnStart => {
				hooks.dispatch(Trigger::TurnStart, &battle_state, registry, &mut events, rng);
				phase = Phase::Actions;
			}
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
				// Swept at the END of the turn rather than the start. Both work
				// for blocking, but sweeping here means a Protect or a flinch is
				// never still sitting on a creature once the turn is over — which
				// matters for anything reading the state, the game window included.
				clear_turn_scoped_volatiles(&mut battle_state);
				tick_volatiles(&mut battle_state);
				hooks.invalidate();
				phase = Phase::Done;
			}
			Phase::Done => break,
		}
	}

	//println!("fainted: {:?}", fainted);

	// `fainted` is collected as HP reaches zero during the step, but the step is
	// not over at that point. Anything that restores HP afterwards would leave a
	// stale entry here, and the caller would be asked to replace a creature that
	// is alive and still on the field — which the agent cannot do, so it picks a
	// move instead and the turn silently goes wrong.
	//
	// `apply_healing` refusing to revive a fainted creature is what should make
	// this impossible. This is the invariant restated where it is consumed, so a
	// future effect that restores HP some other way cannot reintroduce the bug.
	fainted.retain(|pos| has_fainted(&battle_state, *pos));

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
		Event::ChangeStats { target, stat_changes } => {
			match battle_state.get_mut_mon(target) {
				Some(s) => s.stat_changes.combine_in_place(&stat_changes),
				None => {}
			}
			/* We do the hooks later
			hooks.dispatch (

			) */
		},
		Event::DealDamage { amount, target, source } => {
			//println!("dealing {} to {:?}", amount, target);
			// A Substitute soaks damage from attacks before any of it reaches HP.
			// This is the one effect the hook system cannot express: absorbing
			// requires mutating the decoy's own HP, and hooks may not mutate. Chip
			// damage (weather, status, Leech Seed) carries no source and goes
			// straight through, which is how the games behave.
			let absorbed = source.is_some() && absorb_with_substitute(amount, target, battle_state);
			let dealt = if absorbed { 0 } else { apply_damage(amount, target, battle_state) };

			if absorbed {
				hooks.invalidate();
			}

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

			// SwitchOut hooks are dispatched into their own queue and drained
			// *before* the field pointer moves.
			//
			// This matters more than it looks. Events target a `PositionId`, and
			// a switch is precisely the moment a position stops meaning the same
			// creature. If a departing creature's hook (Natural Cure curing its
			// own status, say) went on the main queue, it would be applied after
			// the swap and land on the creature coming in.
			//
			// Recursion depth here is 2, not unbounded: the nested call queues
			// its own follow-ups into `departing` and this same loop drains them.
			let mut departing: VecDeque<Event> = VecDeque::new();
			hooks.dispatch(Trigger::SwitchOut { pos: current }, battle_state, registry, &mut departing, rng);
			while let Some(pending) = departing.pop_front() {
				execute_event(battle_state, pending, fainted, &mut departing, hooks, registry, rng);
			}

			// Volatiles do not survive leaving the field — that is the whole
			// distinction between them and a status. Wiped on the way out so the
			// creature is clean if it comes back later.
			if let Some(mon) = battle_state.get_mut_mon(current) {
				mon.volatiles.clear();
			}

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
				// Any change of status restarts the bad-poison ramp — a fresh
				// poisoning begins at 1/16, and a cure leaves nothing to count.
				mon.volatiles.remove(VolatileKind::ToxicCounter);
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

		Event::ApplyVolatile { target, volatile } => {
			if let Some(mon) = battle_state.get_mut_mon(target) {
				if mon.is_alive() {
					mon.volatiles.add(volatile);
				}
			}
			// A volatile brings its own hooks.
			hooks.invalidate();
		}

		Event::RemoveVolatile { target, kind } => {
			if let Some(mon) = battle_state.get_mut_mon(target) {
				mon.volatiles.remove(kind);
			}
			hooks.invalidate();
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
	use crate::battle::state::volatile::{Volatile, VolatileKind};
	use crate::battle::state::weather::{TimedWeather, Weather};
	use crate::model::ability::AbilityId;
	use crate::model::effect::Effect;
	use crate::model::typing::{Type, Typing};
	use crate::{battle::{command::MoveCommand, state::Team}, model::{pmove::{MoveFlags, MoveId, MoveTargeting, MoveType, PMove}, speciesdata::SpeciesId}};
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
			special_attack: 100,
			special_defense: 100,
			speed: 90,
			typing: Typing::mono(Type::Normal),
			ability: None,
		}
	}

	fn fat_defender() -> SpeciesDatum {
		SpeciesDatum {
			name: String::from("fat_defender"),
			base_hp: 120,
			species_id: SpeciesId(1),
			attack: 80,
			defense: 80,
			special_attack: 80,
			special_defense: 80,
			speed: 30,
			typing: Typing::mono(Type::Normal),
			ability: None,
		}
	}

	/// Same shell as `frail_attacker`, but with an ability bolted on. Lets a test
	/// isolate exactly one ability without any other difference.
	fn with_ability(base: SpeciesDatum, species_id: SpeciesId, ability: AbilityId) -> SpeciesDatum {
		SpeciesDatum {
			name: format!("{}_{:?}", base.name, ability),
			species_id,
			ability: Some(ability),
			..base
		}
	}

	fn big_damage_attack() -> PMove {
		PMove {
			name: String::from("big_damage"),
			element: Type::Water,
			move_id: MoveId(2),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 999999,
			effects: vec![],
			base_prio: 0,
			flags: MoveFlags::NONE,
		}
	}

	fn tackle() -> PMove {
		PMove {
			name: String::from("tackle"),
			element: Type::Water,
			move_id: MoveId(0),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 0,
			flags: MoveFlags::CONTACT,
		}
	}

	fn quick_attack() -> PMove {
		PMove {
			name: String::from("quick-attack"),
			element: Type::Water,
			move_id: MoveId(1),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 1,
			flags: MoveFlags::CONTACT,
		}
	}

	/// MoveId(3): a ground-flagged hit, for Levitate.
	fn earth_jab() -> PMove {
		PMove {
			name: String::from("earth_jab"),
			element: Type::Ground,
			move_id: MoveId(3),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 0,
			flags: MoveFlags::NONE,
		}
	}

	/// MoveId(4): a non-contact special hit, for the physical/special split.
	fn mind_beam() -> PMove {
		PMove {
			name: String::from("mind_beam"),
			element: Type::Water,
			move_id: MoveId(4),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Special,
			base_power: 40,
			effects: vec![],
			base_prio: 0,
			flags: MoveFlags::NONE,
		}
	}

	/// MoveId(5): pure status, no damage component at all.
	fn hex_glare() -> PMove {
		PMove {
			name: String::from("hex_glare"),
			element: Type::Water,
			move_id: MoveId(5),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Status,
			base_power: 0,
			effects: vec![Effect::ParalysisChance { chance: 100 }],
			base_prio: 0,
			flags: MoveFlags::NONE,
		}
	}

	fn test_moves() -> Vec<PMove> {
		vec![tackle(), quick_attack(), big_damage_attack(), earth_jab(), mind_beam(), hex_glare()]
	}

	fn test_registry() -> Registry {
		Registry {
			species_data: vec![frail_attacker(), fat_defender()],
			moves: test_moves(),
		}
	}

	/// Registry where species 2 is `frail_attacker` plus one ability.
	fn registry_with_ability(ability: AbilityId) -> Registry {
		Registry {
			species_data: vec![
				frail_attacker(),
				fat_defender(),
				with_ability(frail_attacker(), SpeciesId(2), ability),
			],
			moves: test_moves(),
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

		// Asserted as a ratio rather than two magic numbers, so this keeps
		// testing the hook rather than the damage formula.
		assert!(healthy_damage > 0, "the control hit should do something");
		let expected = healthy_damage / 2;
		assert!(
			burned_damage.abs_diff(expected) <= 1,
			"burn should roughly halve damage: {} unburned -> {} burned, expected about {}",
			healthy_damage, burned_damage, expected
		);
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
			// a status move that always poisons
			moves: vec![PMove {
				name: String::from("always_poison"),
			element: Type::Water,
				move_id: MoveId(0),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![Effect::PoisonChance { chance: 100 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
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
			element: Type::Water,
				move_id: MoveId(0),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![Effect::PoisonChance { chance: 100 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
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

	/*******************************
	 *
	 * ABILITY TESTS:
	 *
	 * Each of these puts one ability on species 2 and changes nothing else, so a
	 * failure points at exactly one hook.
	 */

	/// Build a battle where position 0 holds `ability` and position 1 is a plain
	/// fat_defender. Both get every test move.
	fn ability_battle(ability: AbilityId) -> (Registry, BattleState) {
		let registry = registry_with_ability(ability);
		let all_moves = vec![MoveId(0), MoveId(3), MoveId(4), MoveId(5)];
		let holder = CreatureState::from_species_data(
			registry.get_species_data(SpeciesId(2)),
			all_moves.clone(),
		);
		let bench = CreatureState::from_species_data(&fat_defender(), all_moves.clone());
		let foe = CreatureState::from_species_data(&fat_defender(), all_moves.clone());
		let foe_bench = CreatureState::from_species_data(&fat_defender(), all_moves);
		let battle_state = BattleState::from(vec![holder, bench], vec![foe, foe_bench], vec![0, 1]);
		(registry, battle_state)
	}

	fn hp_at(battle_state: &BattleState, pos: PositionId) -> u32 {
		battle_state.get_mon(pos).unwrap().current_hp
	}

	/// Levitate zeroes ground damage, and only ground damage.
	#[test]
	fn levitate_blocks_ground_moves_only() {
		let mut rng = rand::rng();
		let (registry, battle_state) = ability_battle(AbilityId::Levitate);

		let ground_hit = Command::MoveAction(MoveCommand {
			move_id: MoveId(3), // earth_jab
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		let before = hp_at(&battle_state, PositionId(0));
		let after = step(battle_state, vec![ground_hit], &registry, &mut rng).battle_state;
		assert_eq!(hp_at(&after, PositionId(0)), before, "levitate should have nullified it");

		// The same creature still takes a normal contact hit.
		let (registry, battle_state) = ability_battle(AbilityId::Levitate);
		let normal_hit = Command::MoveAction(MoveCommand {
			move_id: MoveId(0), // tackle
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		let before = hp_at(&battle_state, PositionId(0));
		let after = step(battle_state, vec![normal_hit], &registry, &mut rng).battle_state;
		assert!(hp_at(&after, PositionId(0)) < before, "non-ground damage should still land");
	}

	/// Rough Skin punishes contact and ignores non-contact.
	#[test]
	fn rough_skin_punishes_contact_only() {
		let mut rng = rand::rng();

		let (registry, battle_state) = ability_battle(AbilityId::RoughSkin);
		let attacker_max = battle_state.get_mon(PositionId(1)).unwrap().max_hp;
		let contact_hit = Command::MoveAction(MoveCommand {
			move_id: MoveId(0), // tackle, contact
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		let before = hp_at(&battle_state, PositionId(1));
		let after = step(battle_state, vec![contact_hit], &registry, &mut rng).battle_state;
		assert_eq!(before - hp_at(&after, PositionId(1)), attacker_max / 8);

		// A special, non-contact move costs the attacker nothing.
		let (registry, battle_state) = ability_battle(AbilityId::RoughSkin);
		let ranged_hit = Command::MoveAction(MoveCommand {
			move_id: MoveId(4), // mind_beam, no contact
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		let before = hp_at(&battle_state, PositionId(1));
		let after = step(battle_state, vec![ranged_hit], &registry, &mut rng).battle_state;
		assert_eq!(hp_at(&after, PositionId(1)), before);
	}

	/// Guts boosts a statused attacker, and cancels rather than compounds burn.
	#[test]
	fn guts_boosts_when_statused_and_ignores_burn() {
		let mut rng = rand::rng();

		let mut damage_with = |status: NonVolatileStatus| {
			let (registry, mut battle_state) = ability_battle(AbilityId::Guts);
			battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = status;
			let before = hp_at(&battle_state, PositionId(1));
			let attack = Command::MoveAction(MoveCommand {
				move_id: MoveId(0),
				user: PositionId(0),
				targets: vec![PositionId(1)],
			});
			let after = step(battle_state, vec![attack], &registry, &mut rng).battle_state;
			before - hp_at(&after, PositionId(1))
		};

		let healthy = damage_with(NonVolatileStatus::NoStatus);
		let poisoned = damage_with(NonVolatileStatus::Poison);
		let burned = damage_with(NonVolatileStatus::Burn);

		assert!(healthy > 0, "the control hit should do something");

		// x1.5 while statused
		let expected_boost = healthy * 3 / 2;
		assert!(
			poisoned.abs_diff(expected_boost) <= 1,
			"Guts should boost a statused attacker: {} healthy -> {} poisoned, expected about {}",
			healthy, poisoned, expected_boost
		);

		// The point of the ordering trick: burn halves, Guts x3, so a burned Guts
		// attacker lands on the same 1.5x as any other status rather than 0.75x.
		assert!(
			burned.abs_diff(poisoned) <= 1,
			"Guts should cancel the burn drop, not compound it: {} burned vs {} poisoned",
			burned, poisoned
		);
		assert!(burned > healthy, "a burned Guts attacker should still hit harder than a healthy one");
	}

	/// Sand Stream sets weather when its holder arrives.
	#[test]
	fn sand_stream_summons_sand_on_switch_in() {
		let mut rng = rand::rng();
		let registry = registry_with_ability(AbilityId::SandStream);
		let moves = vec![MoveId(0)];

		// Position 0 starts as a plain creature; the Sand Stream holder is benched
		// at roster 2, because there is no battle-start trigger yet.
		let team0 = vec![
			CreatureState::from_species_data(&frail_attacker(), moves.clone()),
			CreatureState::from_species_data(registry.get_species_data(SpeciesId(2)), moves.clone()),
		];
		let team1 = vec![CreatureState::from_species_data(&fat_defender(), moves.clone())];
		let battle_state = BattleState::from(team0, team1, vec![0, 1]);
		assert!(battle_state.weather.is_none());

		let switch = Command::Switch { current: PositionId(0), new: RosterId(2) };
		let after = step(battle_state, vec![switch], &registry, &mut rng).battle_state;

		match after.weather {
			Some(TimedWeather { weather: Weather::Sandstorm, turns_left }) => {
				// One turn of the five has already ticked off at TurnEnd.
				assert_eq!(turns_left, 4);
			}
			other => panic!("expected a sandstorm, got {:?}", other),
		}
	}

	/// Natural Cure clears status on the way out — which also proves the status
	/// veto correctly exempts `NoStatus`, or the cure could never apply.
	#[test]
	fn natural_cure_clears_status_on_switch_out() {
		let mut rng = rand::rng();
		let registry = registry_with_ability(AbilityId::NaturalCure);
		let moves = vec![MoveId(0)];

		let team0 = vec![
			CreatureState::from_species_data(registry.get_species_data(SpeciesId(2)), moves.clone()),
			CreatureState::from_species_data(&fat_defender(), moves.clone()),
		];
		let team1 = vec![CreatureState::from_species_data(&fat_defender(), moves.clone())];
		let mut battle_state = BattleState::from(team0, team1, vec![0, 1]);
		battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Burn;

		let switch = Command::Switch { current: PositionId(0), new: RosterId(2) };
		let after = step(battle_state, vec![switch], &registry, &mut rng).battle_state;

		// roster 0 is the creature that left; it should be clean now.
		assert_eq!(
			after.roster.get_mon(RosterId(0)).unwrap().non_vol_status,
			NonVolatileStatus::NoStatus
		);
	}

	/*******************************
	 *
	 * VOLATILE STATUS:
	 *
	 * Built on the real registry, because these are all about interactions
	 * between a move, a condition and the turn structure.
	 */

	fn live_registry_mon(registry: &Registry, id: u32, moves: Vec<MoveId>) -> CreatureState {
		CreatureState::from_species(registry, SpeciesId(id), moves)
	}

	/// A 1v1 with the given movesets, so a single move can be isolated.
	fn duel(registry: &Registry, a: u32, a_moves: Vec<MoveId>, b: u32, b_moves: Vec<MoveId>) -> BattleState {
		BattleState::from(
			vec![live_registry_mon(registry, a, a_moves)],
			vec![live_registry_mon(registry, b, b_moves)],
			vec![0, 1],
		)
	}

	fn use_move(user: PositionId, target: PositionId, move_id: u32) -> Command {
		Command::MoveAction(MoveCommand {
			move_id: MoveId(move_id),
			user,
			targets: vec![target],
		})
	}

	/// Taunt is the deterministic half of the design: no roll anywhere, and it
	/// gives `TryMove` its first real subscriber.
	#[test]
	fn taunt_blocks_status_moves_but_not_attacks() {
		let registry = Registry::load();
		let mut rng = rand::rng();

		// Taunt lands on its own turn. Doing it in one turn alongside the status
		// move would prove nothing: thornbeast is faster than brackenox, so the
		// seed would resolve before the taunt regardless.
		let battle_state = duel(&registry, 7, vec![MoveId(21)], 5, vec![MoveId(22), MoveId(14)]);
		let taunt = use_move(PositionId(0), PositionId(1), 21);
		let after = step(battle_state, vec![taunt], &registry, &mut rng).battle_state;
		assert!(
			after.get_mon(PositionId(1)).unwrap().volatiles.has(VolatileKind::Taunt),
			"taunt should have landed"
		);

		// Next turn, the taunted creature's status move should fail.
		let seed = use_move(PositionId(1), PositionId(0), 22);
		let after = step(after, vec![seed], &registry, &mut rng).battle_state;
		assert!(
			!after.get_mon(PositionId(0)).unwrap().volatiles.has(VolatileKind::LeechSeed),
			"a taunted creature must not get its status move off"
		);

		// The same creature can still attack.
		let before = after.get_mon(PositionId(0)).unwrap().current_hp;
		let attack = use_move(PositionId(1), PositionId(0), 14);
        let after = step(after, vec![attack], &registry, &mut rng).battle_state;
		assert!(
			after.get_mon(PositionId(0)).unwrap().current_hp < before,
			"taunt only blocks status moves"
		);
	}

	/// Protect blocks an incoming move entirely — damage and riders alike.
	#[test]
	fn protect_blocks_an_incoming_move() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let battle_state = duel(&registry, 3, vec![MoveId(25)], 5, vec![MoveId(14)]);
		let before = battle_state.get_mon(PositionId(0)).unwrap().current_hp;

		let guard = use_move(PositionId(0), PositionId(0), 25);
		let attack = use_move(PositionId(1), PositionId(0), 14);
		let after = step(battle_state, vec![guard, attack], &registry, &mut rng).battle_state;

		assert_eq!(
			after.get_mon(PositionId(0)).unwrap().current_hp,
			before,
			"protect should have taken the hit to zero"
		);
		// And it does not persist: Protect is turn-scoped.
		assert!(!after.get_mon(PositionId(0)).unwrap().volatiles.has(VolatileKind::Protect));
	}

	/// A Substitute soaks attack damage, and chip damage goes straight past it.
	#[test]
	fn substitute_absorbs_attacks_but_not_chip() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let battle_state = duel(&registry, 2, vec![MoveId(24)], 5, vec![MoveId(14)]);
		let max_hp = battle_state.get_mon(PositionId(0)).unwrap().max_hp;

		// Turn 1: put a Substitute up. It costs a quarter of max HP.
		let decoy = use_move(PositionId(0), PositionId(0), 24);
		let after = step(battle_state, vec![decoy], &registry, &mut rng).battle_state;
		let hp_behind_sub = after.get_mon(PositionId(0)).unwrap().current_hp;
		assert_eq!(hp_behind_sub, max_hp - max_hp / 4);
		assert_eq!(
			after.get_mon(PositionId(0)).unwrap().volatiles.value(VolatileKind::Substitute),
			max_hp / 4
		);

		// Turn 2: an attack should hit the decoy, not the creature.
		let attack = use_move(PositionId(1), PositionId(0), 14);
		let after = step(after, vec![attack], &registry, &mut rng).battle_state;
		assert_eq!(
			after.get_mon(PositionId(0)).unwrap().current_hp,
			hp_behind_sub,
			"the substitute should have taken the hit"
		);
	}

	/// Chip damage ignores a Substitute — no `DamageSource`, so nothing to soak.
	#[test]
	fn chip_damage_goes_through_a_substitute() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut battle_state = duel(&registry, 2, vec![MoveId(24)], 5, vec![MoveId(14)]);
		battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Poison;

		let decoy = use_move(PositionId(0), PositionId(0), 24);
		let before = battle_state.get_mon(PositionId(0)).unwrap().current_hp;
		let max_hp = battle_state.get_mon(PositionId(0)).unwrap().max_hp;
		let after = step(battle_state, vec![decoy], &registry, &mut rng).battle_state;

		// Sub cost plus a poison tick, so strictly more than the sub cost alone.
		let lost = before - after.get_mon(PositionId(0)).unwrap().current_hp;
		assert!(
			lost > max_hp / 4,
			"poison should have chipped past the substitute: lost {} vs sub cost {}",
			lost, max_hp / 4
		);
	}

	/// Leech Seed drains the seeded creature and feeds the one that planted it.
	#[test]
	fn leech_seed_transfers_hp_to_the_planter() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut battle_state = duel(&registry, 5, vec![MoveId(22)], 7, vec![MoveId(16)]);
		// Hurt the seeder so the drain has somewhere to go.
		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 50;

		let seed = use_move(PositionId(0), PositionId(1), 22);
		let after = step(battle_state, vec![seed], &registry, &mut rng).battle_state;

		assert!(after.get_mon(PositionId(1)).unwrap().volatiles.has(VolatileKind::LeechSeed));
		assert!(
			after.get_mon(PositionId(0)).unwrap().current_hp > 50,
			"the planter should have been healed by the drain"
		);
	}

	/// Volatiles do not survive leaving the field. This is the entire point of
	/// the volatile / non-volatile split.
	#[test]
	fn switching_out_wipes_volatiles_but_keeps_status() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let moves = vec![MoveId(0)];
		let team0 = vec![
			live_registry_mon(&registry, 5, moves.clone()),
			live_registry_mon(&registry, 7, moves.clone()),
		];
		let team1 = vec![live_registry_mon(&registry, 3, moves.clone())];
		let mut battle_state = BattleState::from(team0, team1, vec![0, 1]);
		{
			let mon = battle_state.get_mut_mon(PositionId(0)).unwrap();
			mon.volatiles.add(Volatile::lasting(VolatileKind::Taunt, 3));
			mon.volatiles.add(Volatile::new(VolatileKind::LeechSeed));
			mon.non_vol_status = NonVolatileStatus::Burn;
		}

		let switch = Command::Switch { current: PositionId(0), new: RosterId(2) };
		let after = step(battle_state, vec![switch], &registry, &mut rng).battle_state;

		let left = after.roster.get_mon(RosterId(0)).unwrap();
		assert!(left.volatiles.is_empty(), "volatiles must not survive a switch");
		assert_eq!(
			left.non_vol_status,
			NonVolatileStatus::Burn,
			"a non-volatile status must survive a switch"
		);
	}

	/// A creature carrying `Immobilised` loses its turn.
	///
	/// Tested through the query rather than through `step`, deliberately: the
	/// flag is meant to be set by a TurnStart hook *during* the turn it applies
	/// to, so pre-setting it and stepping would just measure the end-of-turn
	/// sweep. This asserts the veto itself, which is the deterministic half of
	/// confusion and full paralysis.
	#[test]
	fn immobilised_costs_the_creature_its_move() {
		let registry = Registry::load();
		let mut battle_state = duel(&registry, 5, vec![MoveId(14)], 3, vec![MoveId(5)]);

		let mut hooks = HookTable::new();
		hooks.refresh(&battle_state, &registry);
		assert!(
			hooks.allows_move(&battle_state, &registry, PositionId(0), MoveId(14)),
			"a healthy creature should be allowed to move"
		);

		battle_state
			.get_mut_mon(PositionId(0))
			.unwrap()
			.volatiles
			.add(Volatile::new(VolatileKind::Immobilised));
		hooks.invalidate();
		hooks.refresh(&battle_state, &registry);

		assert!(
			!hooks.allows_move(&battle_state, &registry, PositionId(0), MoveId(14)),
			"an immobilised creature must not be allowed to move"
		);
	}

	/// Turn-scoped volatiles are swept at the start of the next turn, so a flinch
	/// cannot silently block two turns in a row.
	#[test]
	fn turn_scoped_volatiles_do_not_survive_into_the_next_turn() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut battle_state = duel(&registry, 5, vec![MoveId(14)], 3, vec![MoveId(5)]);
		battle_state
			.get_mut_mon(PositionId(0))
			.unwrap()
			.volatiles
			.add(Volatile::new(VolatileKind::Flinch));

		let attack = use_move(PositionId(0), PositionId(1), 14);
		let after = step(battle_state, vec![attack], &registry, &mut rng).battle_state;
		// Swept during this turn's TurnStart, so it is gone whether or not it bit.
		assert!(!after.get_mon(PositionId(0)).unwrap().volatiles.has(VolatileKind::Flinch));
	}

	/// Timed volatiles count down and expire on their own.
	#[test]
	fn timed_volatiles_expire() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut battle_state = duel(&registry, 5, vec![MoveId(14)], 3, vec![MoveId(5)]);
		battle_state
			.get_mut_mon(PositionId(0))
			.unwrap()
			.volatiles
			.add(Volatile::lasting(VolatileKind::Taunt, 2));

		let mut state = battle_state;
		for _ in 0..2 {
			state = step(state, vec![], &registry, &mut rng).battle_state;
		}
		assert!(
			!state.get_mon(PositionId(0)).unwrap().volatiles.has(VolatileKind::Taunt),
			"two turns of Taunt should have run out after two turns"
		);
	}

	/*******************************
	 *
	 * RAMPING COUNTERS:
	 *
	 * Protect getting less reliable, and bad poison getting worse. Both keep
	 * their count in a volatile, so both reset on switch-out for free.
	 */

	/// Protect spam must stop working. This is the regression test for the bug
	/// that stalled training: with no failure chance, a creature spamming Protect
	/// could not be touched, and battles ran to the 1000-turn cap.
	#[test]
	fn repeated_protect_starts_failing() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		// stonewarden spams guard; thornbeast attacks every turn.
		let mut state = duel(&registry, 3, vec![MoveId(25)], 5, vec![MoveId(14)]);
		let full_hp = state.get_mon(PositionId(0)).unwrap().current_hp;

		let mut got_through = false;
		for _ in 0..12 {
			let guard = use_move(PositionId(0), PositionId(0), 25);
			let attack = use_move(PositionId(1), PositionId(0), 14);
			state = step(state, vec![guard, attack], &registry, &mut rng).battle_state;
			if state.get_mon(PositionId(0)).unwrap().current_hp < full_hp {
				got_through = true;
				break;
			}
		}
		assert!(
			got_through,
			"twelve consecutive Protects should not all have worked - odds are 100/33/11/3/1/0%"
		);
	}

	/// The chain breaks when the creature does something else, so Protect stays
	/// usable as an occasional tool rather than being permanently spent.
	#[test]
	fn using_another_move_resets_the_protect_streak() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut state = duel(&registry, 3, vec![MoveId(25), MoveId(5)], 5, vec![MoveId(14)]);

		// One Protect: streak goes to 1.
		state = step(state, vec![use_move(PositionId(0), PositionId(0), 25)], &registry, &mut rng).battle_state;
		assert_eq!(
			state.get_mon(PositionId(0)).unwrap().volatiles.value(VolatileKind::ProtectStreak),
			1
		);

		// Then attack instead, and the chain is gone.
		state = step(state, vec![use_move(PositionId(0), PositionId(1), 5)], &registry, &mut rng).battle_state;
		assert!(
			!state.get_mon(PositionId(0)).unwrap().volatiles.has(VolatileKind::ProtectStreak),
			"any other move should break the chain"
		);
	}

	/// Bad poison hurts more every turn it stays in.
	#[test]
	fn bad_poison_ramps_each_turn() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut state = duel(&registry, 7, vec![MoveId(16)], 5, vec![MoveId(14)]);
		state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::BadPoison;
		let max_hp = state.get_mon(PositionId(0)).unwrap().max_hp;

		let mut ticks: Vec<u32> = Vec::new();
		for _ in 0..4 {
			let before = state.get_mon(PositionId(0)).unwrap().current_hp;
			state = step(state, vec![], &registry, &mut rng).battle_state;
			ticks.push(before - state.get_mon(PositionId(0)).unwrap().current_hp);
		}

		assert_eq!(ticks[0], max_hp / 16, "the first tick is a plain 1/16");
		for i in 1..ticks.len() {
			assert!(
				ticks[i] > ticks[i - 1],
				"each tick should hurt more than the last, got {:?}",
				ticks
			);
		}
	}

	/// Switching out resets the ramp — which is the whole reason the counter
	/// lives in a volatile rather than on the status.
	#[test]
	fn switching_out_resets_the_toxic_ramp() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let moves = vec![MoveId(16)];
		let team0 = vec![
			live_registry_mon(&registry, 7, moves.clone()),
			live_registry_mon(&registry, 3, moves.clone()),
		];
		let team1 = vec![live_registry_mon(&registry, 5, moves.clone())];
		let mut state = BattleState::from(team0, team1, vec![0, 1]);
		state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::BadPoison;

		// Let the ramp build for a few turns.
		for _ in 0..3 {
			state = step(state, vec![], &registry, &mut rng).battle_state;
		}
		assert!(state.get_mon(PositionId(0)).unwrap().volatiles.value(VolatileKind::ToxicCounter) > 1);

		let max_hp = state.roster.get_mon(RosterId(0)).unwrap().max_hp;
		let ramped_tick = {
			let before = state.roster.get_mon(RosterId(0)).unwrap().current_hp;
			let after = step(state.clone(), vec![], &registry, &mut rng).battle_state;
			before - after.roster.get_mon(RosterId(0)).unwrap().current_hp
		};
		assert!(ramped_tick > max_hp / 16, "the ramp should be past its first step by now");

		// Switch out, then back in.
		state = step(state, vec![Command::Switch { current: PositionId(0), new: RosterId(2) }], &registry, &mut rng).battle_state;
		let hp_before_return = state.roster.get_mon(RosterId(0)).unwrap().current_hp;
		state = step(state, vec![Command::Switch { current: PositionId(0), new: RosterId(0) }], &registry, &mut rng).battle_state;

		let returning = state.roster.get_mon(RosterId(0)).unwrap();
		assert_eq!(returning.non_vol_status, NonVolatileStatus::BadPoison, "the poison itself persists");
		// The turn it comes back it takes a tick, and that tick is a fresh 1/16
		// rather than a continuation of the old ramp.
		assert_eq!(
			hp_before_return - returning.current_hp,
			max_hp / 16,
			"coming back in should restart the ramp, not resume it (ramped tick was {})",
			ramped_tick
		);
	}

	/// Curing and re-applying restarts the count rather than resuming it.
	#[test]
	fn a_status_change_restarts_the_toxic_ramp() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut state = duel(&registry, 7, vec![MoveId(16)], 5, vec![MoveId(14)]);
		state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::BadPoison;

		for _ in 0..3 {
			state = step(state, vec![], &registry, &mut rng).battle_state;
		}
		assert!(state.get_mon(PositionId(0)).unwrap().volatiles.value(VolatileKind::ToxicCounter) > 1);

		// Cure, then re-poison.
		state.get_mut_mon(PositionId(0)).unwrap().volatiles.remove(VolatileKind::ToxicCounter);
		state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::BadPoison;
		let max_hp = state.get_mon(PositionId(0)).unwrap().max_hp;

		let before = state.get_mon(PositionId(0)).unwrap().current_hp;
		let state = step(state, vec![], &registry, &mut rng).battle_state;
		assert_eq!(
			before - state.get_mon(PositionId(0)).unwrap().current_hp,
			max_hp / 16,
			"a fresh poisoning starts at 1/16 again"
		);
	}

	/*******************************
	 *
	 * REPLACEMENT PHASE:
	 *
	 * The game window used to throw `step_request` away and feed the player's
	 * replacement switch back in as an ordinary turn action, next to a freely
	 * chosen attack from the opponent — so the creature coming in ate a hit.
	 * These pin the two halves of that.
	 */

	/// Setup: position 0 is about to be knocked out by a big hit.
	fn about_to_faint() -> (Registry, BattleState, Command) {
		let registry = test_registry();
		let mut battle_state = test_battle_state();
		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 1;
		let finisher = Command::MoveAction(MoveCommand {
			move_id: MoveId(2), // big_damage
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		(registry, battle_state, finisher)
	}

	/// A replacement step carries no attacks, so the incoming creature arrives
	/// untouched. This is the behaviour the game window now implements.
	#[test]
	fn a_replacement_step_does_not_hit_the_incoming_creature() {
		let mut rng = rand::rng();
		let (registry, battle_state, finisher) = about_to_faint();

		let StepResult { battle_state, step_request } =
			step(battle_state, vec![finisher], &registry, &mut rng);
		assert_eq!(step_request, StepRequest::NeedsReplacements(vec![PositionId(0)]));

		// The engine asked for a replacement, so the ONLY command it gets back is
		// the switch. Nobody attacks during a replacement.
		let incoming_full_hp = battle_state.roster.get_mon(RosterId(2)).unwrap().max_hp;
		let switch = Command::Switch { current: PositionId(0), new: RosterId(2) };
		let StepResult { battle_state, step_request } =
			step(battle_state, vec![switch], &registry, &mut rng);

		assert_eq!(
			battle_state.get_mon(PositionId(0)).unwrap().current_hp,
			incoming_full_hp,
			"the replacement should not have been hit on the way in"
		);
		assert_eq!(step_request, StepRequest::NeedsActions);
	}

	/// The bug, reproduced: bundling the replacement in with a normal turn lets
	/// the opponent attack into it. Kept as a test so the difference between the
	/// two flows is visible rather than folklore.
	#[test]
	fn bundling_a_replacement_with_a_turn_lets_it_be_hit() {
		let mut rng = rand::rng();
		let (registry, battle_state, finisher) = about_to_faint();

		let StepResult { battle_state, .. } =
			step(battle_state, vec![finisher], &registry, &mut rng);

		let incoming_full_hp = battle_state.roster.get_mon(RosterId(2)).unwrap().max_hp;
		let switch = Command::Switch { current: PositionId(0), new: RosterId(2) };
		let attack = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(1),
			targets: vec![PositionId(0)],
		});
		let StepResult { battle_state, .. } =
			step(battle_state, vec![switch, attack], &registry, &mut rng);

		assert!(
			battle_state.get_mon(PositionId(0)).unwrap().current_hp < incoming_full_hp,
			"this is the old behaviour: a switch submitted as a turn action gets attacked"
		);
	}

	/*******************************
	 *
	 * TYPE CHART:
	 *
	 * These use a purpose-built registry so exactly one thing varies per test.
	 */

	fn typed_species(id: u32, name: &str, typing: Typing) -> SpeciesDatum {
		SpeciesDatum {
			name: String::from(name),
			species_id: SpeciesId(id),
			base_hp: 400, // deep enough that nothing dies mid-measurement
			attack: 100, defense: 100, special_attack: 100, special_defense: 100, speed: 50,
			typing,
			ability: None,
		}
	}

	fn typed_move(id: u32, name: &str, element: Type, effects: Vec<Effect>) -> PMove {
		PMove {
			name: String::from(name),
			move_id: MoveId(id),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			element,
			base_power: 60,
			effects,
			base_prio: 0,
			flags: MoveFlags::NONE,
		}
	}

	/// 0 normal attacker, 1 grass, 2 water, 3 fire attacker, 4 levitating electric.
	fn type_registry() -> Registry {
		Registry {
			species_data: vec![
				typed_species(0, "normal_atk", Typing::mono(Type::Normal)),
				typed_species(1, "grass_def", Typing::mono(Type::Grass)),
				typed_species(2, "water_def", Typing::mono(Type::Water)),
				typed_species(3, "fire_atk", Typing::mono(Type::Fire)),
				SpeciesDatum {
					ability: Some(AbilityId::Levitate),
					..typed_species(4, "floater", Typing::mono(Type::Electric))
				},
			],
			moves: vec![
				typed_move(0, "fire_jab", Type::Fire, vec![]),
				typed_move(1, "ground_jab", Type::Ground, vec![Effect::BurnChance { chance: 100 }]),
			],
		}
	}

	/// One hit from `attacker_species` on `defender_species` with move 0 or 1.
	fn typed_hit(registry: &Registry, attacker: u32, defender: u32, move_id: u32) -> (u32, BattleState) {
		let mut rng = rand::rng();
		let a = CreatureState::from_species(registry, SpeciesId(attacker), vec![MoveId(0), MoveId(1)]);
		let d = CreatureState::from_species(registry, SpeciesId(defender), vec![MoveId(0), MoveId(1)]);
		let battle_state = BattleState::from(vec![a], vec![d], vec![0, 1]);
		let before = hp_at(&battle_state, PositionId(1));
		let command = Command::MoveAction(MoveCommand {
			move_id: MoveId(move_id),
			user: PositionId(0),
			targets: vec![PositionId(1)],
		});
		let after = step(battle_state, vec![command], registry, &mut rng).battle_state;
		(before - hp_at(&after, PositionId(1)), after)
	}

	/// The chart multiplier actually reaches the damage roll.
	#[test]
	fn effectiveness_scales_damage() {
		let registry = type_registry();
		// Fire vs Grass is 2x, Fire vs Water is 0.5x -> a 4x spread.
		let (super_effective, _) = typed_hit(&registry, 0, 1, 0);
		let (resisted, _) = typed_hit(&registry, 0, 2, 0);

		assert!(resisted > 0, "a resisted hit should still do something");
		let ratio = super_effective as f32 / resisted as f32;
		assert!(
			(3.5..4.5).contains(&ratio),
			"expected about a 4x spread between 2x and 0.5x, got {}/{} = {:.2}",
			super_effective, resisted, ratio
		);
	}

	/// STAB is applied, and only to an attacker whose type matches.
	#[test]
	fn stab_applies_to_matching_types_only() {
		let registry = type_registry();
		// Identical stats and identical move; only the attacker's own type differs.
		let (no_stab, _) = typed_hit(&registry, 0, 1, 0);
		let (with_stab, _) = typed_hit(&registry, 3, 1, 0);

		let ratio = with_stab as f32 / no_stab as f32;
		assert!(
			(1.4..1.6).contains(&ratio),
			"expected about x1.5 from STAB, got {}/{} = {:.2}",
			with_stab, no_stab, ratio
		);
	}

	/// An immunity from an ability blocks the damage AND the secondary effect.
	///
	/// This is why effectiveness is its own query rather than a damage multiplier:
	/// zeroing the damage alone would still have let the burn land.
	#[test]
	fn levitate_blocks_ground_damage_and_its_rider() {
		let registry = type_registry();
		// move 1 is Ground and burns 100% of the time it connects.
		let (damage, after) = typed_hit(&registry, 0, 4, 1);

		assert_eq!(damage, 0, "Levitate should have made the Ground move do nothing");
		assert_eq!(
			after.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::NoStatus,
			"a move that was nullified must not still apply its secondary effect"
		);

		// The same rider lands fine on something that is not immune.
		let (_, after) = typed_hit(&registry, 0, 1, 1);
		assert_eq!(
			after.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::Burn
		);
	}

	/// A chart immunity (not an ability) blocks a move just as completely.
	#[test]
	fn chart_immunity_blocks_a_move_entirely() {
		let mut registry = type_registry();
		// Make the defender Flying, which is immune to Ground on the chart alone.
		registry.species_data[1].typing = Typing::mono(Type::Flying);

		let (damage, after) = typed_hit(&registry, 0, 1, 1);
		assert_eq!(damage, 0);
		assert_eq!(
			after.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::NoStatus
		);
	}

	/// A creature with two types multiplies both halves.
	#[test]
	fn dual_types_stack_in_battle() {
		let mut registry = type_registry();
		let (single, _) = typed_hit(&registry, 0, 1, 0); // Fire vs Grass = 2x
		registry.species_data[1].typing = Typing::dual(Type::Grass, Type::Steel);
		let (dual, _) = typed_hit(&registry, 0, 1, 0); // Fire vs Grass/Steel = 4x

		let ratio = dual as f32 / single as f32;
		assert!(
			(1.8..2.2).contains(&ratio),
			"Grass/Steel should take twice what Grass does from Fire, got {}/{} = {:.2}",
			dual, single, ratio
		);
	}

	/*******************************
	 *
	 * PHYSICAL / SPECIAL SPLIT:
	 *
	 */

	/// A special move scales off Special Attack vs Special Defense, so bending
	/// the special stats changes its damage and bending Attack does not.
	#[test]
	fn special_moves_use_the_special_stats() {
		let mut rng = rand::rng();
		let mut registry = test_registry();
		// Make the special spread differ sharply from the physical one, but keep
		// the result under the target's 120 max HP so the assert measures damage
		// rather than the target's remaining health.
		registry.species_data[0].special_attack = 200;
		registry.species_data[1].special_defense = 100;

		let battle_state = test_battle_state();
		let before = hp_at(&battle_state, PositionId(1));
		let special = Command::MoveAction(MoveCommand {
			move_id: MoveId(4), // mind_beam
			user: PositionId(0),
			targets: vec![PositionId(1)],
		});
		let after = step(battle_state, vec![special], &registry, &mut rng).battle_state;

		let special_damage = before - hp_at(&after, PositionId(1));

		// The same creature's physical hit, for comparison: it reads Attack 100
		// against Defense 80, untouched by the special stats bent above.
		let physical_damage = {
			let battle_state = test_battle_state();
			let before = hp_at(&battle_state, PositionId(1));
			let after = step(battle_state, vec![frail_uses_tackle()], &registry, &mut rng).battle_state;
			before - hp_at(&after, PositionId(1))
		};

		assert!(
			special_damage > physical_damage,
			"special move read the wrong stats: {} special vs {} physical, with SpA 200 and SpD 100",
			special_damage, physical_damage
		);
	}

	/// A burn no longer weakens special attackers.
	#[test]
	fn burn_does_not_weaken_special_moves() {
		let mut rng = rand::rng();
		let registry = test_registry();

		let mut damage = |burned: bool| {
			let mut battle_state = test_battle_state();
			if burned {
				battle_state.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Burn;
			}
			let before = hp_at(&battle_state, PositionId(1));
			let special = Command::MoveAction(MoveCommand {
				move_id: MoveId(4),
				user: PositionId(0),
				targets: vec![PositionId(1)],
			});
			let after = step(battle_state, vec![special], &registry, &mut rng).battle_state;
			before - hp_at(&after, PositionId(1))
		};

		assert_eq!(damage(false), damage(true));
	}

	/// A status move applies its effect and deals no damage at all.
	#[test]
	fn status_moves_deal_no_damage() {
		let mut rng = rand::rng();
		let registry = test_registry();
		let battle_state = test_battle_state();
		let before = hp_at(&battle_state, PositionId(1));

		let glare = Command::MoveAction(MoveCommand {
			move_id: MoveId(5), // hex_glare, always paralyses
			user: PositionId(0),
			targets: vec![PositionId(1)],
		});
		let after = step(battle_state, vec![glare], &registry, &mut rng).battle_state;

		assert_eq!(hp_at(&after, PositionId(1)), before, "status moves must not chip");
		assert_eq!(
			after.get_mon(PositionId(1)).unwrap().non_vol_status,
			NonVolatileStatus::Paralysis
		);
	}

	/// Paralysis halves Speed, which flips the turn order between two creatures
	/// that were close in speed.
	#[test]
	fn paralysis_halves_speed() {
		let registry = test_registry();
		let battle_state = {
			let mut bs = test_battle_state();
			bs.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::Paralysis;
			bs
		};
		let mut hooks = HookTable::new();
		hooks.refresh(&battle_state, &registry);

		let raw = battle_state.get_mon(PositionId(0)).unwrap().get_stat(Stat::Speed, &registry);
		let effective = hooks.effective_stat(&battle_state, &registry, PositionId(0), Stat::Speed, raw);

		assert_eq!(raw, 90);
		assert_eq!(effective, 45);
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

	/// A fainted creature stays fainted. Healing may not revive it.
	#[test]
	fn healing_cannot_revive_a_fainted_creature() {
		let registry = Registry::load();
		let mut battle = BattleState::from(
			vec![CreatureState::from_species(&registry, SpeciesId(2), vec![MoveId(3)])],
			vec![CreatureState::from_species(&registry, SpeciesId(5), vec![MoveId(14)])],
			vec![0, 1],
		);
		let pos = PositionId(0);
		battle.get_mut_mon(pos).unwrap().current_hp = 0;

		let healed = apply_healing(50, pos, &mut battle);

		assert_eq!(healed, 0, "a fainted creature must not be healed");
		assert_eq!(battle.get_mon(pos).unwrap().current_hp, 0);
	}

	/// The invariant the replacement phase depends on: every position the engine
	/// asks to have replaced is actually holding a fainted creature.
	///
	/// This is the regression test for a real bug. A creature knocked to 0 during
	/// the action phase could be healed back above 0 later in the same step — a
	/// Leech Seed drain heals the *seeder*, who carries no volatile of their own —
	/// while the faint stayed recorded. The engine then asked for a replacement
	/// for a living creature. The mask correctly allows moves for a creature that
	/// is alive, so the agent answered a replacement request with an attack and
	/// the turn silently went wrong. It fired in roughly 8% of battles.
	#[test]
	fn replacements_are_only_ever_requested_for_fainted_creatures() {
		use crate::rl::agent::{Agent, random_agent::RandomAgent};
		use crate::rl::mask::Mask;

		let registry = Registry::load();
		let mut rng = rand::rng();
		// Leech Seed on both sides, which is what surfaced the bug.
		let team = |ids: [(u32, [u32; 4]); 3]| -> Vec<CreatureState> {
			ids.iter()
				.map(|(sid, mv)| CreatureState::from_species(
					&registry, SpeciesId(*sid), mv.iter().map(|m| MoveId(*m)).collect()))
				.collect()
		};

		for _ in 0..400 {
			let mut battle = BattleState::from(
				team([(2, [3, 5, 13, 24]), (3, [5, 6, 16, 25]), (4, [12, 15, 9, 22])]),
				team([(5, [14, 6, 20, 22]), (6, [10, 12, 19, 4]), (7, [16, 6, 5, 26])]),
				vec![0, 1],
			);
			let mut request = StepRequest::NeedsActions;
			let mut agent = RandomAgent {};

			for _ in 0..300 {
				let positions: Vec<PositionId> = match request.clone() {
					StepRequest::Finished(_) => break,
					StepRequest::NeedsReplacements(positions) => {
						for pos in &positions {
							let mon = battle.get_mon(*pos).expect("a creature at the position");
							assert_eq!(
								mon.current_hp, 0,
								"replacement requested for {:?}, which holds a creature on {}/{} hp",
								pos, mon.current_hp, mon.max_hp,
							);
						}
						positions
					}
					StepRequest::NeedsActions => [Team::Zero, Team::One]
						.into_iter()
						.map(|team| battle.field.team_positions(&team)[0])
						.collect(),
				};

				let encoding = vec![0.0f32; crate::rl::encoder::TOTAL_ENCODING_LEN];
				let commands: Vec<Command> = positions
					.into_iter()
					.map(|pos| {
						let mask = Mask::from_battle_state(&pos.team(), pos, &battle);
						agent.choose_move(&encoding, &mask, &mut rng).to_command(pos, &battle, &registry)
					})
					.collect();

				StepResult { battle_state: battle, step_request: request } =
					step(battle, commands, &registry, &mut rng);
			}
		}
	}
}
