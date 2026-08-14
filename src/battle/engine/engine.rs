use rand::{Rng, RngCore};

use crate::battle::engine::execute_move::execute_move;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::battle::state::field::PositionId;
use crate::battle::state::roster::RosterId;
use crate::battle::state::Outcome;
use crate::battle::command::Command;
use crate::battle::event::Event;
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::{SpeciesDatum, Stat};

use crate::battle::state::non_volatile_status::NonVolatileStatus;

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

fn handle_damage_event(amount: u32, target: PositionId, battle_state: &mut BattleState) -> Option<PositionId> {
	let target_state: &mut CreatureState = battle_state.get_mut_mon(target).unwrap();
	target_state.current_hp = target_state.current_hp.saturating_sub(amount);
	match target_state.current_hp {
		0 => Some(target),
		_ => None
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

pub fn step(mut battle_state: BattleState, mut commands: Vec<Command>, registry: &Registry, rng: &mut dyn RngCore) -> StepResult {
	let mut events: Vec<Event> = Vec::new();
	let mut fainted: Vec<PositionId> = Vec::new();
	let mut non_vol_status_handled = false;
	loop {
		match events.pop() {
			Some(Event::DealDamage { amount, target}) => {
				//println!("dealing {} to {:?}", amount, target);
				if let Some(pos) = handle_damage_event(amount, target, &mut battle_state) {
					fainted.push(pos);
				}
			}

			Some(Event::Switch { current, new }) => {
				handle_switch_event(current, new, &mut battle_state);
			},

			Some(Event::ApplyNonVolStatus { status, target }) => {
				battle_state.get_mut_mon(target).unwrap().non_vol_status = status;
			}
			None => {
				// this is where we will handle our commands (there are no events to
				// deal with atm!!)
				match get_next_command(&mut commands, registry, &battle_state, rng) {
					Some(Command::MoveAction(move_command)) => {
						//println!("executing move: {:?}", move_command);
						execute_move(move_command, registry, &battle_state, &mut events, rng);
					},
					Some(Command::Switch {current, new}) => 
						events.push(Event::Switch { current, new }),
					None => {
						if !non_vol_status_handled {
							queue_non_volatile_status(&mut battle_state, &mut events);
							non_vol_status_handled = true;
						} else {
							break;
						}
					}
				}
			}
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

fn queue_non_volatile_status(bs: &mut BattleState, queue: &mut Vec<Event>) {
	for pos in bs.field.all_field_positions() {
		let mut_mon = bs.get_mut_mon(pos);
		if let Some(m) = mut_mon {
			match m.non_vol_status {
				NonVolatileStatus::NoStatus => {},
				NonVolatileStatus::Poison => {
					queue.push(Event::DealDamage { amount: m.max_hp / 8, target: pos });
				},
				NonVolatileStatus::BadPoison => {
					queue.push(Event::DealDamage { amount: m.max_hp / 16, target: pos });
				},
				NonVolatileStatus::Burn => {
					queue.push(Event::DealDamage { amount: m.max_hp / 16, target: pos });
				},
			}
		}
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
}
