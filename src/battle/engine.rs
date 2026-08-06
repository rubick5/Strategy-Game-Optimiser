use rand::Rng;

use crate::battle::state::{BattleState, Outcome, PositionId, RosterId};
use crate::battle::command::{Command, MoveCommand};
use crate::battle::event::Event;
use crate::battle::state::CreatureState;
use crate::model::pmove::MoveId;
use crate::model::pmove::MoveType;
use crate::model::registry::Registry;
use crate::model::speciesdata::{SpeciesDatum, Stat};

const SWITCHING_PRIO: i8 = 9;
const MIN_PRIORITY: i8 = -7;
const MIN_SPEED: u32 = 0;

fn get_next_command(commands: &mut Vec<Command>, registry: &Registry, battle_state: &BattleState, rng: &mut impl Rng) -> Option<Command> {
	if commands.is_empty() {
		return None;
	}
	// we need the thing with the highest priority and the highest speed!
	let mut highest_prio = MIN_PRIORITY;
	let mut highest_speed = MIN_SPEED;
	let mut highest_command_index: Option<usize> = None;
	for (index, command) in commands.iter().enumerate() {
		let (prio, speed) = match command {
			Command::MoveAction(move_command) => {
				let mv = registry.get_move(move_command.move_id);
				let creature = battle_state.get_mon(move_command.user).unwrap();
				if creature.current_hp <= 0 {
					continue;
				}
				let prio = mv.base_prio;
				let speed = creature.get_stat(Stat::Speed, &registry);
				(prio, speed)
			}
			Command::Switch {current, new: _} => {
				let speed = battle_state.get_mon(*current).unwrap().get_stat(Stat::Speed, &registry);
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
		None
	}
}

fn execute_move(move_command: MoveCommand, registry: &Registry, battle_state: &BattleState, events: &mut Vec<Event>) {
	let mv = registry.get_move(move_command.move_id);
	let user: &CreatureState = battle_state.get_mon(move_command.user).unwrap();

	let user_attack = user.get_stat(Stat::Attack, registry);

	for target_pos in move_command.targets {
		let target: &CreatureState = battle_state.get_mon(target_pos).unwrap();
		let target_defense =  target.get_stat(Stat::Defense, registry);

		//log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

		events.push(Event::DealDamage {
			amount: calculate_damage(user_attack, target_defense, mv.base_power),
			target: target_pos
		});
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

pub enum StepRequest {
	NeedsActions,
	NeedsReplacements(Vec<PositionId>),
	Finished(Outcome),
}

pub struct StepResult {
	pub battle_state: BattleState,
	pub step_request: StepRequest,
}

pub fn step(mut battle_state: BattleState, commands: Vec<Command>, registry: &Registry, rng: &mut impl Rng) -> StepResult {
	let mut commands: Vec<Command> = commands.clone();
	let mut events: Vec<Event> = Vec::new();
	let mut fainted: Vec<PositionId> = vec![];
	loop {
		match events.pop() {
			Some(Event::DealDamage { amount, target}) => {
				if let Some(pos) = handle_damage_event(amount, target, &mut battle_state) {
					fainted.push(pos);
				}
			}

			Some(Event::Switch { current, new }) => {
				handle_switch_event(current, new, &mut battle_state);
			},
			None => {
				// this is where we will handle our commands (there are no events to
				// deal with atm!!)
				match get_next_command(&mut commands, registry, &battle_state, rng) {
					Some(Command::MoveAction(move_command)) => {
						execute_move(move_command, registry, &battle_state, &mut events);
					},
					Some(Command::Switch {current, new}) => 
						events.push(Event::Switch { current, new }),
					None => break
				}
			}
		}
	}
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

fn calculate_damage(attack_stat: u32, defense_stat: u32, base_power: u32) -> u32 {
	attack_stat * base_power / defense_stat
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
	use crate::{battle::{command::MoveCommand, state::PositionId}, model::{pmove::{MoveId, MoveTargeting, MoveType, PMove}, speciesdata::SpeciesId}};
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
			CreatureState::from_species_data(frail_attacker(), vec![MoveId(0), MoveId(1)]),
		];
		let team1: Vec<CreatureState> = vec![
			CreatureState::from_species_data(fat_defender(), vec![MoveId(0), MoveId(1)]),
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
		use crate::battle::state::RosterId;

		let registry = test_registry();
		let mut rng = rand::rng();

		// team0 has TWO mons: frail_attacker (roster 0) and fat_defender (roster 2).
		// team1 has one creature (roster 1). field starts pointing position 0 -> roster 0.
		let team0 = vec![
			CreatureState::from_species_data(frail_attacker(), vec![MoveId(0), MoveId(1)]),
			CreatureState::from_species_data(fat_defender(), vec![MoveId(0), MoveId(1)]),
		];
		let team1 = vec![
			CreatureState::from_species_data(fat_defender(), vec![MoveId(0), MoveId(1)]),
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
		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 0;
		let attack_command = Command::MoveAction(MoveCommand {
			move_id: MoveId(2),
			targets: vec![PositionId(1)],
			user: PositionId(0),
		});
		let step_request = step(battle_state, vec![attack_command], &test_registry(), &mut rng).step_request;
		assert!(matches!(step_request, StepRequest::NeedsReplacements(_)));
	}
}
