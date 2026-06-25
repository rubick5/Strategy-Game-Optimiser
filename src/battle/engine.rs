use crate::battle::state::{BattleState, PositionId};
use crate::battle::command::Command;
use crate::battle::event::Event;
use crate::battle::state::PokemonState;
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::{SpeciesData, Stat};

const SWITCHING_PRIO: i8 = 9;

fn get_next_command(commands: &mut Vec<Command>, registry: &Registry, battle_state: &BattleState) -> Option<Command> {
	if commands.is_empty() {
		return None;
	}
	// we need the thing with the highest priority and the highest speed!
	let mut highest_prio = -10;
	let mut highest_speed = 0;
	let mut highest_command_index: Option<usize> = None;
	for (index, command) in commands.iter().enumerate() {
		let (prio, speed) = match command {
			Command::MoveAction(move_command) => {
				let mv = registry.get_move(move_command.move_id);
				let mon = battle_state.mons.get(&move_command.user).unwrap();
				let prio = (mv.calc_prio)(&battle_state, move_command.user);
				let speed = mon.get_stat(Stat::Speed, &registry);
				(prio, speed)
			}
			Command::Switch(pos) => {
				let speed = battle_state.mons.get(&pos).unwrap().get_stat(Stat::Speed, &registry);
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
		}
	}

	if let Some(i) = highest_command_index {
		return Some(commands.remove(i));
	} else {
		return None;
	}
	
}

pub fn step(mut battle_state: BattleState, commands: Vec<Command>, registry: &Registry) -> BattleState {
	let mut commands: Vec<Command> = commands.clone();
	let mut events: Vec<Event> = Vec::new();
	loop {
		match events.pop() {
			Some(Event::DealDamage { amount, target}) => {
				let target_state: &mut PokemonState = battle_state.mons.get_mut(&target).unwrap();
				let defense = target_state.get_stat(Stat::Defense, &registry);
				let pure_damage = amount / defense;
				println!("Dealing {} damage to {} hp", pure_damage, target_state.current_hp);
				target_state.current_hp = target_state.current_hp.saturating_sub(pure_damage);
				if target_state.current_hp == 0 {
					println!("TODO: FLAG ANY DEATH");
				}
			}

			Some(Event::Switch { pos }) => println!("switch, position {}", pos),
			None => {
				// this is where we will handle our commands (there are no events to
				// deal with atm!!)
				match get_next_command(&mut commands, registry, &battle_state) {
					Some(Command::MoveAction(move_command)) => {
						let mv = registry.get_move(move_command.move_id);
						let user: &PokemonState = battle_state.mons.get(&move_command.user).unwrap();

						let user_attack = user.get_stat(Stat::Attack, registry);

						for target_pos in move_command.targets {
							let target: &PokemonState = battle_state.mons.get(&move_command.user).unwrap();
							let target_defense =  target.get_stat(Stat::Defense, registry);

							log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

							events.push(Event::DealDamage {
								amount: calculate_damage(user_attack, target_defense, mv.base_power),
								target: target_pos
							});
						}
					},
					Some(Command::Switch(pos)) => 
						events.push(Event::Switch { pos }),
					None => break
				}
			}
		}
	}

	battle_state
}

fn calculate_damage(attack_stat: u32, defense_stat: u32, base_power: u32) -> u32 {
	attack_stat * base_power / defense_stat
}

fn log_move_usage(battle_state: &BattleState, registry: &Registry, user: PositionId, target: PositionId, mv: MoveId) {
	let user_name = get_species_data(battle_state, registry, user).name;
	let target_name = get_species_data(battle_state, registry, target).name;

	let move_name = &registry.get_move(mv).name;
	println!("{} used {} on {}", user_name, move_name, target_name);
}

fn get_species_data(battle_state: &BattleState, registry: &Registry, mon: PositionId) -> SpeciesData {
	registry.get_pokemon(battle_state.mons.get(&mon).unwrap().species_id).clone()
}



#[cfg(test)]
mod tests {
	use std::collections::HashMap;

use crate::{battle::{command::MoveCommand, state::PositionId}, model::{pmove::MoveId, speciesdata::SpeciesId}};

	use super::*;

	#[test]
	fn faster_speed_goes_first() {
		let registry = Registry::load();
		let mons: HashMap<PositionId, PokemonState> = HashMap::from([
			(PositionId(0), PokemonState::from_species(&registry, SpeciesId(0))),
			(PositionId(1), PokemonState::from_species(&registry, SpeciesId(1))),
		]);
		let battle_state = BattleState {
			mons: mons,
		};

		let command1 = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(0),
			targets: vec![PositionId(1)]
		});
		let command1_copy = command1.clone();

		let command2 = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(1),
			targets: vec![PositionId(0)]
		});
		let command2_copy = command2.clone();

		let mut v = vec![command2, command1];

		assert!(get_next_command(&mut v, &registry, &battle_state) == Some(command1_copy));
		assert!(get_next_command(&mut v, &registry, &battle_state) == Some(command2_copy));

	}

	#[test]
	fn example_step() {
		let registry = Registry::load();
		let mons: HashMap<PositionId, PokemonState> = HashMap::from([
			(PositionId(0), PokemonState::from_species(&registry, SpeciesId(0))),
			(PositionId(1), PokemonState::from_species(&registry, SpeciesId(1))),
		]);
		let battle_state = BattleState {
			mons: mons,
		};

		let command1 = Command::MoveAction(MoveCommand {
			move_id: MoveId(0),
			user: PositionId(0),
			targets: vec![PositionId(1)]
		});

		let new_state = step(battle_state, vec![command1], &registry);

		println!("{:?}", new_state);
	}
}
