use crate::battle::state::BattleState;
use crate::battle::command::Command;
use crate::battle::event::Event;
use crate::battle::state::PokemonState;
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;

pub fn step(mut battle_state: BattleState, commands: Vec<Command>, registry: &Registry) -> BattleState {
	let mut events: Vec<Event> = commands.into_iter().map(|x| Event::CommandEvent(x)).collect();
	while let Some(event) = events.pop() {
		match event {
			Event::CommandEvent(command) => {
				match command {
					Command::MoveAction(move_command) => {
						let mv = registry.get_move(move_command.move_id);
						let user: &PokemonState = battle_state.mons.get(&move_command.user).unwrap();
						for target_pos in move_command.targets {
							events.push(Event::DealDamage {
								amount: user.get_stat(Stat::Attack, registry) * mv.base_power,
								target: target_pos
							});
						}
					},
					Command::Switch => todo!(),
				}
			},
			Event::DealDamage { amount, target} => {
				let target_state: &mut PokemonState = battle_state.mons.get_mut(&target).unwrap();
				let defense = target_state.get_stat(Stat::Defense, &registry);
				let pure_damage = amount / defense;
				println!("Dealing {} damage to {} hp", pure_damage, target_state.current_hp);
				target_state.current_hp = target_state.current_hp.saturating_sub(pure_damage);
				if target_state.current_hp == 0 {
					println!("TODO: FLAG ANY DEATH");
				}

			}
		}
	}

	battle_state
}



#[cfg(test)]
mod tests {
	use std::collections::HashMap;

use crate::{battle::{self, command::MoveCommand, state::PositionId}, model::{pmove::MoveId, speciesdata::SpeciesId}};

	use super::*;

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
