use crate::battle::state::BattleState;
use crate::battle::command::Command;
use crate::battle::event::Event;
use crate::battle::state::PokemonState;
use crate::model::registry::Registry;

pub fn step(mut battle_state: BattleState, commands: Vec<Command>, registry: &Registry) -> BattleState {
	let mut events: Vec<Event> = commands.into_iter().map(|x| Event::CommandEvent(x)).collect();
	for event in events {
		match event {
			Event::CommandEvent(command) => {
				match command {
					Command::MoveAction(move_command) => {
						let mv = registry.get(move_command.move_id);
					},
					Command::Switch => todo!(),
				}
			},
			Event::DealDamage { amount, target} => {
				let target_state: &mut PokemonState = battle_state.mons.get_mut(&target).unwrap();
				target_state.current_hp -= amount;
			}
		}
	}

	battle_state
}