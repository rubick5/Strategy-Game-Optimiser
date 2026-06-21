use crate::battle::state::BattleState;
use crate::battle::command::Command;
use crate::battle::event::Event;

pub fn step(battle_state: BattleState, commands: Vec<Command>) -> BattleState {
	let mut events: Vec<Event> = commands.into_iter().map(|x| Event::CommandEvent(x)).collect();
	for event in events {
		match event {
			Event::CommandEvent(command) => {
			},
			Event::DealDamage { amount, target} => todo!()
		}
	}

	battle_state
}