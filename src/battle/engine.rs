use crate::battle::state::BattleState;
use crate::battle::command::Command;

pub fn step(battle_state: BattleState, commands: Vec<Command>) -> BattleState {
	for command in commands {
		match command {
			Command::MoveAction(move_action) => todo!(),
			Command::Switch => todo!()
		}
	}

	battle_state
}