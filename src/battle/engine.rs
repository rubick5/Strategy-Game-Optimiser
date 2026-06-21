use crate::battle::state::BattleState;
use crate::battle::command::Command;

pub fn step(battle_state: BattleState, _commands: Vec<Command>) -> BattleState {
	battle_state
}