use crate::{battle::{command::{Command::{self, MoveAction}, MoveCommand}, state::{BattleState, Outcome, PositionId}}, model::{pmove::MoveId, registry::Registry}};
use crate::battle::engine;

fn step_against_random(battle_state: BattleState, action: Command, registry: &Registry) -> (BattleState, f32, bool) {
	let random_command = MoveAction(
		MoveCommand {
			move_id: MoveId(0),
			user: PositionId(1),
			targets: vec![PositionId(0)],
		}
	);


	let next_state = engine::step(battle_state, vec![action, random_command] ,registry);
	let finished: bool;
	let reward = match next_state.outcome() {
		Some(Outcome::Side0Wins) => {
			finished = true;
			1.0
		},
		Some(Outcome::Side1Wins) => {
			finished = true;
			-1.0
		}
		Some(Outcome::Draw) => {
			finished = true;
			0.0
		},
		None => {
			finished = false;
			0.0
		}
	};
	return (next_state, reward, finished);
}