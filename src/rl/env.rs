/*/
use rand::Rng;

use crate::{battle::{command::Command, engine::{StepRequest, StepResult}, state::{BattleState, Outcome::{self, Side0Wins}}}, model::registry::Registry};
use crate::battle::engine;
pub fn step(old_battle_state: BattleState, actions: Vec<Command>, registry: &Registry, rng: &mut impl Rng) -> (BattleState, ) {

	let StepResult { battle_state, step_request } = engine::step(old_battle_state, actions, registry, rng);

	let finished: bool;
	let reward = match step_request {
		StepRequest::Finished(Outcome::Side0Wins) => {
			finished = true;
			1.0
		},
		StepRequest::Finished(Outcome::Side1Wins) => {
			finished = true;
			-1.0
		}
		StepRequest => {
			finished = true;
			0.0
		},
		None => {
			finished = false;
			0.0
		}
	};
	return (battle_state, reward, finished);
} */