use rand::Rng;

use crate::{battle::{command::Command, state::{BattleState, Outcome}}, model::registry::Registry};
use crate::battle::engine;

pub fn step(battle_state: BattleState, actions: Vec<Command>, registry: &Registry, rng: &mut impl Rng) -> (BattleState, f32, bool) {

	let next_state = engine::step(battle_state, actions, registry, rng);

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