//! Letting the PPO prober loose on the solver's strategy.
//!
//! # Why bother, given there is an exact best response already
//!
//! [`crate::cfr::exploit`] computes the best response directly, which is stronger
//! than anything a learner will find — but only *within the horizon it can
//! afford*. On a 2v2 that is five or six turns; on anything larger it is two or
//! three. Whatever a strategy gets wrong beyond that point, the exact measurement
//! cannot see, because it stops looking at the same place the strategy does.
//!
//! [`crate::rl::exploit`] has the opposite shape. A PPO prober plays *whole
//! battles* against a frozen opponent and is rewarded only for winning them, so
//! nothing is beyond its horizon — it has none. It will not find the best
//! counter-strategy, and it proves nothing when it fails. But when it succeeds it
//! has found a real hole, in a part of the game no depth-limited measurement was
//! ever going to reach.
//!
//! So the two are complementary, and neither replaces the other: one is a tight
//! bound on a short game, the other a loose bound on the whole one.
//!
//! # The awkward part, stated plainly
//!
//! [`Agent`] receives an *encoding*, not a battle state, because that is all a
//! network needs. A solver needs the state — it has to search from it. So a
//! solver cannot implement `Agent` directly, and the strategy has to be worked
//! out in advance and looked up by encoding.
//!
//! That leaves positions the prober steers into that were never precomputed.
//! [`CfrAgent`] plays randomly at those and counts them, and the hit rate is
//! reported beside the result, because the two readings mean different things: a
//! prober beating a well-covered agent has found a hole in the strategy, while
//! one beating a poorly covered agent may only have found the gaps in the
//! lookup. The fix is more coverage, not a different reading of the number.

use std::cell::Cell;
use std::collections::HashMap;

use rand::RngCore;

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::cfr::infoset::{sample as sample_action, Strategy};
use crate::cfr::node::DecisionNode;
use crate::cfr::resolve::ResolvingPolicy;
use crate::model::registry::Registry;
use crate::rl::agent::Agent;
use crate::rl::encoder::encode;
use crate::rl::mask::Mask;
use crate::rl::moveslot::Moveslot;
use crate::cfr::solver::PLAYOUT_CAP;

/// A solver's strategy, in a form the PPO machinery can play against.
pub struct CfrAgent {
	/// Encoding bits to strategy. `f32` is not hashable and does not need to be:
	/// the same position always produces the same bits.
	table: HashMap<Vec<u32>, Strategy>,
	hits: Cell<u64>,
	misses: Cell<u64>,
}

impl CfrAgent {
	/// Positions answered from the table, as a share of all questions.
	///
	/// Read the probe's result next to this. A low figure means the prober spent
	/// its time in positions where this agent was playing at random, and beating
	/// that says little about the strategy.
	pub fn hit_rate(&self) -> f32 {
		let total = self.hits.get() + self.misses.get();
		if total == 0 {
			return 1.0;
		}
		self.hits.get() as f32 / total as f32
	}

	pub fn positions(&self) -> usize {
		self.table.len()
	}

	pub fn reset_counts(&self) {
		self.hits.set(0);
		self.misses.set(0);
	}
}

impl Agent for CfrAgent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		let key: Vec<u32> = representation.iter().map(|value| value.to_bits()).collect();
		match self.table.get(&key) {
			Some(strategy) => {
				self.hits.set(self.hits.get() + 1);
				Moveslot::from_number(sample_action(strategy, rng))
			}
			None => {
				self.misses.set(self.misses.get() + 1);
				mask.get_random_valid(rng)
					.expect("asked for a command with no legal action")
			}
		}
	}
}

/// Work out what the policy plays across the positions a game actually reaches,
/// and index it by encoding.
///
/// `exploration` mixes random play into the walk. Without it the same handful of
/// lines would be covered over and over; the prober will not stay on them, so
/// neither should this.
pub fn precompute(
	registry: &Registry,
	policy: &ResolvingPolicy,
	roots: &[BattleState],
	games: usize,
	exploration: f32,
	rng: &mut dyn RngCore,
) -> CfrAgent {
	let mut table: HashMap<Vec<u32>, Strategy> = HashMap::new();

	for root in roots {
		for _ in 0..games {
			let mut state = root.clone();
			let mut request = StepRequest::NeedsActions;

			for _ in 0..PLAYOUT_CAP {
				let actors = match DecisionNode::from(&state, &request) {
					DecisionNode::Terminal(_) => break,
					DecisionNode::Decision { actors } => actors,
				};

				let strategies = policy.strategies(&state, &request);
				let replacement = matches!(request, StepRequest::NeedsReplacements(_));

				let mut commands = Vec::new();
				for actor in &actors {
					let strategy = strategies
						.iter()
						.find(|(team, _)| *team == actor.team)
						.map(|(_, strategy)| *strategy)
						.unwrap_or_else(|| {
							crate::cfr::infoset::InfosetData::new().average_strategy(&actor.mask)
						});

					// Indexed exactly as `Agent` will see it.
					let view = encode(&state, registry, replacement, &actor.team);
					let key: Vec<u32> = view.iter().map(|value| value.to_bits()).collect();
					table.insert(key, strategy);

					let action = if rand::Rng::random::<f32>(rng) < exploration {
						actor
							.mask
							.get_random_valid(rng)
							.expect("a decision node has a legal action")
							.to_number()
					} else {
						sample_action(&strategy, rng)
					};
					commands.push(actor.command(action, &state, registry));
				}

				let StepResult { battle_state, step_request } =
					engine::step(state, commands, registry, rng);
				state = battle_state;
				request = step_request;
			}
		}
	}

	CfrAgent { table, hits: Cell::new(0), misses: Cell::new(0) }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::switch_prediction_2v2;
	use crate::cfr::solver::SolverConfig;
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	/// A precomputed position must be answered from the table, not at random —
	/// otherwise the probe is measuring the lookup rather than the strategy.
	#[test]
	fn a_precomputed_position_is_answered_from_the_table() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(1);

		let policy = ResolvingPolicy::with_heuristic(
			&registry, SolverConfig::for_lookahead(40, 3), 2);
		let mut agent = precompute(&registry, &policy, &[root.clone()], 3, 0.0, &mut rng);
		assert!(agent.positions() > 0);

		agent.reset_counts();
		let view = encode(&root, &registry, false, &crate::battle::state::Team::Zero);
		let mask = Mask::from_battle_state(
			&crate::battle::state::Team::Zero,
			crate::battle::state::field::PositionId(0),
			&root,
		);
		let chosen = agent.choose_move(&view, &mask, &mut rng);

		assert_eq!(agent.hit_rate(), 1.0, "the root was precomputed and must be known");
		assert!(mask.allowed[chosen.to_number()], "must pick a legal action");
	}

	/// An unknown position is played at random and counted, so the hit rate says
	/// how much of a probe result to believe.
	#[test]
	fn an_unknown_position_is_counted_as_a_miss() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(2);

		let policy = ResolvingPolicy::with_heuristic(
			&registry, SolverConfig::for_lookahead(40, 3), 2);
		let agent = precompute(&registry, &policy, &[root.clone()], 1, 0.0, &mut rng);
		agent.reset_counts();

		let mut agent = agent;
		let mask = Mask::from_battle_state(
			&crate::battle::state::Team::Zero,
			crate::battle::state::field::PositionId(0),
			&root,
		);
		let nonsense = vec![0.5f32; crate::rl::encoder::TOTAL_ENCODING_LEN];
		let chosen = agent.choose_move(&nonsense, &mask, &mut rng);

		assert_eq!(agent.hit_rate(), 0.0);
		assert!(mask.allowed[chosen.to_number()], "a miss still plays legally");
	}
}
