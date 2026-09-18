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
//! lookup.
//!
//! [`ResolvingAgent`] avoids the problem. [`Agent::observe`] hands over the
//! position before the encoding is asked about, so the agent can search from it —
//! no precomputation, no gaps, and its horizon travels with it instead of being
//! fixed at turn one.
//!
//! # What the two actually measure
//!
//! Head to head on the 2v2, same search budget, 1,500 measurement battles each:
//!
//! ```text
//!          frozen table        live resolver
//! seed 1   0.529 (80% cover)   0.543
//! seed 2   0.606 (66% cover)   0.557
//! ```
//!
//! Searching live is **not** meaningfully harder to beat. It was expected to be,
//! on the reasoning that a table fixed at turn one has fallen behind by turn ten
//! while a resolver has not. Seed 1 says otherwise, and the two together put both
//! around 0.55.
//!
//! What live resolving does buy is a figure that means something. The frozen
//! table's result swings with how well covered it happens to be — 80% coverage
//! scores 0.529, 66% scores 0.606 — so it measures the harness as much as the
//! strategy. The resolver has no coverage at all to vary, and its two runs land
//! within 0.014 of each other.
//!
//! The reading worth taking: the prober wins about 55% either way, so the
//! strategy really is exploitable over a full battle, and *staleness is not why*.
//! Both agents search with the same shallow horizon and the same leaf estimate,
//! and that estimate is what they are both limited by — which points at the leaf,
//! not at when the searching happens.

use std::cell::Cell;
use std::collections::HashMap;

use rand::RngCore;

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::Team;
use crate::cfr::infoset::{sample as sample_action, Strategy};
use crate::cfr::node::DecisionNode;
use crate::cfr::resolve::ResolvingPolicy;
use crate::model::registry::Registry;
use crate::rl::agent::Agent;
use crate::rl::encoder::encode;
use crate::rl::mask::Mask;
use crate::rl::moveslot::Moveslot;
use crate::cfr::solver::PLAYOUT_CAP;

/// A solver that searches afresh from whatever position it is shown.
///
/// This is what a person using a solver actually does, and it is stronger than a
/// precomputed table in a way that matters here. A table is built once, so its
/// horizon is fixed at the moment of building and the rest of the battle happens
/// beyond it. A live resolver's horizon *moves with it*: standing on turn twelve
/// and looking four turns ahead is a view of turns twelve to sixteen, which no
/// table built at turn one ever had.
///
/// It also never meets a position it cannot answer, so unlike [`CfrAgent`] there
/// is no random play to muddy the result.
///
/// The cost is a fresh search at every decision, which is why the policy caches.
pub struct ResolvingAgent<'a, 'r> {
	policy: &'a ResolvingPolicy<'r>,
	/// Set by `observe` immediately before each `choose_move`.
	pending: Option<(BattleState, StepRequest, Team)>,
	fallbacks: Cell<u64>,
	decisions: Cell<u64>,
}

impl<'a, 'r> ResolvingAgent<'a, 'r> {
	pub fn new(policy: &'a ResolvingPolicy<'r>) -> Self {
		ResolvingAgent {
			policy,
			pending: None,
			fallbacks: Cell::new(0),
			decisions: Cell::new(0),
		}
	}

	/// Decisions made without a position to search from.
	///
	/// Should be zero. Anything else means `observe` was not called before
	/// `choose_move`, and the agent was playing blind — which would quietly
	/// understate it.
	pub fn fallbacks(&self) -> u64 {
		self.fallbacks.get()
	}

	pub fn decisions(&self) -> u64 {
		self.decisions.get()
	}
}

impl Agent for ResolvingAgent<'_, '_> {
	fn observe(&mut self, state: &BattleState, request: &StepRequest, team: Team) {
		self.pending = Some((state.clone(), request.clone(), team));
	}

	fn choose_move(&mut self, _representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		self.decisions.set(self.decisions.get() + 1);

		let strategy = self
			.pending
			.as_ref()
			.and_then(|(state, request, team)| self.policy.strategy(state, request, *team));

		match strategy {
			Some(strategy) => Moveslot::from_number(sample_action(&strategy, rng)),
			None => {
				self.fallbacks.set(self.fallbacks.get() + 1);
				mask.get_random_valid(rng)
					.expect("asked for a command with no legal action")
			}
		}
	}
}

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
	/// Shown a position, the resolver searches it and plays a legal action — with
	/// no blind decisions, which is the whole advantage over a lookup table.
	#[test]
	fn a_resolver_answers_from_the_position_it_was_shown() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(8);

		let policy = ResolvingPolicy::with_heuristic(
			&registry, SolverConfig::for_lookahead(40, 3), 4);
		let mut agent = ResolvingAgent::new(&policy);

		let team = crate::battle::state::Team::Zero;
		let mask = Mask::from_battle_state(
			&team, crate::battle::state::field::PositionId(0), &root);
		let view = encode(&root, &registry, false, &team);

		agent.observe(&root, &StepRequest::NeedsActions, team);
		let chosen = agent.choose_move(&view, &mask, &mut rng);

		assert!(mask.allowed[chosen.to_number()], "must pick a legal action");
		assert_eq!(agent.fallbacks(), 0, "it was shown the position; it should not play blind");
		assert_eq!(agent.decisions(), 1);
		assert_eq!(policy.solves(), 1);
	}

	/// If the hook is ever missed the agent is playing blind, which would flatter
	/// it in a probe. That has to be counted rather than hidden.
	#[test]
	fn a_resolver_never_shown_a_position_counts_it() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(9);

		let policy = ResolvingPolicy::with_heuristic(
			&registry, SolverConfig::for_lookahead(40, 3), 4);
		let mut agent = ResolvingAgent::new(&policy);

		let team = crate::battle::state::Team::Zero;
		let mask = Mask::from_battle_state(
			&team, crate::battle::state::field::PositionId(0), &root);
		let view = encode(&root, &registry, false, &team);

		let chosen = agent.choose_move(&view, &mask, &mut rng);

		assert_eq!(agent.fallbacks(), 1);
		assert!(mask.allowed[chosen.to_number()], "a blind decision still plays legally");
	}

	/// The playout loops must call `observe`, or a searching agent silently plays
	/// at random inside every harness in the project.
	#[test]
	fn the_playout_loop_hands_the_position_over() {
		use crate::rl::agent::random_agent::RandomAgent;
		use crate::rl::battle_playout::play_headless_as;
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(10);

		let policy = ResolvingPolicy::with_heuristic(
			&registry, SolverConfig::for_lookahead(30, 2), 6);
		let mut agent = ResolvingAgent::new(&policy);
		let mut opponent = RandomAgent {};

		let (_, turns) = play_headless_as(
			root, &registry, &mut agent, &mut opponent,
			crate::battle::state::Team::Zero, &mut rng,
		);

		assert!(turns > 1);
		assert!(agent.decisions() > 0, "the agent should have been asked something");
		assert_eq!(
			agent.fallbacks(), 0,
			"every decision should have had a position behind it",
		);
	}

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
