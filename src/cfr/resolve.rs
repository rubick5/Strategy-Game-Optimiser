//! A strategy defined as "solve from wherever you are".
//!
//! # The two problems this answers
//!
//! A solve produces a table covering the positions it happened to visit. Reading
//! a strategy out of that table works until something asks about a position the
//! solve never reached — and a best response asks about exactly those, because
//! going where the strategy is not expecting you is what best responding *is*.
//!
//! [`crate::cfr::exploit`] answers "assume uniform play" there, and this replaces
//! the assumption with an answer.
//!
//! Worth recording that the assumption turned out **not** to be the problem it
//! looked like. Exploitability on the 2v2 plateaus around 0.17 however long it is
//! solved, and coverage falls from 88% to 80% over the same range, which made the
//! uniform fallback the obvious suspect. Measuring with and without this resolver
//! settles it: 0.1648 against 0.1648, and 0.1743 against 0.1744. The plateau is
//! chance-sampling bias in the measurement, not the fallback. Removing the
//! guesswork is still worth having — it is what makes a *deeper* measurement mean
//! anything — but it did not move that number.
//!
//! Separately, a depth-limited solve answers a *truncated* game, and nothing so
//! far has measured what the truncation costs. That number is invisible while the
//! best response is capped at the same horizon as the solve — both are looking at
//! the same short game and agreeing about it.
//!
//! # One object answers both
//!
//! Treat the strategy not as a table but as a *procedure*: from any position,
//! solve with a fixed lookahead and play the answer. That is what a person using
//! a solver actually does, and it is well defined everywhere.
//!
//! Coverage becomes total, because there is no position it cannot answer. And the
//! truncation cost becomes visible, because the best response can now be given a
//! *deeper* horizon than the policy has: the gap between what a four-turn
//! searcher plays and what an eight-turn opponent can do to it is exactly the
//! price of the horizon.
//!
//! The cost is a solve per distinct position asked about, which is why answers
//! are cached — a best response revisits the same positions constantly through
//! transpositions.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::battle::engine::engine::StepRequest;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::Team;
use crate::cfr::infoset::Strategy;
use crate::cfr::key::StateKey;
use crate::cfr::leaf::{HealthHeuristic, LeafEvaluator};
use crate::cfr::node::DecisionNode;
use crate::cfr::solver::{Solver, SolverConfig};
use crate::model::registry::Registry;

/// Builds the leaf estimate for each fresh solve.
///
/// A `Solver` owns its evaluator, so re-solving needs a way to make another one
/// rather than a reference to share.
pub type LeafFactory = Box<dyn Fn() -> Box<dyn LeafEvaluator>>;

pub struct ResolvingPolicy<'r> {
	registry: &'r Registry,
	config: SolverConfig,
	leaf: LeafFactory,
	cache: RefCell<HashMap<(StateKey, Team), Strategy>>,
	solves: Cell<u64>,
	lookups: Cell<u64>,
	rng: RefCell<StdRng>,
}

impl<'r> ResolvingPolicy<'r> {
	/// A policy that re-solves with `config` wherever it is asked.
	pub fn new(registry: &'r Registry, config: SolverConfig, leaf: LeafFactory, seed: u64) -> Self {
		ResolvingPolicy {
			registry,
			config,
			leaf,
			cache: RefCell::new(HashMap::new()),
			solves: Cell::new(0),
			lookups: Cell::new(0),
			rng: RefCell::new(StdRng::seed_from_u64(seed)),
		}
	}

	/// The usual case: re-solve with the plain health heuristic at the leaves.
	pub fn with_heuristic(registry: &'r Registry, config: SolverConfig, seed: u64) -> Self {
		Self::new(registry, config, Box::new(|| Box::new(HealthHeuristic)), seed)
	}

	/// How far ahead this policy looks before estimating.
	pub fn lookahead(&self) -> usize {
		self.config.max_depth
	}

	/// Fresh solves run, and questions asked. The ratio is how much the cache is
	/// earning.
	pub fn solves(&self) -> u64 {
		self.solves.get()
	}

	pub fn lookups(&self) -> u64 {
		self.lookups.get()
	}

	/// What this policy plays at `state`, for every side that owes a command.
	///
	/// One solve answers for both players at once, so both are cached.
	pub fn strategies(
		&self,
		state: &BattleState,
		request: &StepRequest,
	) -> Vec<(Team, Strategy)> {
		self.lookups.set(self.lookups.get() + 1);

		let Some(key) = StateKey::new(state, request) else {
			return Vec::new();
		};
		let actors = match DecisionNode::from(state, request) {
			DecisionNode::Terminal(_) => return Vec::new(),
			DecisionNode::Decision { actors } => actors,
		};

		// Cached already? Every actor has to be present, or the solve is redone.
		let cached: Option<Vec<(Team, Strategy)>> = {
			let cache = self.cache.borrow();
			actors
				.iter()
				.map(|actor| {
					cache
						.get(&(key.clone(), actor.team))
						.map(|strategy| (actor.team, *strategy))
				})
				.collect()
		};
		if let Some(found) = cached {
			return found;
		}

		self.solves.set(self.solves.get() + 1);
		let mut solver = Solver::with_leaf(
			self.registry,
			SolverConfig {
				iterations: self.config.iterations,
				max_depth: self.config.max_depth,
				max_nodes: self.config.max_nodes,
				transpositions: self.config.transpositions,
			},
			(self.leaf)(),
		);
		solver.solve_from(state, request.clone(), &mut *self.rng.borrow_mut());

		let mut answers = Vec::new();
		let mut cache = self.cache.borrow_mut();
		for actor in actors {
			let strategy = solver
				.average_strategy(state, request, actor.team)
				.unwrap_or_else(|| {
					crate::cfr::infoset::InfosetData::new().average_strategy(&actor.mask)
				});
			cache.insert((key.clone(), actor.team), strategy);
			answers.push((actor.team, strategy));
		}
		answers
	}

	/// What this policy plays for one side.
	pub fn strategy(
		&self,
		state: &BattleState,
		request: &StepRequest,
		team: Team,
	) -> Option<Strategy> {
		self.strategies(state, request)
			.into_iter()
			.find(|(side, _)| *side == team)
			.map(|(_, strategy)| strategy)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::switch_prediction_2v2;
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	fn policy(registry: &Registry) -> ResolvingPolicy<'_> {
		ResolvingPolicy::with_heuristic(registry, SolverConfig::for_lookahead(50, 3), 1)
	}

	/// The whole point: there is no position it cannot answer, which is what
	/// removes the guesswork from a coverage gap.
	#[test]
	fn it_answers_anywhere_including_positions_no_solve_visited() {
		let registry = Registry::load();
		let policy = policy(&registry);
		let mut rng = StdRng::seed_from_u64(3);

		// Walk somewhere arbitrary by playing at random, then ask from there.
		let mut state = switch_prediction_2v2(&registry);
		let mut request = StepRequest::NeedsActions;
		for _ in 0..3 {
			let actors = match DecisionNode::from(&state, &request) {
				DecisionNode::Terminal(_) => break,
				DecisionNode::Decision { actors } => actors,
			};
			let commands = actors
				.iter()
				.map(|actor| {
					let action = actor.mask.get_random_valid(&mut rng).unwrap();
					actor.command(action.to_number(), &state, &registry)
				})
				.collect();
			let result = crate::battle::engine::engine::step(state, commands, &registry, &mut rng);
			state = result.battle_state;
			request = result.step_request;
		}

		if let DecisionNode::Decision { actors } = DecisionNode::from(&state, &request) {
			for actor in actors {
				let strategy = policy
					.strategy(&state, &request, actor.team)
					.expect("a resolving policy has an answer everywhere");
				let total: f32 = strategy.iter().sum();
				assert!((total - 1.0).abs() < 1e-4, "not a distribution: {total}");
				for action in 0..strategy.len() {
					if !actor.mask.allowed[action] {
						assert_eq!(strategy[action], 0.0, "illegal action given weight");
					}
				}
			}
		}
	}

	/// Answers are cached, or a best response revisiting positions through
	/// transpositions would re-solve each one.
	#[test]
	fn asking_twice_solves_once() {
		let registry = Registry::load();
		let policy = policy(&registry);
		let state = switch_prediction_2v2(&registry);

		let first = policy.strategy(&state, &StepRequest::NeedsActions, Team::Zero);
		let after_one = policy.solves();
		let second = policy.strategy(&state, &StepRequest::NeedsActions, Team::Zero);

		assert_eq!(after_one, 1);
		assert_eq!(policy.solves(), 1, "the second question should come from cache");
		assert_eq!(first, second);
	}

	/// One solve answers for both sides, since a solve produces both strategies.
	#[test]
	fn one_solve_answers_both_players() {
		let registry = Registry::load();
		let policy = policy(&registry);
		let state = switch_prediction_2v2(&registry);

		let answers = policy.strategies(&state, &StepRequest::NeedsActions);
		assert_eq!(answers.len(), 2);
		assert_eq!(policy.solves(), 1);
	}
}
