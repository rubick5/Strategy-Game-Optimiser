//! Leaves that do not assume how the rest of the battle is played.
//!
//! # The assumption a single leaf value makes
//!
//! Cutting the search off at turn N and calling an estimator there commits to a
//! belief about turns N+1 onward — perhaps twenty more decisions. The opponent is
//! under no obligation to play them that way, and a strategy solved against one
//! assumed continuation is safe only against that continuation. That is the gap
//! the PPO prober walks through: the exact best response says 0.07 inside the
//! horizon while the prober wins 55% of whole battles.
//!
//! Note that this has nothing to do with reacting to a move within a turn. Moves
//! here resolve simultaneously, so nobody reacts to anything — but the opponent
//! still chooses how to play the remaining twenty turns, and that is the freedom
//! being hedged against.
//!
//! # Letting the opponent choose
//!
//! So the leaf offers the opponent a set of continuation strategies and takes the
//! worst of them. The solve can then no longer lean on a single guess about what
//! happens next: whatever it settles on has to hold up against every continuation
//! in the set.
//!
//! This is the depth-limited solving of Brown and Sandholm's *Modicum*, and two
//! of its complications simply do not arise here. There are no beliefs to keep
//! consistent, because the position at the horizon is public — poker needs a
//! value per information set, a vector over hands; this needs one number per
//! continuation. And there is no move-order subtlety at the leaf, because the
//! continuation is chosen at a state both players can see and nothing
//! simultaneous happens inside it.
//!
//! # Keeping it zero-sum
//!
//! Each side taking its own pessimistic minimum would break the one property
//! [`HealthHeuristic`] had for free: both
//! sides would be estimated as slightly losing, the values would not sum to zero,
//! and CFR would be solving a game where value leaks out of the board.
//!
//! So both pessimistic views are taken and the answer is their difference,
//! halved — the same antisymmetrisation the learned critic uses. Zero-sum by
//! construction, with each side's hedge represented.
//!
//! # What this can and cannot promise
//!
//! It is a hedge, not a proof. The guarantee is only as good as the set: a
//! strategy is unexploitable by the continuations offered, and says nothing about
//! one nobody thought of. That is worth being precise about, because it is
//! exactly the claim that can honestly be made — *"unexploitable against every
//! continuation we could find, and here they are"* — rather than the unqualified
//! "GTO" that a depth-limited solve cannot support.
//!
//! It also points at how to strengthen it. A trained prober is an approximate
//! best response, so adding one to the set hardens the solve against precisely
//! the hole it found; re-probe, add the next one, repeat. That is a double-oracle
//! loop, and the probe win rate already measures whether it is working.
//!
//! # It does not help yet
//!
//! One turn of that loop, on the 2v2: train a prober against the plain heuristic,
//! put it in the continuation set, and send a fresh prober at the result.
//!
//! ```text
//! leaf                       fresh prober wins    cost
//! health heuristic                       0.536     351s
//! multi-valued, 1 rollout                0.575     677s
//! multi-valued, 4 rollouts               0.547    1778s
//! ```
//!
//! Worse, and five times the price. The rollout count explains most of the gap:
//! at one rollout the estimate carries a standard deviation of 0.10 against a
//! signal of about 0.21, and quadrupling rollouts halves that and recovers most
//! of the regression. Extrapolating the trend, a noiseless version would be at
//! parity with the heuristic rather than ahead of it.
//!
//! Which is the same result as the learned critic and the ReBeL loop before it,
//! and by now the pattern is worth naming. **CFR reads differences between
//! sibling positions, not levels.** The heuristic's error is large but *smooth* —
//! it moves the same way across the children of a node, so most of it cancels out
//! of the regret. Every replacement tried so far has been more accurate on
//! average and less consistent between siblings, and has lost.
//!
//! That says the next thing to try is not a better estimator of the value, but an
//! estimator trained on sibling *differences* — the quantity actually consumed.
//! A single continuation and more rollouts would also be worth separating, since
//! this test varied both the hedge and the noise at once.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::{Outcome, Team};
use crate::cfr::critic::pending_request;
use crate::cfr::key::StateKey;
use crate::cfr::leaf::{HealthHeuristic, LeafEvaluator};
use crate::cfr::node::{Actor, DecisionNode};
use crate::cfr::solver::PLAYOUT_CAP;
use crate::model::registry::Registry;
use crate::rl::agent::Agent;
use crate::rl::encoder::encode;

/// A way of playing on from a position.
///
/// Distinct from [`Agent`] because a continuation is asked about a *position*,
/// and `Agent` is handed an encoding. Anything implementing `Agent` can be
/// wrapped — see [`AgentContinuation`] — which is what lets a trained prober be
/// offered to the opponent as one of their options.
pub trait Continuation {
	fn act(
		&mut self,
		state: &BattleState,
		request: &StepRequest,
		actor: &Actor,
		registry: &Registry,
		rng: &mut dyn RngCore,
	) -> usize;
}

/// Any [`Agent`] as a continuation.
pub struct AgentContinuation<A: Agent> {
	pub agent: A,
}

impl<A: Agent> Continuation for AgentContinuation<A> {
	fn act(
		&mut self,
		state: &BattleState,
		request: &StepRequest,
		actor: &Actor,
		registry: &Registry,
		rng: &mut dyn RngCore,
	) -> usize {
		let replacement = matches!(request, StepRequest::NeedsReplacements(_));
		let view = encode(state, registry, replacement, &actor.team);
		// Agents that think by searching need the position; the rest ignore it.
		self.agent.observe(state, request, actor.team);
		self.agent.choose_move(&view, &actor.mask, rng).to_number()
	}
}

/// Plays whatever leaves the board looking best one turn from now.
///
/// A reasonable blueprint and a reasonable continuation, and cheap: one engine
/// step per candidate action rather than a search. Greedy on the same health
/// measure the plain heuristic uses, so it is the natural "play sensibly on"
/// baseline.
pub struct Greedy;

impl Continuation for Greedy {
	fn act(
		&mut self,
		state: &BattleState,
		_request: &StepRequest,
		actor: &Actor,
		registry: &Registry,
		rng: &mut dyn RngCore,
	) -> usize {
		let mut best = None;
		let mut best_value = f32::NEG_INFINITY;

		for action in 0..actor.mask.allowed.len() {
			if !actor.mask.allowed[action] {
				continue;
			}
			// Only this side's command: the engine resolves a partial turn by
			// letting the missing side do nothing, which is enough to rank moves.
			let command = actor.command(action, state, registry);
			let StepResult { battle_state, .. } =
				engine::step(state.clone(), vec![command], registry, rng);
			let value = HealthHeuristic.value(&battle_state, actor.team, registry);
			if value > best_value {
				best_value = value;
				best = Some(action);
			}
		}

		best.unwrap_or_else(|| {
			actor
				.mask
				.get_random_valid(rng)
				.expect("a decision node has a legal action")
				.to_number()
		})
	}
}

/// A leaf estimate that lets the opponent pick how the battle continues.
pub struct MultiValuedLeaf {
	blueprint: RefCell<Box<dyn Continuation>>,
	continuations: RefCell<Vec<Box<dyn Continuation>>>,
	/// Games averaged per continuation. Each is a real playout, so this trades
	/// directly against time.
	rollouts: usize,
	cache: RefCell<HashMap<StateKey, f32>>,
	rng: RefCell<StdRng>,
	evaluations: Cell<u64>,
	rollouts_run: Cell<u64>,
}

impl MultiValuedLeaf {
	pub fn new(
		blueprint: Box<dyn Continuation>,
		continuations: Vec<Box<dyn Continuation>>,
		rollouts: usize,
		seed: u64,
	) -> Self {
		assert!(!continuations.is_empty(), "the opponent needs something to choose from");
		MultiValuedLeaf {
			blueprint: RefCell::new(blueprint),
			continuations: RefCell::new(continuations),
			rollouts: rollouts.max(1),
			cache: RefCell::new(HashMap::new()),
			rng: RefCell::new(StdRng::seed_from_u64(seed)),
			evaluations: Cell::new(0),
			rollouts_run: Cell::new(0),
		}
	}

	/// Questions asked, and games actually played. The ratio is what the cache is
	/// saving — the same positions come back constantly through transpositions.
	pub fn evaluations(&self) -> u64 {
		self.evaluations.get()
	}

	pub fn rollouts_run(&self) -> u64 {
		self.rollouts_run.get()
	}

	pub fn cached_positions(&self) -> usize {
		self.cache.borrow().len()
	}

	/// The worst this side can be held to, over every continuation on offer.
	fn worst_case(&self, state: &BattleState, defender: Team, registry: &Registry) -> f32 {
		let mut worst = f32::INFINITY;
		let count = self.continuations.borrow().len();

		for index in 0..count {
			let mut total = 0.0;
			for _ in 0..self.rollouts {
				total += self.play_out(state, defender, index, registry);
			}
			worst = worst.min(total / self.rollouts as f32);
		}
		worst
	}

	/// One game from `state`, with `defender` playing the blueprint and the other
	/// side playing continuation `index`. Returns the result to `defender`.
	fn play_out(
		&self,
		state: &BattleState,
		defender: Team,
		index: usize,
		registry: &Registry,
	) -> f32 {
		self.rollouts_run.set(self.rollouts_run.get() + 1);

		let mut state = state.clone();
		let mut request = pending_request(&state);
		let mut rng = self.rng.borrow_mut();

		for _ in 0..PLAYOUT_CAP {
			let actors = match DecisionNode::from(&state, &request) {
				DecisionNode::Terminal(outcome) => {
					return match outcome {
						Outcome::Win { team } => if team == defender { 1.0 } else { -1.0 },
						Outcome::Draw => 0.0,
					};
				}
				DecisionNode::Decision { actors } => actors,
			};

			let mut commands = Vec::new();
			for actor in &actors {
				let action = if actor.team == defender {
					self.blueprint.borrow_mut().act(&state, &request, actor, registry, &mut *rng)
				} else {
					self.continuations.borrow_mut()[index]
						.act(&state, &request, actor, registry, &mut *rng)
				};
				commands.push(actor.command(action, &state, registry));
			}

			let StepResult { battle_state, step_request } =
				engine::step(state, commands, registry, &mut *rng);
			state = battle_state;
			request = step_request;
		}

		// Nobody finished. Fall back on the plain measure rather than guessing.
		HealthHeuristic.value(&state, defender, registry)
	}
}

impl LeafEvaluator for MultiValuedLeaf {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		self.evaluations.set(self.evaluations.get() + 1);

		let key = StateKey::new(state, &pending_request(state));
		if let Some(key) = key.as_ref() {
			if let Some(cached) = self.cache.borrow().get(key) {
				return match team {
					Team::Zero => *cached,
					Team::One => -*cached,
				};
			}
		}

		// Both sides' worst case, combined into one zero-sum number.
		let zero = self.worst_case(state, Team::Zero, registry);
		let one = self.worst_case(state, Team::One, registry);
		let to_zero = ((zero - one) / 2.0).clamp(-1.0, 1.0);

		if let Some(key) = key {
			self.cache.borrow_mut().insert(key, to_zero);
		}

		match team {
			Team::Zero => to_zero,
			Team::One => -to_zero,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::switch_prediction_2v2;
	use crate::rl::agent::random_agent::RandomAgent;

	fn leaf(rollouts: usize) -> MultiValuedLeaf {
		MultiValuedLeaf::new(
			Box::new(Greedy),
			vec![Box::new(Greedy), Box::new(AgentContinuation { agent: RandomAgent {} })],
			rollouts,
			7,
		)
	}

	/// The property the antisymmetrisation exists for. Each side taking its own
	/// pessimistic minimum would leave both looking slightly behind, and CFR would
	/// be solving a game where value leaks off the board.
	#[test]
	fn the_estimate_is_zero_sum() {
		let registry = Registry::load();
		let leaf = leaf(1);
		let state = switch_prediction_2v2(&registry);

		let zero = leaf.value(&state, Team::Zero, &registry);
		let one = leaf.value(&state, Team::One, &registry);

		assert!((zero + one).abs() < 1e-6, "not zero-sum: {zero} vs {one}");
	}

	#[test]
	fn estimates_stay_on_the_result_scale() {
		let registry = Registry::load();
		let leaf = leaf(1);
		let state = switch_prediction_2v2(&registry);

		for team in [Team::Zero, Team::One] {
			let value = leaf.value(&state, team, &registry);
			assert!((-1.0..=1.0).contains(&value), "out of range: {value}");
		}
	}

	/// Rollouts are the expensive part, so asking twice must not play twice.
	#[test]
	fn a_repeated_position_is_answered_from_cache() {
		let registry = Registry::load();
		let leaf = leaf(2);
		let state = switch_prediction_2v2(&registry);

		leaf.value(&state, Team::Zero, &registry);
		let after_first = leaf.rollouts_run();
		assert!(after_first > 0, "the first answer needs games played");

		leaf.value(&state, Team::Zero, &registry);
		leaf.value(&state, Team::One, &registry);
		assert_eq!(leaf.rollouts_run(), after_first, "repeats should come from cache");
		assert_eq!(leaf.evaluations(), 3);
	}

	#[test]
	fn probe_leaf_noise() {
		use crate::rl::agent::random_agent::RandomAgent;
		let registry = Registry::load();
		let state = switch_prediction_2v2(&registry);

		// How much does the estimate move if only the seed changes? The health
		// heuristic is deterministic, so its answer is the reference: zero spread.
		println!("heuristic: {:.4} (deterministic)",
			HealthHeuristic.value(&state, Team::Zero, &registry));

		for rollouts in [1usize, 4, 16] {
			let values: Vec<f32> = (0..24u64)
				.map(|seed| {
					let leaf = MultiValuedLeaf::new(
						Box::new(Greedy),
						vec![
							Box::new(Greedy) as Box<dyn Continuation>,
							Box::new(AgentContinuation { agent: RandomAgent {} }),
						],
						rollouts,
						seed,
					);
					leaf.value(&state, Team::Zero, &registry)
				})
				.collect();
			let n = values.len() as f32;
			let mean = values.iter().sum::<f32>() / n;
			let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0)).sqrt();
			println!("rollouts {rollouts:>2}: mean {mean:+.4}  sd {sd:.4}");
		}
	}

	/// A mirror position has to be worth nothing, whatever the continuations are —
	/// both sides face the same set from the same position.
	#[test]
	fn a_mirror_is_worth_nothing_to_either_side() {
		let registry = Registry::load();
		// Greedy alone, so the answer is deterministic apart from engine chance.
		let leaf = MultiValuedLeaf::new(Box::new(Greedy), vec![Box::new(Greedy)], 8, 3);
		let state = switch_prediction_2v2(&registry);

		let value = leaf.value(&state, Team::Zero, &registry);
		assert!(value.abs() < 0.35, "a mirror should be near zero, got {value}");
	}
}
