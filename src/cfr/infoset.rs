//! Regret matching, and the table that holds it.
//!
//! This is the core of CFR and the one piece every other part of the solver
//! leans on, so it is deliberately free of any dependency on the battle engine:
//! it sees an action mask and a pile of accumulated regret, and nothing else.
//! That is what lets [`crate::cfr::matrix`] exercise this exact code against
//! matrix games whose equilibria are known in advance.
//!
//! The distinction that matters most here is **current strategy vs average
//! strategy**. Regret matching produces the current strategy, which is what you
//! play during a traversal — and it does *not* converge to anything. It cycles.
//! The thing that converges to equilibrium is the running average, which is why
//! [`InfosetData::average_strategy`] is what the solver reports and
//! [`InfosetData::current_strategy`] is what it samples from.

use std::collections::HashMap;

use rand::{Rng, RngCore};

use crate::battle::state::Team;
use crate::cfr::key::StateKey;
use crate::rl::mask::Mask;
use crate::rl::moveslot::MAX_DECISION;

/// A distribution over the 10 possible decisions. Illegal actions are always
/// exactly zero, never merely small.
pub type Strategy = [f32; MAX_DECISION];

/// Everything CFR accumulates at one information set.
#[derive(Debug, Clone)]
pub struct InfosetData {
	/// Counterfactual regret for each action, summed over every visit.
	regret_sum: Strategy,
	/// Reach-weighted sum of the strategies played here. Its normalisation is
	/// the equilibrium estimate.
	strategy_sum: Strategy,
}

impl Default for InfosetData {
	fn default() -> Self {
		InfosetData {
			regret_sum: [0.0; MAX_DECISION],
			strategy_sum: [0.0; MAX_DECISION],
		}
	}
}

impl InfosetData {
	pub fn new() -> Self {
		Self::default()
	}

	/// Regret matching: play each action in proportion to how much regret has
	/// accumulated for *not* having played it.
	///
	/// Only legal actions are considered, so an action the mask forbids gets zero
	/// probability no matter how much regret it has piled up — regret can survive
	/// from an earlier visit where the same position allowed something this one
	/// does not.
	///
	/// With no positive regret anywhere there is nothing to prefer, so this falls
	/// back to uniform over the legal actions.
	pub fn current_strategy(&self, mask: &Mask) -> Strategy {
		let mut strategy = [0.0; MAX_DECISION];
		let mut total = 0.0;

		for action in 0..MAX_DECISION {
			if !mask.allowed[action] {
				continue;
			}
			let positive = self.regret_sum[action].max(0.0);
			strategy[action] = positive;
			total += positive;
		}

		if total > 0.0 {
			for probability in strategy.iter_mut() {
				*probability /= total;
			}
			strategy
		} else {
			uniform_over_legal(mask)
		}
	}

	/// The equilibrium estimate: the reach-weighted average of every strategy
	/// played here.
	pub fn average_strategy(&self, mask: &Mask) -> Strategy {
		let mut strategy = [0.0; MAX_DECISION];
		let mut total = 0.0;

		for action in 0..MAX_DECISION {
			if !mask.allowed[action] {
				continue;
			}
			let weight = self.strategy_sum[action].max(0.0);
			strategy[action] = weight;
			total += weight;
		}

		if total > 0.0 {
			for probability in strategy.iter_mut() {
				*probability /= total;
			}
			strategy
		} else {
			uniform_over_legal(mask)
		}
	}

	/// Fold one visit's strategy into the running average.
	pub fn accumulate_strategy(&mut self, strategy: &Strategy, weight: f32) {
		if weight <= 0.0 {
			return;
		}
		for action in 0..MAX_DECISION {
			self.strategy_sum[action] += weight * strategy[action];
		}
	}

	pub fn add_regret(&mut self, action: usize, amount: f32) {
		self.regret_sum[action] += amount;
	}

	pub fn regret(&self, action: usize) -> f32 {
		self.regret_sum[action]
	}
}

/// Regret for each action, given what each one turned out to be worth.
///
/// This is the CFR update in its plainest form: an action's regret is how much
/// better it would have been than what the current strategy actually gets. The
/// same function serves the battle solver and [`crate::cfr::matrix`], so the
/// matrix games with known equilibria are testing the exact update the solver
/// applies rather than a lookalike.
///
/// `action_values` is only read at legal actions.
pub fn regrets_from_action_values(
	action_values: &Strategy,
	strategy: &Strategy,
	mask: &Mask,
) -> Strategy {
	let mut node_value = 0.0;
	for action in 0..MAX_DECISION {
		if mask.allowed[action] {
			node_value += strategy[action] * action_values[action];
		}
	}

	let mut deltas = [0.0; MAX_DECISION];
	for action in 0..MAX_DECISION {
		if mask.allowed[action] {
			deltas[action] = action_values[action] - node_value;
		}
	}
	deltas
}

/// Mix a strategy towards uniform, so every legal action keeps being visited.
///
/// Outcome sampling only ever learns about the action it happened to sample, so
/// an action that drops to probability zero stops being explored and its regret
/// stops updating — it can never come back, even if it was abandoned by accident
/// early on. This is what stops that happening.
pub fn explore(strategy: &Strategy, mask: &Mask, exploration: f32) -> Strategy {
	let legal = mask.allowed.iter().filter(|allowed| **allowed).count();
	if legal == 0 {
		return [0.0; MAX_DECISION];
	}
	let share = exploration / legal as f32;
	std::array::from_fn(|action| {
		if mask.allowed[action] {
			share + (1.0 - exploration) * strategy[action]
		} else {
			0.0
		}
	})
}

/// Draw an action index from a distribution.
pub fn sample(strategy: &Strategy, rng: &mut dyn RngCore) -> usize {
	let roll: f32 = rng.random();
	let mut cumulative = 0.0;
	let mut last_positive = None;

	for action in 0..MAX_DECISION {
		if strategy[action] <= 0.0 {
			continue;
		}
		last_positive = Some(action);
		cumulative += strategy[action];
		if cumulative >= roll {
			return action;
		}
	}

	// Only reachable through floating-point drift at the very top of the range,
	// where the probabilities sum to a hair under the roll.
	last_positive.expect("cannot sample an action from an empty distribution")
}

/// An equal chance of every legal action, or all zeros if there are none.
fn uniform_over_legal(mask: &Mask) -> Strategy {
	let mut strategy = [0.0; MAX_DECISION];
	let legal = mask.allowed.iter().filter(|allowed| **allowed).count();
	if legal == 0 {
		return strategy;
	}
	let share = 1.0 / legal as f32;
	for action in 0..MAX_DECISION {
		if mask.allowed[action] {
			strategy[action] = share;
		}
	}
	strategy
}

/// Every information set discovered so far.
///
/// Keyed by position *and* team: the two players sit at the same public state
/// but hold entirely separate regrets, which is exactly what makes this a
/// simultaneous-move game rather than a perfect-information one.
#[derive(Debug, Default)]
pub struct InfosetTable {
	entries: HashMap<(StateKey, Team), InfosetData>,
}

impl InfosetTable {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn entry(&mut self, key: &StateKey, team: Team) -> &mut InfosetData {
		self.entries
			.entry((key.clone(), team))
			.or_default()
	}

	pub fn get(&self, key: &StateKey, team: Team) -> Option<&InfosetData> {
		self.entries.get(&(key.clone(), team))
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A mask allowing the first `count` actions.
	fn first_n(count: usize) -> Mask {
		Mask { allowed: std::array::from_fn(|i| i < count) }
	}

	#[test]
	fn no_regret_yet_means_uniform_over_legal_actions() {
		let data = InfosetData::new();
		let strategy = data.current_strategy(&first_n(4));

		for action in 0..4 {
			assert!((strategy[action] - 0.25).abs() < 1e-6);
		}
		assert_eq!(strategy[4..].iter().sum::<f32>(), 0.0);
	}

	#[test]
	fn regret_matching_is_proportional_to_positive_regret() {
		let mut data = InfosetData::new();
		data.add_regret(0, 3.0);
		data.add_regret(1, 1.0);

		let strategy = data.current_strategy(&first_n(2));
		assert!((strategy[0] - 0.75).abs() < 1e-6);
		assert!((strategy[1] - 0.25).abs() < 1e-6);
	}

	#[test]
	fn negative_regret_everywhere_falls_back_to_uniform() {
		let mut data = InfosetData::new();
		data.add_regret(0, -5.0);
		data.add_regret(1, -2.0);

		let strategy = data.current_strategy(&first_n(2));
		assert!((strategy[0] - 0.5).abs() < 1e-6);
		assert!((strategy[1] - 0.5).abs() < 1e-6);
	}

	/// Regret survives between visits, but legality does not — a position can
	/// allow an action once and forbid it later. An illegal action must get
	/// exactly zero, not a small number.
	#[test]
	fn an_illegal_action_gets_no_probability_however_much_regret_it_has() {
		let mut data = InfosetData::new();
		data.add_regret(0, 100.0);
		data.add_regret(1, 1.0);

		let mut mask = first_n(2);
		mask.allowed[0] = false;

		let strategy = data.current_strategy(&mask);
		assert_eq!(strategy[0], 0.0, "forbidden actions are zero, not merely unlikely");
		assert!((strategy[1] - 1.0).abs() < 1e-6);
	}

	#[test]
	fn the_average_strategy_is_reach_weighted() {
		let mut data = InfosetData::new();
		let mask = first_n(2);

		// Played 1.0-weighted as (1, 0), then 3.0-weighted as (0, 1).
		data.accumulate_strategy(&{ let mut s = [0.0; MAX_DECISION]; s[0] = 1.0; s }, 1.0);
		data.accumulate_strategy(&{ let mut s = [0.0; MAX_DECISION]; s[1] = 1.0; s }, 3.0);

		let average = data.average_strategy(&mask);
		assert!((average[0] - 0.25).abs() < 1e-6);
		assert!((average[1] - 0.75).abs() < 1e-6);
	}

	#[test]
	fn a_mask_with_nothing_legal_yields_no_probability_at_all() {
		let data = InfosetData::new();
		let strategy = data.current_strategy(&first_n(0));
		assert_eq!(strategy.iter().sum::<f32>(), 0.0);
	}
}
