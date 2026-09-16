//! Matrix games: the validation harness.
//!
//! CFR's nastiest property is that a wrong implementation does not crash or
//! diverge. It converges smoothly — to the equilibrium of some *other* game.
//! Average regret falls, the strategy settles, nothing looks amiss. The only
//! defence is positions whose answer is known independently, and for a battle
//! position there is no such thing.
//!
//! So the pieces the battle solver is built from are exercised here first,
//! against two-player zero-sum matrix games small enough to have hand-computable
//! equilibria. Both entry points drive the *real* code:
//!
//! [`MatrixGame::solve`] drives [`InfosetData`]'s regret matching and averaging
//! and [`regrets_from_action_values`] — the same update [`crate::cfr::solver`]
//! applies at a battle node, so this is testing the solver's own code rather
//! than a lookalike.
//!
//! Unlike a battle, a matrix game also has an **exactly** computable
//! exploitability, so convergence can be asserted rather than eyeballed.

use crate::cfr::infoset::{regrets_from_action_values, InfosetData, Strategy};
use crate::rl::mask::Mask;
use crate::rl::moveslot::MAX_DECISION;

/// A two-player zero-sum game in normal form.
///
/// `payoff[row][column]` is the value to player 0; player 1 receives its
/// negation.
pub struct MatrixGame {
	payoff: Vec<Vec<f32>>,
}

impl MatrixGame {
	/// Panics on a ragged payoff matrix, or one larger than the 10-action space
	/// [`Mask`] describes — this reuses the battle action space deliberately, so
	/// that it is the battle code being tested.
	pub fn new(payoff: Vec<Vec<f32>>) -> Self {
		assert!(!payoff.is_empty(), "a game needs at least one row");
		let columns = payoff[0].len();
		assert!(columns > 0, "a game needs at least one column");
		assert!(
			payoff.iter().all(|row| row.len() == columns),
			"payoff matrix is ragged"
		);
		assert!(
			payoff.len() <= MAX_DECISION && columns <= MAX_DECISION,
			"matrix games here reuse the {MAX_DECISION}-action battle space"
		);
		MatrixGame { payoff }
	}

	pub fn rows(&self) -> usize {
		self.payoff.len()
	}

	pub fn columns(&self) -> usize {
		self.payoff[0].len()
	}

	fn row_mask(&self) -> Mask {
		Mask { allowed: std::array::from_fn(|i| i < self.rows()) }
	}

	fn column_mask(&self) -> Mask {
		Mask { allowed: std::array::from_fn(|i| i < self.columns()) }
	}

	/// Player 0's expected value when both sides play the given strategies.
	pub fn value(&self, row_strategy: &Strategy, column_strategy: &Strategy) -> f32 {
		let mut total = 0.0;
		for row in 0..self.rows() {
			for column in 0..self.columns() {
				total += row_strategy[row] * column_strategy[column] * self.payoff[row][column];
			}
		}
		total
	}

	/// How much the pair of strategies leaves on the table, summed over both
	/// players.
	///
	/// Each term is what that player would gain by switching to their best
	/// response. In a zero-sum game the sum is zero exactly at equilibrium and
	/// positive everywhere else, so this is the number that says whether a solve
	/// worked.
	pub fn exploitability(&self, row_strategy: &Strategy, column_strategy: &Strategy) -> f32 {
		let mut best_row = f32::NEG_INFINITY;
		for row in 0..self.rows() {
			let value: f32 = (0..self.columns())
				.map(|column| column_strategy[column] * self.payoff[row][column])
				.sum();
			best_row = best_row.max(value);
		}

		let mut best_column = f32::NEG_INFINITY;
		for column in 0..self.columns() {
			let value: f32 = (0..self.rows())
				.map(|row| row_strategy[row] * -self.payoff[row][column])
				.sum();
			best_column = best_column.max(value);
		}

		best_row + best_column
	}

	/// Vanilla CFR: at a single-node game, every action's counterfactual value
	/// can be computed exactly, so nothing is sampled.
	///
	/// Returns both players' average strategies.
	pub fn solve(&self, iterations: usize) -> (Strategy, Strategy) {
		let mut row_player = InfosetData::new();
		let mut column_player = InfosetData::new();
		let row_mask = self.row_mask();
		let column_mask = self.column_mask();

		for _ in 0..iterations {
			let row_strategy = row_player.current_strategy(&row_mask);
			let column_strategy = column_player.current_strategy(&column_mask);

			row_player.accumulate_strategy(&row_strategy, 1.0);
			column_player.accumulate_strategy(&column_strategy, 1.0);

			// What each row is worth against the opponent's current mix, and the
			// same for each column in player 1's own (negated) currency.
			let mut row_values = [0.0; MAX_DECISION];
			for row in 0..self.rows() {
				row_values[row] = (0..self.columns())
					.map(|column| column_strategy[column] * self.payoff[row][column])
					.sum();
			}
			let mut column_values = [0.0; MAX_DECISION];
			for column in 0..self.columns() {
				column_values[column] = (0..self.rows())
					.map(|row| row_strategy[row] * -self.payoff[row][column])
					.sum();
			}

			let row_deltas = regrets_from_action_values(&row_values, &row_strategy, &row_mask);
			let column_deltas =
				regrets_from_action_values(&column_values, &column_strategy, &column_mask);
			for action in 0..MAX_DECISION {
				row_player.add_regret(action, row_deltas[action]);
				column_player.add_regret(action, column_deltas[action]);
			}
		}

		(
			row_player.average_strategy(&row_mask),
			column_player.average_strategy(&column_mask),
		)
	}

}

#[cfg(test)]
mod tests {
	use super::*;

	fn rock_paper_scissors() -> MatrixGame {
		MatrixGame::new(vec![
			vec![0.0, -1.0, 1.0],
			vec![1.0, 0.0, -1.0],
			vec![-1.0, 1.0, 0.0],
		])
	}

	/// Payoffs chosen so the equilibrium is lopsided rather than uniform: player
	/// 0 must play row 0 a quarter of the time. Uniform play would pass a weaker
	/// test, so this one catches an implementation that merely smooths out.
	///
	/// Equilibrium: row (1/4, 3/4), column (1/2, 1/2), value 0.
	fn asymmetric_two_by_two() -> MatrixGame {
		MatrixGame::new(vec![
			vec![3.0, -3.0],
			vec![-1.0, 1.0],
		])
	}

	fn assert_close(actual: f32, expected: f32, tolerance: f32, label: &str) {
		assert!(
			(actual - expected).abs() < tolerance,
			"{label}: expected ~{expected}, got {actual}"
		);
	}

	#[test]
	fn vanilla_cfr_finds_the_rock_paper_scissors_equilibrium() {
		let game = rock_paper_scissors();
		let (row, column) = game.solve(20_000);

		for action in 0..3 {
			assert_close(row[action], 1.0 / 3.0, 0.02, "row");
			assert_close(column[action], 1.0 / 3.0, 0.02, "column");
		}
	}

	#[test]
	fn vanilla_cfr_finds_a_lopsided_equilibrium() {
		let game = asymmetric_two_by_two();
		let (row, column) = game.solve(50_000);

		assert_close(row[0], 0.25, 0.02, "row 0");
		assert_close(row[1], 0.75, 0.02, "row 1");
		assert_close(column[0], 0.5, 0.02, "column 0");
		assert_close(column[1], 0.5, 0.02, "column 1");
	}

	/// The headline property: exploitability falls towards zero, at the rate CFR
	/// is supposed to manage.
	///
	/// The lopsided game is used rather than RPS on purpose. Both players start
	/// from uniform, and uniform *is* the RPS equilibrium, so RPS would score zero
	/// exploitability before doing any work and the test would pass without
	/// testing anything.
	///
	/// The assertion is on the *rate* rather than against some chosen constant.
	/// CFR converges as O(1/sqrt(T)), so a hundredfold increase in iterations
	/// should cut exploitability by roughly ten. That is a real property of the
	/// algorithm; an absolute threshold is just a number that happened to pass.
	#[test]
	fn vanilla_cfr_drives_exploitability_towards_zero() {
		let game = asymmetric_two_by_two();

		let measure = |iterations| {
			let (row, column) = game.solve(iterations);
			game.exploitability(&row, &column)
		};

		let untrained = measure(1);
		let some = measure(2_000);
		let more = measure(200_000);

		assert!(
			untrained > 0.5,
			"uniform play should be clearly exploitable, got {untrained}",
		);
		assert!(
			some < untrained && more < some,
			"exploitability should keep falling: {untrained} -> {some} -> {more}",
		);
		assert!(
			more < some / 5.0,
			"a hundredfold more iterations should cut exploitability by roughly ten, \
			 got {some} -> {more}",
		);
	}

	/// A game where one option is simply better. The solver must commit to it,
	/// not hedge — this is the matrix analogue of the dominant-move battle test.
	#[test]
	fn a_dominated_action_is_abandoned() {
		let game = MatrixGame::new(vec![
			vec![1.0, 1.0],
			vec![-1.0, -1.0],
		]);
		let (row, _) = game.solve(10_000);

		assert!(row[0] > 0.99, "should commit to the dominant row, got {}", row[0]);
	}

	#[test]
	fn exploitability_is_zero_at_a_known_equilibrium_and_positive_away_from_it() {
		let game = rock_paper_scissors();

		let mut equilibrium = [0.0; MAX_DECISION];
		for action in 0..3 {
			equilibrium[action] = 1.0 / 3.0;
		}
		assert_close(
			game.exploitability(&equilibrium, &equilibrium),
			0.0,
			1e-5,
			"uniform RPS is the equilibrium",
		);

		let mut always_rock = [0.0; MAX_DECISION];
		always_rock[0] = 1.0;
		assert!(
			game.exploitability(&always_rock, &equilibrium) > 0.5,
			"always-rock should be very exploitable"
		);
	}
}
