//! Valuing a position the search stops short of.
//!
//! At 1v1 this was not needed: those positions end on their own, so every line
//! could be followed to a real win or loss. A 2v2 with switching cannot be. Each
//! creature has four moves and a switch, and external sampling branches on every
//! one of the traverser's actions, so the tree grows as `5^turns` — a single
//! full-depth iteration on `example_battles/coverage_test.json` passes twenty
//! million nodes without finishing.
//!
//! So the search is cut off at a fixed horizon and whatever it finds there is
//! *estimated*. That changes what a solve means, and the change is worth stating
//! plainly: the answer is an equilibrium of the **truncated** game, and it is only
//! as good as the estimate at the horizon. A solver that looks four turns ahead
//! and then guesses will not find a sacrifice that pays off on turn six.
//!
//! This is also why the 1v1 work was done first — there, a wrong answer could
//! only come from the CFR itself, with no leaf estimate to share the blame.

use crate::battle::state::battle_state::BattleState;
use crate::battle::state::Team;
use crate::model::registry::Registry;

/// Estimates what a non-terminal position is worth.
///
/// Values are on the same scale as a real result — `-1` lost, `+1` won — so that
/// a truncated line and a finished one can be compared without rescaling.
///
/// Takes `&self` rather than `&mut self`, so an implementation holding something
/// that mutates while evaluating — a network caching activations, say — keeps
/// that behind interior mutability. See [`crate::cfr::critic`].
pub trait LeafEvaluator {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32;
}

/// Remaining health, counted as a fraction of each creature's own maximum.
///
/// A fainted creature contributes zero, so losing one of two costs half the
/// available margin — material and chip damage land on the same scale rather
/// than needing to be traded off against each other.
///
/// This is a deliberately plain heuristic, and it is the weakest part of a 2v2
/// solve. It cannot see a type matchup, a status condition, or a creature that is
/// healthy but walled, so it will undervalue a switch made for position rather
/// than for HP. Replacing it with the PPO critic from [`crate::rl`] is the
/// obvious upgrade — the trait exists so that can be dropped in without touching
/// the solver.
pub struct HealthHeuristic;

impl LeafEvaluator for HealthHeuristic {
	fn value(&self, state: &BattleState, team: Team, _registry: &Registry) -> f32 {
		let health = |side: &Team| -> f32 {
			state
				.roster
				.team(side)
				.iter()
				.filter_map(|slot| slot.as_ref())
				.map(|mon| mon.current_hp as f32 / mon.max_hp.max(1) as f32)
				.sum()
		};

		let count = |side: &Team| -> usize {
			state.roster.team(side).iter().filter(|slot| slot.is_some()).count()
		};

		let other = team.other();
		// Divide by the larger side so the result lands in -1..1 whatever the
		// team sizes are, and stays comparable with a real win or loss.
		let scale = count(&team).max(count(&other)).max(1) as f32;
		((health(&team) - health(&other)) / scale).clamp(-1.0, 1.0)
	}
}

/// Scores every unfinished position as a draw.
///
/// Useful for isolating the solver from the heuristic: any difference between a
/// solve with this and one with [`HealthHeuristic`] is the leaf estimate talking,
/// not the CFR.
pub struct Indifferent;

impl LeafEvaluator for Indifferent {
	fn value(&self, _state: &BattleState, _team: Team, _registry: &Registry) -> f32 {
		0.0
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::field::PositionId;
	use crate::cfr::position::known_answer_duel;

	#[test]
	fn an_even_position_is_worth_nothing_to_either_side() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);

		// Not a mirror, but both sides are at full health, which is all this
		// heuristic looks at.
		assert!(HealthHeuristic.value(&state, Team::Zero, &registry).abs() < 1e-6);
	}

	#[test]
	fn being_ahead_on_health_is_worth_something_to_the_side_that_is_ahead() {
		let registry = Registry::load();
		let mut state = known_answer_duel(&registry);
		state.get_mut_mon(PositionId(1)).unwrap().current_hp /= 2;

		let zero = HealthHeuristic.value(&state, Team::Zero, &registry);
		let one = HealthHeuristic.value(&state, Team::One, &registry);

		assert!(zero > 0.0, "the healthier side should be ahead, got {zero}");
		assert!(
			(zero + one).abs() < 1e-6,
			"the estimate has to be zero-sum: {zero} vs {one}",
		);
	}

	#[test]
	fn a_wiped_out_side_is_valued_as_a_loss() {
		let registry = Registry::load();
		let mut state = known_answer_duel(&registry);
		state.get_mut_mon(PositionId(1)).unwrap().current_hp = 0;

		assert!((HealthHeuristic.value(&state, Team::Zero, &registry) - 1.0).abs() < 1e-6);
	}

	#[test]
	fn the_estimate_never_exceeds_a_real_result() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);
		for team in [Team::Zero, Team::One] {
			let value = HealthHeuristic.value(&state, team, &registry);
			assert!((-1.0..=1.0).contains(&value), "out of range: {value}");
		}
	}
}
