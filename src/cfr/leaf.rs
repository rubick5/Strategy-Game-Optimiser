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

use std::error::Error;
use std::rc::Rc;

use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::volatile::VolatileKind;
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

/// Remaining health, plus the four other resources already on the board.
///
/// [`HealthHeuristic`] reads two of `CreatureState`'s fields, `current_hp` and
/// `max_hp`, and ignores `stat_changes`, `non_vol_status` and `volatiles`
/// entirely. So a creature at +2 Attack scores exactly the same as one at +0, and
/// *blade dance is not undervalued by it — it is invisible to it*. The same goes
/// for a Substitute standing in front of you and for the poison that is going to
/// take a quarter of your health over the next three turns.
///
/// That blindness is what forces the search to go deep. A deeper solve does not
/// improve the estimate; it grinds on until the boost has converted itself into
/// damage, which is the one currency the estimate can see. Measured on
/// `six_asymmetric`, pricing a stonewarden's blade dance set against its guard
/// set moved by 0.032 between two and three turns of lookahead — on a matrix
/// whose whole range is 0.09.
///
/// **This is not the matchup-aware heuristic that was argued against**, and the
/// distinction is the point. Scoring a position by type effectiveness encodes a
/// *prediction about what will happen* — "the 4x move lands" — which pre-empts
/// the game theory, since the opponent can simply Protect in front of it. A stat
/// stage is not a prediction. It is a fact about the board right now, exactly
/// like current HP, and poison is nothing more than health already lost but not
/// yet debited. Counting them is bookkeeping, not strategy.
///
/// Every coefficient below is a guess, and deliberately a *tunable* one. Until
/// [`crate::cfr::contraction`] existed there was no way to tell a good guess from
/// a bad one; now each can be scored by R^2 against exactly solved duels, so
/// these are starting points to be fitted rather than claims.
pub struct ResourceHeuristic {
	/// Health-bar fractions per unit of weighted stat stage, for a creature at
	/// full health. A +2 Attack is worth about an eighth of a bar at this default.
	pub boost: f32,
	/// Scale on the status penalties, which are quoted below in bar fractions.
	pub status: f32,
	/// Cost of being Leech Seeded.
	pub seed: f32,
	/// Cost of being confused, which is turns rather than health.
	pub confusion: f32,
}

impl Default for ResourceHeuristic {
	fn default() -> Self {
		ResourceHeuristic { boost: 0.10, status: 1.0, seed: 0.12, confusion: 0.06 }
	}
}

impl ResourceHeuristic {
	/// Deferred health loss, as a fraction of a bar.
	///
	/// Poison ticks for a fixed share of maximum health every turn, so its cost is
	/// that share times however many turns the creature has left — which is itself
	/// roughly proportional to its remaining health. Burn adds a cut to physical
	/// damage on top of a smaller tick, and paralysis costs turns rather than
	/// health.
	fn status_cost(&self, status: NonVolatileStatus) -> f32 {
		self.status
			* match status {
				NonVolatileStatus::NoStatus => 0.0,
				NonVolatileStatus::Poison => 0.10,
				NonVolatileStatus::BadPoison => 0.18,
				NonVolatileStatus::Burn => 0.15,
				NonVolatileStatus::Paralysis => 0.12,
			}
	}

	/// What the stat stages are worth, in bar fractions.
	///
	/// Offence is taken as the better of the two attacking stages, on the reading
	/// that a creature uses whichever side of its moveset the boost helps. That is
	/// optimistic for a creature holding only the wrong kind of move, and reading
	/// the moveset to find out is a refinement this deliberately leaves for when
	/// the coefficients are being fitted rather than guessed.
	///
	/// Scaled by remaining health, because a boost on a creature about to faint
	/// buys nothing. This is also what keeps a fainted creature contributing
	/// exactly zero, the same as it does to the health term.
	fn boost_value(&self, mon: &CreatureState) -> f32 {
		let stages = &mon.stat_changes;
		let offence = stages.attack.max(stages.special_attack) as f32;
		let defence = (stages.defense as f32 + stages.special_defense as f32) / 2.0;
		let tempo = stages.speed as f32;

		self.boost * (0.6 * offence + 0.3 * defence + 0.2 * tempo)
	}

	/// One side's total, in bar-equivalents. Antisymmetry is preserved because
	/// this is computed per side and the two are subtracted.
	fn score(&self, state: &BattleState, side: &Team) -> f32 {
		state
			.roster
			.team(side)
			.iter()
			.filter_map(|slot| slot.as_ref())
			.map(|mon| {
				let max = mon.max_hp.max(1) as f32;
				let health = mon.current_hp as f32 / max;
				if mon.current_hp == 0 {
					// Nothing a fainted creature carries is worth anything, and
					// leaving it in would break the wiped-side-is-a-loss property.
					return 0.0;
				}

				// A Substitute is not like extra health, it *is* extra health:
				// damage has to chew through it before reaching the creature.
				let substitute =
					mon.volatiles.value(VolatileKind::Substitute) as f32 / max;

				let mut penalties = self.status_cost(mon.non_vol_status);
				if mon.volatiles.has(VolatileKind::LeechSeed) {
					// The drain also *credits* whoever planted it, which is not
					// modelled here because the volatile does not name them. So
					// this under-counts the swing rather than over-counting it.
					penalties += self.seed;
				}
				if mon.volatiles.has(VolatileKind::Confusion) {
					penalties += self.confusion;
				}

				// Deferred losses cannot exceed what is left to lose.
				let penalties = penalties.min(health + substitute);

				health + substitute + self.boost_value(mon) * health - penalties
			})
			.sum()
	}
}

impl LeafEvaluator for ResourceHeuristic {
	fn value(&self, state: &BattleState, team: Team, _registry: &Registry) -> f32 {
		let other = team.other();
		let count = |side: &Team| {
			state.roster.team(side).iter().filter(|slot| slot.is_some()).count()
		};
		let scale = count(&team).max(count(&other)).max(1) as f32;
		((self.score(state, &team) - self.score(state, &other)) / scale).clamp(-1.0, 1.0)
	}
}

/// One leaf estimate shared by many solvers.
///
/// A `Solver` owns its evaluator, so re-solving builds a new one each time — and
/// an evaluator that caches expensive work would throw the cache away on every
/// solve. [`crate::cfr::resolve::ResolvingPolicy`] re-solves constantly, so this
/// matters: sharing one cache across thousands of solves is the difference
/// between a rollout-based leaf being usable and being unaffordable.
pub struct SharedLeaf(pub Rc<dyn LeafEvaluator>);

impl LeafEvaluator for SharedLeaf {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		self.0.value(state, team, registry)
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

/// Which leaf estimate to solve with, chosen at the command line.
///
/// Built once and handed out behind an [`Rc`], because a `Solver` owns its
/// evaluator and the diagnostics build hundreds of solvers — re-reading a
/// critic's weights from disk for each one would dominate the run.
pub struct LeafSource {
	inner: Rc<dyn LeafEvaluator>,
	label: String,
}

impl LeafSource {
	/// `health`, `resource`, or `critic:<path>`.
	pub fn parse(spec: &str) -> Result<Self, Box<dyn Error>> {
		let (inner, label): (Rc<dyn LeafEvaluator>, String) = match spec {
			"health" => (Rc::new(HealthHeuristic), String::from("health heuristic")),
			"resource" => (
				Rc::new(ResourceHeuristic::default()),
				String::from("resource heuristic (health, boosts, status, volatiles)"),
			),
			"indifferent" => (Rc::new(Indifferent), String::from("indifferent (every leaf a draw)")),
			other => match other.strip_prefix("critic:") {
				Some(path) => {
					let mut critic = crate::cfr::critic::Critic::from_file(path)?;
					// A critic loads with whatever trust it was saved at, and the
					// default is zero — which is the health heuristic exactly, so
					// a run meaning to test the network would quietly test the
					// heuristic instead.
					critic.trust = 1.0;
					(Rc::new(critic), format!("critic from {path}, trust 1.0"))
				}
				None => {
					return Err(format!(
						"unknown leaf '{other}' (want health | resource | indifferent | critic:<path>)"
					)
					.into())
				}
			},
		};
		Ok(LeafSource { inner, label })
	}

	/// A fresh handle to the shared estimate, for a solver to own.
	pub fn make(&self) -> Box<dyn LeafEvaluator> {
		Box::new(SharedLeaf(self.inner.clone()))
	}

	pub fn label(&self) -> &str {
		&self.label
	}
}

impl Default for LeafSource {
	fn default() -> Self {
		LeafSource { inner: Rc::new(HealthHeuristic), label: String::from("health heuristic") }
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

	// --- ResourceHeuristic ------------------------------------------------

	use crate::battle::state::stat_stages::StatStages;
	use crate::battle::state::volatile::Volatile;
	use crate::model::speciesdata::Stat;

	/// The one property whose failure would not look like a failure. A leaf that
	/// is not zero-sum means the solver is no longer solving a zero-sum game, and
	/// it would carry on returning confident, meaningless values.
	#[test]
	fn the_resource_estimate_is_zero_sum_with_every_resource_in_play() {
		let registry = Registry::load();
		let leaf = ResourceHeuristic::default();
		let mut state = known_answer_duel(&registry);

		let mon = state.get_mut_mon(PositionId(0)).unwrap();
		mon.current_hp = mon.max_hp / 2;
		mon.stat_changes = StatStages::from_pairs(vec![(Stat::Attack, 2), (Stat::Speed, -1)]);
		mon.non_vol_status = NonVolatileStatus::BadPoison;
		mon.volatiles.add(Volatile::new(VolatileKind::LeechSeed));

		let other = state.get_mut_mon(PositionId(1)).unwrap();
		other.current_hp = (other.max_hp * 3) / 4;
		other.volatiles.add(Volatile::new(VolatileKind::Confusion));
		// `set_value` is a no-op on a condition that is not there, so the
		// Substitute has to be added carrying its health rather than set after.
		let quarter = other.max_hp / 4;
		other
			.volatiles
			.add(Volatile::new(VolatileKind::Substitute).with_value(quarter));

		let zero = leaf.value(&state, Team::Zero, &registry);
		let one = leaf.value(&state, Team::One, &registry);
		assert!((zero + one).abs() < 1e-5, "not zero-sum: {zero} against {one}");
	}

	/// The whole point of the thing: blade dance has to be *visible*.
	#[test]
	fn a_boost_is_worth_something_and_the_health_heuristic_cannot_see_it() {
		let registry = Registry::load();
		let flat = known_answer_duel(&registry);
		let mut boosted = flat.clone();
		boosted.get_mut_mon(PositionId(0)).unwrap().stat_changes =
			StatStages::from_pairs(vec![(Stat::Attack, 2)]);

		let health_flat = HealthHeuristic.value(&flat, Team::Zero, &registry);
		let health_boosted = HealthHeuristic.value(&boosted, Team::Zero, &registry);
		assert_eq!(
			health_flat, health_boosted,
			"the health heuristic is supposed to be blind to this — the premise has changed",
		);

		let leaf = ResourceHeuristic::default();
		assert!(
			leaf.value(&boosted, Team::Zero, &registry)
				> leaf.value(&flat, Team::Zero, &registry) + 1e-4,
			"a +2 Attack should be worth something",
		);
	}

	/// A boost on a creature that is about to faint buys nothing, so its value has
	/// to scale with what is left. Without this a dying creature could be scored
	/// above a healthy one.
	#[test]
	fn a_boost_is_worth_less_on_a_creature_that_is_nearly_dead() {
		let registry = Registry::load();
		let mut healthy = known_answer_duel(&registry);
		healthy.get_mut_mon(PositionId(0)).unwrap().stat_changes =
			StatStages::from_pairs(vec![(Stat::Attack, 2)]);
		let mut dying = healthy.clone();
		let mon = dying.get_mut_mon(PositionId(0)).unwrap();
		mon.current_hp = mon.max_hp / 20;

		let leaf = ResourceHeuristic::default();
		let gain = |state: &BattleState| {
			let mut flat = state.clone();
			flat.get_mut_mon(PositionId(0)).unwrap().stat_changes = StatStages::new();
			leaf.value(state, Team::Zero, &registry) - leaf.value(&flat, Team::Zero, &registry)
		};

		assert!(
			gain(&healthy) > gain(&dying),
			"boost worth {} at full health and {} at 5%",
			gain(&healthy),
			gain(&dying),
		);
	}

	/// Poison is health already lost but not yet debited, so it has to cost
	/// something — and never more than there is left to lose.
	#[test]
	fn status_costs_something_and_never_more_than_remains() {
		let registry = Registry::load();
		let leaf = ResourceHeuristic::default();

		let clean = known_answer_duel(&registry);
		let mut poisoned = clean.clone();
		poisoned.get_mut_mon(PositionId(0)).unwrap().non_vol_status = NonVolatileStatus::BadPoison;
		assert!(
			leaf.value(&poisoned, Team::Zero, &registry)
				< leaf.value(&clean, Team::Zero, &registry) - 1e-4,
		);

		// A sliver of health left and a status that nominally costs more than
		// that: the penalty must not push the side below a clean loss.
		let mut doomed = clean.clone();
		let mon = doomed.get_mut_mon(PositionId(0)).unwrap();
		mon.current_hp = 1;
		mon.non_vol_status = NonVolatileStatus::BadPoison;
		assert!(leaf.value(&doomed, Team::Zero, &registry) >= -1.0);
	}

	/// A Substitute is not like extra health, it is extra health.
	#[test]
	fn a_substitute_counts_as_the_health_it_is() {
		let registry = Registry::load();
		let leaf = ResourceHeuristic::default();
		let bare = known_answer_duel(&registry);

		let mut screened = bare.clone();
		let max = screened.get_mon(PositionId(0)).unwrap().max_hp;
		screened
			.get_mut_mon(PositionId(0))
			.unwrap()
			.volatiles
			.add(Volatile::new(VolatileKind::Substitute).with_value(max / 4));

		assert!(
			leaf.value(&screened, Team::Zero, &registry)
				> leaf.value(&bare, Team::Zero, &registry) + 1e-4,
		);
	}

	/// Whatever else it counts, a side with nothing left standing has lost.
	#[test]
	fn a_wiped_out_side_is_still_a_loss_however_it_was_decorated() {
		let registry = Registry::load();
		let mut state = known_answer_duel(&registry);
		let mon = state.get_mut_mon(PositionId(1)).unwrap();
		mon.current_hp = 0;
		// A fainted creature keeps whatever it was carrying; none of it may count.
		mon.stat_changes = StatStages::from_pairs(vec![(Stat::Attack, 6)]);
		mon.non_vol_status = NonVolatileStatus::Burn;

		let value = ResourceHeuristic::default().value(&state, Team::Zero, &registry);
		assert!((value - 1.0).abs() < 1e-6, "expected a win, got {value}");
	}

	#[test]
	fn a_leaf_source_is_parsed_from_its_name_and_rejects_nonsense() {
		assert!(LeafSource::parse("health").is_ok());
		assert!(LeafSource::parse("resource").is_ok());
		assert!(LeafSource::parse("indifferent").is_ok());
		assert!(LeafSource::parse("critic:/no/such/file.json").is_err());
		assert!(LeafSource::parse("nonsense").is_err());
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
