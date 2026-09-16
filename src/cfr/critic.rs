//! A value network the solver trains for itself.
//!
//! # Why
//!
//! A depth-limited solve is only as good as its guess at the horizon, and
//! [`HealthHeuristic`](crate::cfr::leaf::HealthHeuristic) guesses from remaining
//! HP alone. It cannot see a type matchup, a status condition or a creature that
//! is healthy but walled, so it systematically undervalues a switch made for
//! position rather than for damage — which is exactly the decision a 2v2 turns
//! on. Every 2v2 answer currently rests on that.
//!
//! So the solver learns its own estimate instead, from its own play. Nothing here
//! touches the PPO agent: it borrows [`NeuralNet`](crate::rl::nn::neural_net) and
//! the state encoder, both of which are plain utilities with no learner attached,
//! and defines its own activations rather than importing the ones that live
//! beside PPO.
//!
//! # How the targets are made, and why this does not eat its own tail
//!
//! The obvious loop — solve with the current estimate, train on the values that
//! solve produced, repeat — has an obvious failure. A wrong estimate makes a
//! wrong solve, which makes wrong targets, which keeps the estimate wrong: a
//! stable fixed point that is not the truth, reached with no sign anything went
//! amiss.
//!
//! Two things keep that at bay here.
//!
//! **Targets are real outcomes, not the solver's own opinion.** After solving, the
//! position is *played to an actual win or loss* under the solved strategies, and
//! every position along the way is labelled with what really happened. That is a
//! Monte Carlo target: noisier than bootstrapping off the solver's own numbers,
//! and not an estimate of an estimate.
//!
//! **The curriculum starts where the answer is known.** A 1v1 ends on its own, so
//! the solver searches it to a real result with no horizon and no guess anywhere
//! — those positions are ground truth. Training on them first anchors the network
//! before it is asked about 2v2 positions where the horizon does bind. See
//! [`default_curriculum`].
//!
//! # Does it work? Not yet — and the measurement is the point
//!
//! Solving the 2v2 and measuring exploitability against it, lower being better:
//!
//! | leaf estimate | exploitability |
//! |---|---|
//! | health heuristic | **0.199** |
//! | trained critic, trust 0.25 | 0.320 |
//! | trained critic, trust 0.50 | 0.281 |
//! | trained critic, trust 1.00 | 0.559 |
//!
//! So the trained estimate is, for now, *worse* than the hand-written one, and
//! [`HealthHeuristic`] remains the default. Four formulations were tried —
//! replacing the heuristic outright, doing so with eight times the data, learning
//! a correction, and shrinking that correction — and none beat it.
//!
//! The diagnosis is target noise rather than anything structural. A label here is
//! the result of a *single* game, so a position whose true value is near zero is
//! labelled +1 or -1, and the network spends its capacity fitting that coin flip.
//! Mean squared error sits around 0.62-0.67 across every run and barely moves
//! between rounds, which is what it looks like when there is little signal per
//! sample to extract. The corrections that come out are consequently much larger
//! than the heuristic's real error, and CFR optimises against them.
//!
//! What would fix it, in the order worth trying:
//!
//! 1. **Average the targets.** Label a position with the mean of many playouts
//!    from *that* position rather than one, or bootstrap from the values the
//!    solver already computes. Either cuts the variance that is currently
//!    drowning the signal.
//! 2. **Train on more distinct positions.** Fourteen thousand samples drawn from
//!    three roots are heavily correlated — they are mostly the same few games.
//! 3. **A better optimiser.** [`NeuralNet`] does per-sample SGD with no Adam, no
//!    minibatching and no input normalisation, which makes this slow and noisy to
//!    fit. PPO would benefit from the same work.
//!
//! It is worth saying why this is known at all. Nothing in the training run looks
//! wrong — the error falls a little, the rounds complete, the network trains.
//! Without [`crate::cfr::exploit`] measuring the thing that actually matters, this
//! would have shipped as an improvement.
//!
//! # A correction, not a replacement
//!
//! The network does not estimate the position directly. It estimates what the
//! health heuristic gets *wrong*, and the two are added.
//!
//! Replacing the heuristic outright was tried first and came out worse — solving
//! the 2v2 against a freshly trained network gave exploitability 0.26 where the
//! plain heuristic gave 0.20. The reason is not that the network had learned
//! nothing; it is that a half-trained network is *noisy*, while the heuristic,
//! crude as it is, is smooth, monotone in health and exactly zero-sum. CFR
//! optimises against whatever it is given, noise included, so a rough estimate
//! that is wrong in a consistent direction beats one that is wrong in random
//! directions.
//!
//! Learning the residual keeps the heuristic's shape and asks the network only
//! for what it can genuinely add — the type matchups and positional judgements
//! health cannot express. An untrained correction is small, so the estimate
//! starts near the heuristic instead of far from it, and improves from there.
//!
//! # Antisymmetry
//!
//! The health heuristic is zero-sum by construction: what one side gains the
//! other loses. A network is not, and a solver handed an estimate where both
//! sides can be winning will happily find value that does not exist. So the
//! network is asked about *both* perspectives and the answer is the difference,
//! halved — and it is trained on both views of every sample, with opposite signs,
//! so the property is learned as well as enforced.

use std::cell::RefCell;
use std::error::Error;

use rand::seq::SliceRandom;
use rand::RngCore;

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::{Outcome, Team};
use crate::cfr::leaf::{HealthHeuristic, LeafEvaluator};
use crate::cfr::node::DecisionNode;
use crate::cfr::position::{known_answer_duel, mirror_duel, switch_prediction_2v2};
use crate::cfr::solver::{Solver, SolverConfig, PLAYOUT_CAP};
use crate::model::registry::Registry;
use crate::rl::encoder::{encode_both, TOTAL_ENCODING_LEN};
use crate::rl::nn::neural_net::NeuralNet;

/// Leak on the negative side, matching what the rest of the project uses.
const RELU_LEAK: f32 = 0.01;

fn relu(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK * x } else { x }
}

fn relu_prime(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK } else { 1.0 }
}

/// Hidden layer widths. The same shape PPO's critic uses, which is already known
/// to cope with an encoding this wide.
const HIDDEN: [usize; 2] = [128, 128];

/// Default share of the learned correction to apply. See [`Critic::trust`].
const DEFAULT_TRUST: f32 = 0.25;

/// One position, and how far the health heuristic was off for the side looking
/// at it.
pub struct Sample {
	pub encoding: Vec<f32>,
	/// The real result minus the heuristic's guess — what the network has to add.
	pub target: f32,
}

/// The health heuristic, plus a learned correction to it.
#[derive(Clone)]
pub struct Critic {
	// `NeuralNet::forward` caches activations, so it needs `&mut`. The evaluator
	// interface is `&self`, because a leaf estimate is a question, not a change.
	net: RefCell<NeuralNet<fn(f32) -> f32>>,
	/// How much of the learned correction to actually apply.
	///
	/// Targets are single-game outcomes, so a position whose true value is 0.1 is
	/// labelled +1 or -1 and the network spends most of its capacity fitting that
	/// noise. Its corrections come out far larger than the real error in the
	/// heuristic, and applied whole they swamp the thing they were meant to
	/// refine. Shrinking them toward the baseline is the standard answer to an
	/// estimator fitted on noisy targets.
	pub trust: f32,
}

impl Critic {
	pub fn new_random(rng: &mut dyn RngCore) -> Self {
		let mut sizes = vec![TOTAL_ENCODING_LEN];
		sizes.extend_from_slice(&HIDDEN);
		sizes.push(1);
		Critic {
			net: RefCell::new(NeuralNet::gen_random(
				rng,
				&sizes,
				relu as fn(f32) -> f32,
				relu_prime as fn(f32) -> f32,
			)),
			trust: DEFAULT_TRUST,
		}
	}

	pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
		Ok(Critic {
			net: RefCell::new(NeuralNet::from_file(
				path,
				relu as fn(f32) -> f32,
				relu_prime as fn(f32) -> f32,
			)?),
			trust: DEFAULT_TRUST,
		})
	}

	pub fn to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
		self.net.borrow().to_file(path)
	}

	/// The network's raw opinion of one perspective.
	fn raw(&self, encoding: &[f32]) -> f32 {
		self.net.borrow_mut().forward(encoding)[0]
	}

	/// One pass of squared-error regression over the samples.
	///
	/// Returns the mean squared error before the pass, which is the number to
	/// watch across rounds.
	pub fn fit(&mut self, samples: &mut [Sample], learning_rate: f32, rng: &mut dyn RngCore) -> f32 {
		if samples.is_empty() {
			return 0.0;
		}
		// Per-sample SGD, so the order matters; leaving it alone would walk the
		// network through one position's whole game before seeing another's.
		samples.shuffle(&mut RngWrapper(rng));

		let mut squared_error = 0.0;
		for sample in samples.iter() {
			let predicted = self.raw(&sample.encoding);
			squared_error += (predicted - sample.target).powi(2);

			// d/dv of (v - target)^2, clamped as PPO's critic does — an early
			// network can be wildly wrong and a raw gradient destabilises it.
			let gradient = (2.0 * (predicted - sample.target)).clamp(-10.0, 10.0);
			self.net
				.borrow_mut()
				.backward(vec![gradient], &sample.encoding, learning_rate);
		}
		squared_error / samples.len() as f32
	}
}

impl Critic {
	/// The learned correction for one side, antisymmetric by construction.
	fn correction(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		let (zero_view, one_view) = encode_both(state, registry, false);
		let (mine, theirs) = match team {
			Team::Zero => (zero_view, one_view),
			Team::One => (one_view, zero_view),
		};
		// Halved so the pair of opinions stays on the -1..1 scale of a result.
		(self.raw(&mine) - self.raw(&theirs)) / 2.0
	}
}

impl LeafEvaluator for Critic {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		let base = HealthHeuristic.value(state, team, registry);
		(base + self.trust * self.correction(state, team, registry)).clamp(-1.0, 1.0)
	}
}

/// A position to train on, and how far the solver should look at it.
pub struct TrainingPosition {
	pub name: String,
	pub state: BattleState,
	/// A horizon past the length of the battle means "search it to the end",
	/// which is what makes a position ground truth rather than an estimate.
	pub lookahead: usize,
}

/// Anchor on positions with known answers, then move to ones that need a guess.
///
/// The two 1v1s end on their own and are searched to real results, so their
/// labels are exact. The 2v2 is the position the critic actually exists to
/// improve, and it is the only one whose labels depend on the estimate being
/// trained — which is why it is not the only thing trained on.
pub fn default_curriculum(registry: &Registry) -> Vec<TrainingPosition> {
	vec![
		TrainingPosition {
			name: String::from("known-answer duel (1v1, exact)"),
			state: known_answer_duel(registry),
			lookahead: 200,
		},
		TrainingPosition {
			name: String::from("mirror duel (1v1, exact)"),
			state: mirror_duel(registry),
			lookahead: 200,
		},
		TrainingPosition {
			name: String::from("switch prediction (2v2, horizon-limited)"),
			state: switch_prediction_2v2(registry),
			lookahead: 6,
		},
	]
}

pub struct TrainingConfig {
	/// Solve-then-train cycles. Each one re-solves with the critic trained so
	/// far, so later rounds learn from better play.
	pub rounds: usize,
	pub solver_iterations: usize,
	/// Games played per position per round to produce labels.
	pub playouts: usize,
	/// Passes over the collected samples each round.
	pub epochs: usize,
	pub learning_rate: f32,
}

impl Default for TrainingConfig {
	fn default() -> Self {
		TrainingConfig {
			rounds: 4,
			solver_iterations: 1_500,
			playouts: 150,
			epochs: 3,
			learning_rate: 0.002,
		}
	}
}

/// What one round of training did, for reporting.
pub struct RoundReport {
	pub round: usize,
	pub samples: usize,
	pub mean_squared_error: f32,
}

/// Train a critic by solving positions and learning from how they actually turn
/// out.
pub fn train(
	registry: &Registry,
	curriculum: &[TrainingPosition],
	config: &TrainingConfig,
	rng: &mut dyn RngCore,
) -> (Critic, Vec<RoundReport>) {
	let mut critic = Critic::new_random(rng);
	let mut reports = Vec::new();

	for round in 0..config.rounds {
		let mut samples: Vec<Sample> = Vec::new();

		for position in curriculum {
			// Round zero has nothing trained yet, so it plays off the hand-written
			// heuristic. Every later round solves with what has been learned, and
			// the labels improve because the play does.
			let leaf: Box<dyn LeafEvaluator> = if round == 0 {
				Box::new(HealthHeuristic)
			} else {
				Box::new(critic.clone())
			};

			let mut solver = Solver::with_leaf(
				registry,
				SolverConfig::for_lookahead(config.solver_iterations, position.lookahead),
				leaf,
			);
			solver.solve(&position.state, rng);

			for _ in 0..config.playouts {
				collect(&solver, &position.state, registry, rng, &mut samples);
			}
		}

		let mut error = 0.0;
		for _ in 0..config.epochs {
			error = critic.fit(&mut samples, config.learning_rate, rng);
		}

		reports.push(RoundReport { round, samples: samples.len(), mean_squared_error: error });
	}

	(critic, reports)
}

/// Play one game under the solved strategies and label every position in it with
/// the result.
///
/// Both perspectives of each position are recorded, with opposite targets. That
/// doubles the data and, more usefully, teaches the network the zero-sum property
/// directly rather than relying on the halving in [`Critic::value`] to paper over
/// its absence.
fn collect(
	solver: &Solver,
	root: &BattleState,
	registry: &Registry,
	rng: &mut dyn RngCore,
	out: &mut Vec<Sample>,
) {
	let mut state = root.clone();
	let mut request = StepRequest::NeedsActions;
	let mut seen: Vec<((Vec<f32>, Vec<f32>), f32)> = Vec::new();

	let mut outcome_to_zero = None;
	for _ in 0..PLAYOUT_CAP {
		let actors = match DecisionNode::from(&state, &request) {
			DecisionNode::Terminal(outcome) => {
				outcome_to_zero = Some(match outcome {
					Outcome::Win { team } => if team == Team::Zero { 1.0 } else { -1.0 },
					Outcome::Draw => 0.0,
				});
				break;
			}
			DecisionNode::Decision { actors } => actors,
		};

		// The heuristic's own opinion is recorded alongside, because the network
		// is trained on what it misses rather than on the result itself.
		seen.push((
			encode_both(&state, registry, false),
			HealthHeuristic.value(&state, Team::Zero, registry),
		));

		let commands = actors
			.iter()
			.map(|actor| {
				let strategy = solver
					.average_strategy(&state, &request, actor.team)
					.unwrap_or_else(|| uniform(actor.mask.allowed.iter().filter(|a| **a).count(), &actor.mask));
				let action = crate::cfr::infoset::sample(&strategy, rng);
				actor.command(action, &state, registry)
			})
			.collect();

		let StepResult { battle_state, step_request } =
			engine::step(state, commands, registry, rng);
		state = battle_state;
		request = step_request;
	}

	// A game that never finished has no real label, so it teaches nothing and is
	// dropped rather than being guessed at.
	let Some(result) = outcome_to_zero else {
		return;
	};

	for ((zero_view, one_view), heuristic) in seen {
		// What the heuristic missed, from each side. The two are negatives of each
		// other, so training on both teaches the zero-sum property directly.
		let residual = result - heuristic;
		out.push(Sample { encoding: zero_view, target: residual });
		out.push(Sample { encoding: one_view, target: -residual });
	}
}

fn uniform(legal: usize, mask: &crate::rl::mask::Mask) -> crate::cfr::infoset::Strategy {
	let mut strategy = [0.0; crate::rl::moveslot::MAX_DECISION];
	if legal == 0 {
		return strategy;
	}
	let share = 1.0 / legal as f32;
	for action in 0..crate::rl::moveslot::MAX_DECISION {
		if mask.allowed[action] {
			strategy[action] = share;
		}
	}
	strategy
}

/// `shuffle` wants `Rng`, and the rest of the project passes `&mut dyn RngCore`.
struct RngWrapper<'a>(&'a mut dyn RngCore);

impl rand::RngCore for RngWrapper<'_> {
	fn next_u32(&mut self) -> u32 {
		self.0.next_u32()
	}
	fn next_u64(&mut self) -> u64 {
		self.0.next_u64()
	}
	fn fill_bytes(&mut self, dest: &mut [u8]) {
		self.0.fill_bytes(dest)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	/// A zero-sum game cannot have both sides ahead. The heuristic guarantees this
	/// structurally; the network only does because it is asked about both
	/// perspectives and given the difference, so it is worth pinning down.
	#[test]
	fn an_estimate_is_worth_the_opposite_to_each_side() {
		let registry = Registry::load();
		let mut rng = StdRng::seed_from_u64(1);
		let critic = Critic::new_random(&mut rng);
		let state = switch_prediction_2v2(&registry);

		let zero = critic.value(&state, Team::Zero, &registry);
		let one = critic.value(&state, Team::One, &registry);

		assert!((zero + one).abs() < 1e-5, "not zero-sum: {zero} vs {one}");
	}

	/// Estimates share a scale with real results, or a truncated line cannot be
	/// compared with a finished one.
	#[test]
	fn estimates_stay_on_the_result_scale() {
		let registry = Registry::load();
		let mut rng = StdRng::seed_from_u64(2);
		let critic = Critic::new_random(&mut rng);

		for state in [known_answer_duel(&registry), switch_prediction_2v2(&registry)] {
			for team in [Team::Zero, Team::One] {
				let value = critic.value(&state, team, &registry);
				assert!((-1.0..=1.0).contains(&value), "out of range: {value}");
			}
		}
	}

	/// An untrained network has nothing to say, so the estimate should still be
	/// roughly the heuristic rather than something unrelated to it.
	#[test]
	fn an_untrained_critic_stays_close_to_the_heuristic() {
		let registry = Registry::load();
		let mut rng = StdRng::seed_from_u64(3);
		let mut critic = Critic::new_random(&mut rng);
		critic.trust = 0.0;
		let state = switch_prediction_2v2(&registry);

		let learned = critic.value(&state, Team::Zero, &registry);
		let plain = HealthHeuristic.value(&state, Team::Zero, &registry);
		assert!((learned - plain).abs() < 1e-5, "{learned} vs {plain}");
	}

	/// The loop runs end to end: solve, play out, label, fit.
	#[test]
	fn training_collects_labels_and_fits_them() {
		let registry = Registry::load();
		let mut rng = StdRng::seed_from_u64(4);

		let curriculum = vec![TrainingPosition {
			name: String::from("known-answer duel"),
			state: known_answer_duel(&registry),
			lookahead: 200,
		}];
		let config = TrainingConfig {
			rounds: 2,
			solver_iterations: 50,
			playouts: 20,
			epochs: 1,
			learning_rate: 0.001,
		};

		let (_, reports) = train(&registry, &curriculum, &config, &mut rng);

		assert_eq!(reports.len(), 2);
		for report in &reports {
			assert!(report.samples > 0, "round {} collected nothing", report.round);
			assert!(report.mean_squared_error.is_finite());
		}
	}

	#[test]
	fn a_critic_survives_a_round_trip_through_a_file() {
		let registry = Registry::load();
		let mut rng = StdRng::seed_from_u64(5);
		let critic = Critic::new_random(&mut rng);
		let state = switch_prediction_2v2(&registry);
		let before = critic.value(&state, Team::Zero, &registry);

		let path = std::env::temp_dir().join("cfr_critic_roundtrip.json");
		let path = path.to_str().unwrap();
		critic.to_file(path).unwrap();
		let loaded = Critic::from_file(path).unwrap();

		let after = loaded.value(&state, Team::Zero, &registry);
		assert!((before - after).abs() < 1e-6, "{before} vs {after}");
		let _ = std::fs::remove_file(path);
	}
}
