//! A value network the solver trains for itself.
//!
//! # Why
//!
//! A depth-limited solve is only as good as its guess at the horizon, and
//! [`HealthHeuristic`] guesses from remaining HP alone. It cannot see a type
//! matchup or a creature that is healthy but walled, so it undervalues a switch
//! made for position rather than damage — which is the decision a 2v2 turns on.
//!
//! Nothing here touches the PPO agent. It borrows [`NeuralNet`] and the state
//! encoder, both plain utilities with no learner attached, and defines its own
//! activations rather than importing the ones that live beside PPO.
//!
//! # The problem this module is mostly about
//!
//! A label is the result of one game: `+1`, `-1` or `0`. What the critic needs to
//! predict is the *expected* result — a position worth 0.1 means "you win about
//! 55% of the time", and no single playout ever returns 0.1. So every raw label is
//! `true value + noise`, where the noise variance is about `1 - v²`: for a
//! near-even position, essentially a coin flip, and irreducible from one sample.
//!
//! A first version of this trained on exactly those raw labels and made the
//! solver **more** exploitable — 0.28 against the plain heuristic's 0.20. The
//! network had not failed to learn; it had learned the wrong *scale*. Asked to
//! match targets with RMS 0.85, it settled on emitting values of that size (0.26
//! measured) when the genuinely predictable part is far smaller, and the
//! unpredictable remainder is by definition uncorrelated with the truth. Adding
//! 0.26 of noise to a signal of 0.21 more than doubles the noise floor of the
//! estimate, and CFR faithfully optimises against whatever it is handed.
//!
//! Rough arithmetic on the gap: with per-label noise near 0.9 and a real error in
//! the heuristic of maybe 0.15, the signal-to-noise per sample is about 0.17. It
//! takes on the order of **thirty labels per position** before the signal is even
//! visible. Everything below is a way of getting there.
//!
//! # Averaging by position
//!
//! The noise is zero-mean, so averaging `n` labels for the same position divides
//! its standard deviation by `sqrt(n)` and leaves the signal alone.
//!
//! Playouts overlap heavily — every one visits the root, many visit the turn-two
//! positions — so grouping labels by *exact position* and averaging buys most of
//! that for free, with no extra simulation. [`SampleSet`] does this, keyed on the
//! same [`StateKey`] the regret tables use. Aggregated samples carry a weight, so
//! a position averaged over four hundred games is not treated as equal to one
//! seen once.
//!
//! # Learning where it is used
//!
//! The critic is only ever *called* at horizon states — positions at exactly the
//! search depth. Training it on states drawn from playouts, which are shallow and
//! on-policy, fits the function in one place and uses it in another. That is
//! plain covariate shift, and it also misses a whole category: the search expands
//! every one of the traverser's actions, so horizon states include positions no
//! playout would ever reach.
//!
//! [`HorizonRecorder`] therefore samples the positions actually handed to the
//! leaf during a solve. Labelling them needs play *from* them, and the solved
//! strategies do not extend past the horizon — so each sampled position gets its
//! own short solve first, and is then played out under that. Playing uniformly
//! instead would label it with the value of random play, which is not the
//! quantity wanted.
//!
//! Root playouts are kept alongside, because on positions that end on their own
//! they reach real terminals and are therefore *ground truth*, with no estimate
//! anywhere in them. See [`default_curriculum`].
//!
//! # Refusing to act on what cannot be distinguished
//!
//! Two safeguards, because the fixes above reduce the noise rather than removing
//! it.
//!
//! [`Critic::trust`] scales the learned correction. When signal-to-noise is low
//! the right thing to do with an estimator is pull it toward the prior, which
//! here is the heuristic. Worth being clear that this limits damage rather than
//! creating signal: as trust falls to zero the critic approaches the baseline and
//! can never beat it.
//!
//! [`select_trust`] is what makes the whole thing safe. It measures
//! **exploitability** — the real objective — for a range of trust values
//! including zero, which *is* the plain heuristic, and keeps the best. So a
//! trained critic can never leave the solver worse off than not having one.
//!
//! # Where this actually stands
//!
//! The noise work did what it was meant to. Labels now average 8.7 games per
//! position, and training error falls across rounds — 0.30, 0.23, 0.195 — where
//! before it sat at 0.62-0.67 and barely moved. (Only the *fall* is evidence of
//! learning; the lower absolute figure is partly mechanical, since a target that
//! is a mean of eight games has an eighth the variance of a single one.)
//!
//! **The critic still does not beat the heuristic.** Six paired runs on the 2v2,
//! each comparing trust levels on identical chance draws:
//!
//! ```text
//! trust 0.1 minus baseline: mean -0.0557, sd 0.0653, stderr 0.0266
//! one run of six favoured the critic; selection chose trust 0.00 in the other five
//! ```
//!
//! A single earlier run had shown the critic ahead by 0.011, which was noise —
//! the sort of result that gets published by accident. The honest reading is that
//! cleaner labels were necessary and are not yet sufficient.
//!
//! The likeliest reason is simply that there is still not enough of them. 8.7
//! labels per position is well short of the thirty the arithmetic above asks for,
//! and only forty horizon positions are sampled per round. What remains untried,
//! in order: targets taken from the solver's own node values, which are averaged
//! over the whole subtree and quieter again than any number of playouts; and
//! minibatched gradients with an adaptive step, which would stop each individual
//! label yanking the weights around. Both are real work rather than tuning.
//!
//! Meanwhile [`DEFAULT_TRUST`] is zero, so none of this is on by default and
//! nothing regresses.

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::{Outcome, Team};
use crate::cfr::exploit::{measure, ExploitConfig};
use crate::cfr::infoset::{sample as sample_action, InfosetData};
use crate::cfr::key::StateKey;
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

/// Default share of the learned correction to apply: **none**.
///
/// A fresh critic is therefore exactly [`HealthHeuristic`] until [`select_trust`]
/// measures that some trust is an improvement and sets it. That is the safe
/// default on the evidence: across six paired runs, trust 0.1 came out 0.056
/// worse than the baseline on average and trust 0.25 worse still, so a critic
/// that trusted itself out of the box would be a regression.
///
/// Trust has to be earned by measurement, not assumed.
const DEFAULT_TRUST: f32 = 0.0;

// ---------------------------------------------------------------------------
// The estimate itself
// ---------------------------------------------------------------------------

/// One position, how far the heuristic was off for the side looking at it, and
/// how much evidence that is based on.
pub struct Sample {
	pub encoding: Vec<f32>,
	/// The real result minus the heuristic's guess — what the network has to add.
	pub target: f32,
	/// Relative confidence, from how many games were averaged into `target`.
	pub weight: f32,
}

/// The health heuristic, plus a learned correction to it.
///
/// A correction rather than a replacement: the heuristic is crude but smooth,
/// monotone in health and exactly zero-sum, and CFR does better with an estimate
/// that is wrong in a consistent direction than one wrong in random directions.
/// Learning the residual keeps that shape and asks the network only for what it
/// can genuinely add.
#[derive(Clone)]
pub struct Critic {
	// `NeuralNet::forward` caches activations, so it needs `&mut`. The evaluator
	// interface is `&self`, because a leaf estimate is a question, not a change.
	net: RefCell<NeuralNet<fn(f32) -> f32>>,
	/// How much of the learned correction to apply. Set by [`select_trust`].
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

	/// The learned correction for one side, antisymmetric by construction.
	fn correction(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		// The same flag the labels were collected under. A position owing a
		// replacement is not the position that does not, and encoding both as
		// though they were the same would train one input towards two targets.
		let (zero_view, one_view) = encode_both(state, registry, awaiting_replacement(state));
		let (mine, theirs) = match team {
			Team::Zero => (zero_view, one_view),
			Team::One => (one_view, zero_view),
		};
		// Halved so the pair of opinions stays on the -1..1 scale of a result.
		(self.raw(&mine) - self.raw(&theirs)) / 2.0
	}

	/// One weighted pass of squared-error regression over the samples.
	///
	/// Returns the mean squared error before the pass. Worth remembering what that
	/// number can and cannot tell you: most of it is irreducible label noise, so
	/// it barely moves even when the estimate improves. [`select_trust`] is the
	/// measurement that actually decides anything.
	pub fn fit(&mut self, samples: &mut [Sample], learning_rate: f32, rng: &mut dyn RngCore) -> f32 {
		if samples.is_empty() {
			return 0.0;
		}
		// Per-sample SGD, so order matters; leaving it alone would walk the network
		// through one position's whole game before seeing another's.
		samples.shuffle(&mut RngWrapper(rng));

		let mut squared_error = 0.0;
		let mut total_weight = 0.0;
		for sample in samples.iter() {
			let predicted = self.raw(&sample.encoding);
			squared_error += sample.weight * (predicted - sample.target).powi(2);
			total_weight += sample.weight;

			// d/dv of w(v - target)^2, clamped as PPO's critic does — an early
			// network can be wildly wrong and a raw gradient destabilises it.
			let gradient =
				(2.0 * sample.weight * (predicted - sample.target)).clamp(-10.0, 10.0);
			self.net
				.borrow_mut()
				.backward(vec![gradient], &sample.encoding, learning_rate);
		}
		if total_weight > 0.0 { squared_error / total_weight } else { 0.0 }
	}
}

impl LeafEvaluator for Critic {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		let base = HealthHeuristic.value(state, team, registry);
		(base + self.trust * self.correction(state, team, registry)).clamp(-1.0, 1.0)
	}
}

// ---------------------------------------------------------------------------
// Averaging labels by position
// ---------------------------------------------------------------------------

struct Accumulated {
	encoding: Vec<f32>,
	sum: f32,
	count: u32,
}

/// Labels grouped by the position they describe.
///
/// This is where most of the noise goes. Two hundred playouts from one root all
/// pass through that root, so it ends up with two hundred labels averaged into
/// one target — a fourteen-fold cut in noise, for no extra simulation. Positions
/// deeper in the game get fewer, which is why the count is carried through as a
/// weight rather than thrown away.
#[derive(Default)]
pub struct SampleSet {
	entries: HashMap<(StateKey, Team), Accumulated>,
}

impl SampleSet {
	pub fn new() -> Self {
		Self::default()
	}

	fn add(&mut self, key: StateKey, team: Team, encoding: Vec<f32>, target: f32) {
		let entry = self
			.entries
			.entry((key, team))
			.or_insert_with(|| Accumulated { encoding, sum: 0.0, count: 0 });
		entry.sum += target;
		entry.count += 1;
	}

	/// Distinct positions held.
	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	/// Raw labels collected, before averaging.
	pub fn labels(&self) -> u64 {
		self.entries.values().map(|e| e.count as u64).sum()
	}

	/// Averaged training samples.
	///
	/// Weights are normalised so the mean is one and clamped, because a position
	/// seen four hundred times is genuinely four hundred times better evidence but
	/// a gradient four hundred times larger would simply blow the network up.
	pub fn into_samples(self) -> Vec<Sample> {
		if self.entries.is_empty() {
			return Vec::new();
		}
		let mean_count =
			self.entries.values().map(|e| e.count as f32).sum::<f32>() / self.entries.len() as f32;

		self.entries
			.into_values()
			.map(|entry| Sample {
				target: entry.sum / entry.count as f32,
				weight: (entry.count as f32 / mean_count).clamp(0.2, 5.0),
				encoding: entry.encoding,
			})
			.collect()
	}
}

// ---------------------------------------------------------------------------
// Finding the positions the critic is actually asked about
// ---------------------------------------------------------------------------

struct Reservoir {
	states: Vec<BattleState>,
	seen: u64,
	capacity: usize,
	rng: StdRng,
}

impl Reservoir {
	/// Classic reservoir sampling, so the kept states are a uniform sample of
	/// every position the leaf was asked about rather than the first few.
	fn offer(&mut self, state: &BattleState) {
		self.seen += 1;
		if self.states.len() < self.capacity {
			self.states.push(state.clone());
			return;
		}
		let index = self.rng.random_range(0..self.seen);
		if (index as usize) < self.capacity {
			self.states[index as usize] = state.clone();
		}
	}
}

/// Wraps a leaf estimate and remembers a sample of what it was asked about.
///
/// The solver owns its leaf, so the recorder shares the reservoir with the caller
/// through an `Rc` rather than handing it back.
pub struct HorizonRecorder {
	inner: Box<dyn LeafEvaluator>,
	reservoir: Rc<RefCell<Reservoir>>,
}

/// A handle on what a [`HorizonRecorder`] kept, readable after the solve that
/// filled it.
pub struct HorizonSample {
	reservoir: Rc<RefCell<Reservoir>>,
}

impl HorizonSample {
	/// The sampled positions — a uniform draw from everything the leaf was asked
	/// about, not the first few.
	pub fn states(&self) -> Vec<BattleState> {
		self.reservoir.borrow().states.clone()
	}

	/// How many positions the leaf was consulted on in total.
	pub fn seen(&self) -> u64 {
		self.reservoir.borrow().seen
	}
}

impl HorizonRecorder {
	pub fn new(
		inner: Box<dyn LeafEvaluator>,
		capacity: usize,
		seed: u64,
	) -> (Self, HorizonSample) {
		let reservoir = Rc::new(RefCell::new(Reservoir {
			states: Vec::new(),
			seen: 0,
			capacity,
			rng: StdRng::seed_from_u64(seed),
		}));
		(
			HorizonRecorder { inner, reservoir: Rc::clone(&reservoir) },
			HorizonSample { reservoir },
		)
	}
}

impl LeafEvaluator for HorizonRecorder {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		self.reservoir.borrow_mut().offer(state);
		self.inner.value(state, team, registry)
	}
}

/// Whether somebody on the field owes a replacement.
///
/// The leaf evaluator is handed a state with no say in what is pending, so this
/// recovers it — the encoder carries a replacement flag, and it has to agree with
/// the one the training labels were collected under.
pub fn awaiting_replacement(state: &BattleState) -> bool {
	matches!(pending_request(state), StepRequest::NeedsReplacements(_))
}

/// What the engine would be asking for at this position.
///
/// The leaf evaluator only receives a state, but playing on from one needs the
/// pending request too. This reproduces the decision `engine::step` makes at the
/// end of a turn: the battle is over, or somebody on the field has fainted and
/// owes a replacement, or it is an ordinary turn.
pub fn pending_request(state: &BattleState) -> StepRequest {
	if let Some(outcome) = state.outcome() {
		return StepRequest::Finished(outcome);
	}
	let fainted: Vec<PositionId> = state
		.field
		.all_field_positions()
		.into_iter()
		.filter(|pos| state.get_mon(*pos).map_or(false, |mon| mon.current_hp == 0))
		.collect();

	if fainted.is_empty() {
		StepRequest::NeedsActions
	} else {
		StepRequest::NeedsReplacements(fainted)
	}
}

// ---------------------------------------------------------------------------
// Collecting labels
// ---------------------------------------------------------------------------

/// Play one game from `start` under the solved strategies, labelling every
/// position along the way with the result.
///
/// Both perspectives of each position are recorded with opposite targets. That
/// doubles the data and teaches the zero-sum property directly rather than
/// relying on the halving in [`Critic::correction`] to paper over its absence.
///
/// A game that does not finish has no real label, so it is dropped rather than
/// guessed at.
fn play_and_label(
	solver: &Solver,
	start: &BattleState,
	start_request: StepRequest,
	registry: &Registry,
	rng: &mut dyn RngCore,
	out: &mut SampleSet,
) {
	let mut state = start.clone();
	let mut request = start_request;
	let mut seen: Vec<(StateKey, (Vec<f32>, Vec<f32>), f32)> = Vec::new();
	let mut result = None;

	for _ in 0..PLAYOUT_CAP {
		let actors = match DecisionNode::from(&state, &request) {
			DecisionNode::Terminal(outcome) => {
				result = Some(match outcome {
					Outcome::Win { team } => if team == Team::Zero { 1.0 } else { -1.0 },
					Outcome::Draw => 0.0,
				});
				break;
			}
			DecisionNode::Decision { actors } => actors,
		};

		if let Some(key) = StateKey::new(&state, &request) {
			let replacement = matches!(request, StepRequest::NeedsReplacements(_));
			seen.push((
				key,
				encode_both(&state, registry, replacement),
				// The heuristic's own opinion, because the network is trained on
				// what it misses rather than on the result itself.
				HealthHeuristic.value(&state, Team::Zero, registry),
			));
		}

		let commands = actors
			.iter()
			.map(|actor| {
				let strategy = solver
					.average_strategy(&state, &request, actor.team)
					.unwrap_or_else(|| InfosetData::new().average_strategy(&actor.mask));
				actor.command(sample_action(&strategy, rng), &state, registry)
			})
			.collect();

		let StepResult { battle_state, step_request } =
			engine::step(state, commands, registry, rng);
		state = battle_state;
		request = step_request;
	}

	let Some(result) = result else {
		return;
	};

	for (key, (zero_view, one_view), heuristic) in seen {
		// What the heuristic missed, from each side. The two are negatives of each
		// other, so both are recorded.
		let residual = result - heuristic;
		out.add(key.clone(), Team::Zero, zero_view, residual);
		out.add(key, Team::One, one_view, -residual);
	}
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

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
/// labels contain no estimate at all. The 2v2 is the position the critic exists
/// to improve, and the only one whose labels depend on the estimate being
/// trained — which is why it is not the only thing trained on. Without the
/// anchors, a wrong estimate makes a wrong solve, which makes wrong labels, which
/// keeps the estimate wrong: stable, self-confirming and untrue.
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
	/// Solve-then-train cycles. Each re-solves with the critic trained so far, so
	/// later rounds learn from better play.
	pub rounds: usize,
	pub solver_iterations: usize,
	/// Games played from each training position per round.
	pub root_playouts: usize,
	/// Horizon positions sampled per training position per round.
	pub horizon_positions: usize,
	/// Games played from each sampled horizon position.
	pub horizon_playouts: usize,
	/// Iterations for the short solve that gives a horizon position its strategy.
	pub horizon_solve_iterations: usize,
	/// Lookahead for that short solve.
	pub horizon_lookahead: usize,
	/// Passes over the collected samples each round.
	pub epochs: usize,
	pub learning_rate: f32,
}

impl Default for TrainingConfig {
	fn default() -> Self {
		TrainingConfig {
			rounds: 3,
			solver_iterations: 800,
			root_playouts: 400,
			horizon_positions: 40,
			horizon_playouts: 30,
			horizon_solve_iterations: 150,
			horizon_lookahead: 4,
			epochs: 6,
			learning_rate: 0.001,
		}
	}
}

/// What one round of training did.
pub struct RoundReport {
	pub round: usize,
	/// Distinct positions trained on.
	pub positions: usize,
	/// Raw game results averaged into them.
	pub labels: u64,
	/// Mean labels per position — the factor by which noise was cut.
	pub labels_per_position: f32,
	pub mean_squared_error: f32,
}

/// Train a critic by solving positions and learning from how they turn out.
pub fn train(
	registry: &Registry,
	curriculum: &[TrainingPosition],
	config: &TrainingConfig,
	rng: &mut dyn RngCore,
) -> (Critic, Vec<RoundReport>) {
	let mut critic = Critic::new_random(rng);
	let mut reports = Vec::new();

	for round in 0..config.rounds {
		let mut set = SampleSet::new();

		for position in curriculum {
			// Round zero has nothing trained yet, so it plays off the hand-written
			// heuristic. Later rounds solve with what has been learned.
			let base: Box<dyn LeafEvaluator> = if round == 0 {
				Box::new(HealthHeuristic)
			} else {
				Box::new(critic.clone())
			};

			let (recorder, reservoir) =
				HorizonRecorder::new(base, config.horizon_positions, rng.next_u64());

			let mut solver = Solver::with_leaf(
				registry,
				SolverConfig::for_lookahead(config.solver_iterations, position.lookahead),
				Box::new(recorder),
			);
			solver.solve(&position.state, rng);

			// On-policy labels from the root. On a position that ends by itself
			// these reach real terminals, so they carry no estimate at all.
			for _ in 0..config.root_playouts {
				play_and_label(
					&solver,
					&position.state,
					StepRequest::NeedsActions,
					registry,
					rng,
					&mut set,
				);
			}

			// Labels from where the critic is actually consulted.
			let horizon = reservoir.states();
			for state in horizon {
				let request = pending_request(&state);
				if matches!(request, StepRequest::Finished(_)) {
					continue;
				}
				// The main solve has no strategy past its own horizon, so give this
				// position a short solve of its own before playing it out. Playing
				// uniformly would label it with the value of random play.
				let leaf: Box<dyn LeafEvaluator> = if round == 0 {
					Box::new(HealthHeuristic)
				} else {
					Box::new(critic.clone())
				};
				let mut sub = Solver::with_leaf(
					registry,
					SolverConfig::for_lookahead(
						config.horizon_solve_iterations,
						config.horizon_lookahead,
					),
					leaf,
				);
				sub.solve(&state, rng);

				for _ in 0..config.horizon_playouts {
					play_and_label(&sub, &state, request.clone(), registry, rng, &mut set);
				}
			}
		}

		let positions = set.len();
		let labels = set.labels();
		let mut samples = set.into_samples();

		let mut error = 0.0;
		for _ in 0..config.epochs {
			error = critic.fit(&mut samples, config.learning_rate, rng);
		}

		reports.push(RoundReport {
			round,
			positions,
			labels,
			labels_per_position: if positions == 0 {
				0.0
			} else {
				labels as f32 / positions as f32
			},
			mean_squared_error: error,
		});
	}

	(critic, reports)
}

// ---------------------------------------------------------------------------
// Choosing how far to trust it
// ---------------------------------------------------------------------------

pub struct TrustCandidate {
	pub trust: f32,
	pub exploitability: f32,
}

pub struct SelectionReport {
	pub candidates: Vec<TrustCandidate>,
	pub chosen: f32,
	/// Exploitability with the correction switched off entirely — the plain
	/// heuristic, and the bar the critic has to clear.
	pub baseline: f32,
}

impl SelectionReport {
	/// How much the chosen critic improved on the heuristic. Never negative,
	/// because trust zero is always a candidate.
	pub fn improvement(&self) -> f32 {
		let chosen = self
			.candidates
			.iter()
			.find(|c| c.trust == self.chosen)
			.map(|c| c.exploitability)
			.unwrap_or(self.baseline);
		self.baseline - chosen
	}
}

pub struct SelectionConfig {
	pub trusts: Vec<f32>,
	pub solver_iterations: usize,
	/// Lookahead used for every candidate, and therefore for the best-response
	/// measurement too, so the two describe the same game.
	pub lookahead: usize,
	pub chance_samples: usize,
}

impl Default for SelectionConfig {
	fn default() -> Self {
		SelectionConfig {
			trusts: vec![0.0, 0.1, 0.25, 0.5, 1.0],
			solver_iterations: 1_500,
			lookahead: 5,
			chance_samples: 1,
		}
	}
}

/// Pick how far to trust the correction, by measuring what it does to
/// exploitability.
///
/// This is what keeps a trained critic from ever being a regression. Trust zero
/// makes [`Critic::value`] return the heuristic exactly, so it sits in the
/// candidate list as a floor: the worst outcome is that the search finds nothing
/// better and the critic is switched off.
///
/// Selection is on exploitability rather than training loss on purpose. Loss here
/// is mostly irreducible label noise and barely moves; exploitability is the
/// thing actually wanted, and it moved by a factor of two across the trust values
/// when this was first run.
pub fn select_trust(
	registry: &Registry,
	critic: &Critic,
	positions: &[BattleState],
	config: &SelectionConfig,
	rng: &mut dyn RngCore,
) -> SelectionReport {
	let mut candidates: Vec<TrustCandidate> = Vec::new();

	// Common random numbers across candidates. Each trust level is judged on a
	// different solve, and every solve samples the engine's chance rolls; left to
	// their own seeds the candidates differ by that noise as much as by the thing
	// being compared, and the search would happily pick whichever got lucky. One
	// seed per position, reused for every candidate, pairs them up.
	let seed = rng.next_u64();

	for trust in config.trusts.iter().copied() {
		let mut total = 0.0;
		for (index, position) in positions.iter().enumerate() {
			let mut candidate = critic.clone();
			candidate.trust = trust;

			let mut paired = StdRng::seed_from_u64(seed.wrapping_add(index as u64));

			let mut solver = Solver::with_leaf(
				registry,
				SolverConfig::for_lookahead(config.solver_iterations, config.lookahead),
				Box::new(candidate),
			);
			solver.solve(position, &mut paired);

			let exploit_config = ExploitConfig {
				chance_samples: config.chance_samples,
				..ExploitConfig::matching(&solver)
			};
			total += measure(&solver, position, exploit_config, &mut paired).exploitability;
		}
		candidates.push(TrustCandidate {
			trust,
			exploitability: if positions.is_empty() {
				0.0
			} else {
				total / positions.len() as f32
			},
		});
	}

	let baseline = candidates
		.iter()
		.find(|c| c.trust == 0.0)
		.map(|c| c.exploitability)
		.unwrap_or(f32::INFINITY);

	let chosen = candidates
		.iter()
		.min_by(|a, b| a.exploitability.total_cmp(&b.exploitability))
		.map(|c| c.trust)
		.unwrap_or(0.0);

	SelectionReport { candidates, chosen, baseline }
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
	use crate::battle::state::field::PositionId;
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	fn registry() -> Registry {
		Registry::load()
	}

	// --- the estimate ------------------------------------------------------

	/// A zero-sum game cannot have both sides ahead.
	#[test]
	fn an_estimate_is_worth_the_opposite_to_each_side() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(1);
		let critic = Critic::new_random(&mut rng);
		let state = switch_prediction_2v2(&registry);

		let zero = critic.value(&state, Team::Zero, &registry);
		let one = critic.value(&state, Team::One, &registry);

		assert!((zero + one).abs() < 1e-5, "not zero-sum: {zero} vs {one}");
	}

	#[test]
	fn estimates_stay_on_the_result_scale() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(2);
		let critic = Critic::new_random(&mut rng);

		for state in [known_answer_duel(&registry), switch_prediction_2v2(&registry)] {
			for team in [Team::Zero, Team::One] {
				let value = critic.value(&state, team, &registry);
				assert!((-1.0..=1.0).contains(&value), "out of range: {value}");
			}
		}
	}

	/// Trust zero has to mean *exactly* the heuristic, since that is the floor
	/// `select_trust` relies on. "Close to" would not do.
	#[test]
	fn trust_zero_is_the_heuristic_exactly() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(3);
		let mut critic = Critic::new_random(&mut rng);
		critic.trust = 0.0;

		for state in [known_answer_duel(&registry), switch_prediction_2v2(&registry)] {
			for team in [Team::Zero, Team::One] {
				let learned = critic.value(&state, team, &registry);
				let plain = HealthHeuristic.value(&state, team, &registry);
				assert_eq!(learned, plain, "trust 0 must be the baseline, bit for bit");
			}
		}
	}

	// --- averaging by position ---------------------------------------------

	/// The heart of the noise fix: many labels for one position collapse to their
	/// mean, and the count survives as a weight.
	#[test]
	fn repeated_labels_for_one_position_are_averaged() {
		let registry = registry();
		let state = known_answer_duel(&registry);
		let key = StateKey::new(&state, &StepRequest::NeedsActions).unwrap();

		let mut set = SampleSet::new();
		// Four games from the same position: three wins and a loss, so the true
		// value is +0.5 and no single label says so.
		for target in [1.0, 1.0, 1.0, -1.0] {
			set.add(key.clone(), Team::Zero, vec![0.0; TOTAL_ENCODING_LEN], target);
		}

		assert_eq!(set.len(), 1, "one position, however many games");
		assert_eq!(set.labels(), 4);

		let samples = set.into_samples();
		assert_eq!(samples.len(), 1);
		assert!(
			(samples[0].target - 0.5).abs() < 1e-6,
			"expected the mean, got {}",
			samples[0].target,
		);
	}

	/// Distinct positions stay distinct, and the two perspectives of one position
	/// are separate samples.
	#[test]
	fn different_positions_and_perspectives_are_kept_apart() {
		let registry = registry();
		let state = known_answer_duel(&registry);
		let mut hurt = state.clone();
		hurt.get_mut_mon(PositionId(0)).unwrap().current_hp -= 1;

		let key = StateKey::new(&state, &StepRequest::NeedsActions).unwrap();
		let other = StateKey::new(&hurt, &StepRequest::NeedsActions).unwrap();

		let mut set = SampleSet::new();
		set.add(key.clone(), Team::Zero, vec![0.0; 4], 1.0);
		set.add(key, Team::One, vec![0.0; 4], -1.0);
		set.add(other, Team::Zero, vec![0.0; 4], 1.0);

		assert_eq!(set.len(), 3);
	}

	/// A position seen far more often is better evidence and should weigh more —
	/// but not by its raw count, which would blow the network up.
	#[test]
	fn weights_follow_the_evidence_and_stay_bounded() {
		let registry = registry();
		let state = known_answer_duel(&registry);
		let mut hurt = state.clone();
		hurt.get_mut_mon(PositionId(0)).unwrap().current_hp -= 1;

		let common = StateKey::new(&state, &StepRequest::NeedsActions).unwrap();
		let rare = StateKey::new(&hurt, &StepRequest::NeedsActions).unwrap();

		let mut set = SampleSet::new();
		for _ in 0..500 {
			set.add(common.clone(), Team::Zero, vec![0.0; 4], 1.0);
		}
		set.add(rare, Team::Zero, vec![0.0; 4], 1.0);

		let samples = set.into_samples();
		let heaviest = samples.iter().map(|s| s.weight).fold(0.0f32, f32::max);
		let lightest = samples.iter().map(|s| s.weight).fold(f32::MAX, f32::min);

		assert!(heaviest > lightest, "more evidence should weigh more");
		assert!(heaviest <= 5.0 + 1e-6, "weights must stay clamped, got {heaviest}");
		assert!(lightest >= 0.2 - 1e-6, "weights must stay clamped, got {lightest}");
	}

	// --- horizon sampling ---------------------------------------------------

	/// The recorder has to see the positions the solver truncates at, and pass the
	/// estimate through unchanged.
	#[test]
	fn the_recorder_captures_horizon_positions_without_changing_the_estimate() {
		let registry = registry();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(7);

		let (recorder, sample) = HorizonRecorder::new(Box::new(HealthHeuristic), 25, 99);
		let mut solver = Solver::with_leaf(
			&registry,
			SolverConfig::for_lookahead(100, 4),
			Box::new(recorder),
		);
		solver.solve(&root, &mut rng);

		assert!(sample.seen() > 0, "a depth-limited solve must consult the leaf");
		assert_eq!(
			sample.seen(),
			solver.truncated_positions(),
			"every truncation should have been offered to the recorder",
		);
		let states = sample.states();
		assert!(!states.is_empty());
		assert!(states.len() <= 25, "the reservoir must respect its capacity");
	}

	/// Playing on from a horizon position needs the pending request, which the
	/// leaf interface does not carry — so it is inferred, and has to match what
	/// the engine would have said.
	#[test]
	fn the_pending_request_is_inferred_correctly() {
		let registry = registry();
		let state = switch_prediction_2v2(&registry);
		assert_eq!(pending_request(&state), StepRequest::NeedsActions);

		// Faint one active while its team-mate stands: that owes a replacement.
		let mut fainted = state.clone();
		fainted.get_mut_mon(PositionId(0)).unwrap().current_hp = 0;
		assert_eq!(
			pending_request(&fainted),
			StepRequest::NeedsReplacements(vec![PositionId(0)]),
		);

		// Wipe a whole side and the battle is over instead.
		let mut lost = state.clone();
		for slot in [0usize, 2] {
			if let Some(mon) = lost.roster.get_mut_mon(crate::battle::state::roster::RosterId(slot)) {
				mon.current_hp = 0;
			}
		}
		assert_eq!(
			pending_request(&lost),
			StepRequest::Finished(Outcome::Win { team: Team::One }),
		);
	}

	// --- training and selection ---------------------------------------------

	/// The loop runs end to end, and the averaging actually bites: many more raw
	/// labels than distinct positions.
	#[test]
	fn training_averages_many_labels_into_each_position() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(4);

		let curriculum = vec![TrainingPosition {
			name: String::from("known-answer duel"),
			state: known_answer_duel(&registry),
			lookahead: 200,
		}];
		let config = TrainingConfig {
			rounds: 1,
			solver_iterations: 50,
			root_playouts: 60,
			horizon_positions: 0,
			horizon_playouts: 0,
			horizon_solve_iterations: 10,
			horizon_lookahead: 3,
			epochs: 1,
			learning_rate: 0.001,
		};

		let (_, reports) = train(&registry, &curriculum, &config, &mut rng);

		assert_eq!(reports.len(), 1);
		let report = &reports[0];
		assert!(report.positions > 0, "nothing was collected");
		assert!(
			report.labels_per_position > 2.0,
			"averaging should be doing real work, got {:.1} labels per position",
			report.labels_per_position,
		);
		assert!(report.labels > report.positions as u64);
	}

	/// The safety property: selection can never choose something worse than the
	/// heuristic, because the heuristic is always in the running.
	#[test]
	fn selection_never_lands_below_the_baseline() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(5);
		let critic = Critic::new_random(&mut rng);
		let positions = vec![switch_prediction_2v2(&registry)];

		let config = SelectionConfig {
			trusts: vec![0.0, 0.5, 1.0],
			solver_iterations: 150,
			lookahead: 4,
			chance_samples: 1,
		};
		let report = select_trust(&registry, &critic, &positions, &config, &mut rng);

		assert_eq!(report.candidates.len(), 3);
		let chosen = report
			.candidates
			.iter()
			.find(|c| c.trust == report.chosen)
			.unwrap();
		assert!(
			chosen.exploitability <= report.baseline + 1e-6,
			"chose {} at {:.4}, worse than the baseline {:.4}",
			report.chosen, chosen.exploitability, report.baseline,
		);
		assert!(report.improvement() >= -1e-6, "improvement cannot be negative");
	}

	#[test]
	fn a_critic_survives_a_round_trip_through_a_file() {
		let registry = registry();
		let mut rng = StdRng::seed_from_u64(6);
		let mut critic = Critic::new_random(&mut rng);
		// Trust defaults to zero, which would make this pass without the network
		// being involved at all.
		critic.trust = 1.0;
		let state = switch_prediction_2v2(&registry);
		let before = critic.value(&state, Team::Zero, &registry);

		let path = std::env::temp_dir().join("cfr_critic_roundtrip.json");
		let path = path.to_str().unwrap();
		critic.to_file(path).unwrap();
		let mut loaded = Critic::from_file(path).unwrap();
		loaded.trust = 1.0;

		assert!((before - loaded.value(&state, Team::Zero, &registry)).abs() < 1e-6);
		let _ = std::fs::remove_file(path);
	}
}
