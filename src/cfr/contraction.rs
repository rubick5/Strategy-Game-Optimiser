//! How much of a leaf estimate's error does the search actually remove?
//!
//! Four attempts to beat [`HealthHeuristic`] have failed — a learned critic, a
//! ReBeL loop, multi-valued leaves, and a pairwise loss that was refuted before
//! it was built. Two explanations fit that evidence equally well, and they point
//! in opposite directions:
//!
//! 1. **Every estimate we can build is bad.** Then leaf accuracy is the binding
//!    constraint, and the only route forward is the expensive one — solve a large
//!    number of positions and train on them, the way DeepStack and ReBeL did.
//! 2. **Leaf accuracy barely reaches the root.** Then the four failures are all
//!    one failure: swapping an estimate that gets washed out for another estimate
//!    that gets washed out was never going to move anything, and the effort
//!    belongs in depth rather than in networks.
//!
//! Nothing measured so far distinguishes them. The number that does is how much
//! a depth-`k` solve **contracts** leaf error, and it comes in two forms, which
//! this module reports separately because they answer different questions.
//!
//! **Absolute contraction**, `γ = δ/ε`, compares the error of a depth-`k` solve
//! against the error of the raw estimate at the same position. It says whether
//! searching is better than guessing. It is confounded, though: part of the
//! heuristic's error is systematic, and a systematic error can survive search
//! that random error would not.
//!
//! **Sensitivity**, `γ_s`, perturbs the leaf by a known amount and measures how
//! far the root value moves. This is the contraction factor in the sense that
//! matters for bootstrapping — ReBeL's value iteration converges at rate `γ_s`
//! per round, and needs `γ_s < 1` to converge at all — and it is also the
//! direct answer to "would a better leaf help": if perturbing the leaf by 0.5
//! moves the root by 0.05, nine tenths of any leaf improvement is thrown away
//! before it reaches the answer.
//!
//! A sensitivity measurement is worthless without a **noise floor**. MCCFR is a
//! sampling algorithm, so two solves of the same position with different seeds
//! already disagree. If the perturbed solve moves no further than that, the
//! perturbation did nothing detectable and the ratio is measuring variance. So
//! every perturbed solve is paired with a same-seed baseline (common random
//! numbers, so the comparison is not swamped by sampling noise) *and* with a
//! different-seed baseline that establishes the floor.
//!
//! **The reference has to be exact**, or this measures the reference. A 1v1 ends
//! by itself, so solving one reaches real terminals with no estimate anywhere in
//! it. A deeper 2v2 will not serve: it uses the heuristic at its own leaves, so
//! scoring the heuristic against it is circular — an earlier version of the
//! sibling measurement made exactly that mistake. Positions whose reference solve
//! estimated anything at the horizon are discarded rather than trusted.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::battle::state::Team;
use crate::cfr::leaf::{HealthHeuristic, LeafEvaluator};
use crate::cfr::node::DecisionNode;
use crate::cfr::solver::{Solver, SolverConfig};
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::SpeciesId;

/// The species in the roster.
const ROSTER: std::ops::Range<u32> = 2..8;

/// Moves that stop a duel from ending, stripped from the movesets here because a
/// position that does not terminate has no exact value to be measured against.
///
/// *guard* (Protect) and *decoy* (Substitute) are the two that can be played for
/// free — see the note at the top of [`crate::cfr::position`], which reasons
/// about exactly this and stops one move short.
///
/// *sap seed* (Leech Seed) is the third, and it is there for a different reason
/// that the module note misses: it **restores HP to the seeder**. Every other
/// move in the registry only ever moves health downward, so a duel makes progress
/// whatever is played. Two seeded creatures draining each other do not — the
/// transfer cancels and the position cycles forever.
///
/// This was found the hard way, by the mireling-versus-mireling reference solve
/// never returning. It is the same fact behind the two anomalous diagonals in the
/// six-a-side lead matrix: mireling and thornbeast are the two creatures carrying
/// sap seed, and their self-matchups were the two that failed to price at zero.
const STALLING_MOVES: [u32; 3] = [22, 24, 25];

/// A leaf estimate with a known amount of error added.
///
/// The offset is a deterministic function of the position, not a fresh random
/// draw, because a value function's error *is* a function of the position: an
/// estimator that is wrong about a position is wrong about it the same way every
/// time it is asked. Re-rolling per visit would model something else entirely,
/// and the search would average it away.
///
/// Antisymmetric in `team`, so the perturbed estimate stays zero-sum. Without
/// that the solver is no longer solving a zero-sum game and the value at the root
/// means nothing. The clamp preserves this: it is an odd function about zero.
pub struct Perturbed {
	inner: Box<dyn LeafEvaluator>,
	scale: f32,
	salt: u64,
}

impl Perturbed {
	pub fn new(inner: Box<dyn LeafEvaluator>, scale: f32, salt: u64) -> Self {
		Perturbed { inner, scale, salt }
	}

	/// A position-determined offset in `-scale..scale`.
	pub fn offset(&self, state: &BattleState) -> f32 {
		let mut hasher = DefaultHasher::new();
		self.salt.hash(&mut hasher);
		state.hash(&mut hasher);
		let unit = (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64;
		(unit as f32).mul_add(2.0, -1.0) * self.scale
	}
}

impl LeafEvaluator for Perturbed {
	fn value(&self, state: &BattleState, team: Team, registry: &Registry) -> f32 {
		let sign = if team == Team::Zero { 1.0 } else { -1.0 };
		(self.inner.value(state, team, registry) + self.offset(state) * sign).clamp(-1.0, 1.0)
	}
}

pub struct ContractionConfig {
	/// Iterations for the exact reference solve. This one is searched to the end.
	pub truth_iterations: usize,
	/// Node budget for the reference solve. Stripping the stalling moves is not a
	/// proof that every remaining position ends, so this is the backstop: a
	/// reference that runs past it is discarded rather than trusted, and the cost
	/// of finding that out is bounded.
	pub truth_max_nodes: u64,
	/// Iterations for each depth-limited solve.
	pub solve_iterations: usize,
	/// Horizons to report. Each costs two extra solves per position.
	pub depths: Vec<usize>,
	/// Half-width of the injected leaf error, on the -1..1 result scale.
	pub noise: f32,
	/// How many turns to walk into each duel, collecting a position per turn.
	pub walk: usize,
	pub seed: u64,
	/// Print a line per position. This runs for minutes, and a run that prints
	/// nothing until it finishes cannot be told from one that has hung.
	pub progress: bool,
}

impl Default for ContractionConfig {
	fn default() -> Self {
		ContractionConfig {
			truth_iterations: 4_000,
			truth_max_nodes: 4_000_000,
			solve_iterations: 1_500,
			depths: vec![1, 2, 4],
			noise: 0.5,
			walk: 2,
			seed: 77_001,
			progress: false,
		}
	}
}

/// One horizon's worth of the answer.
pub struct DepthRow {
	pub depth: usize,
	/// `δ` — RMS error of the depth-limited solve against the exact value.
	pub solved_rms: f32,
	/// `γ = δ/ε`. Below 1 means searching beats guessing.
	pub gamma_absolute: f32,
	/// How far the root moved when the leaf was perturbed (same seed).
	pub movement_rms: f32,
	/// How far the root moved on a reseed alone, with the leaf untouched.
	pub floor_rms: f32,
	/// `γ_s` — movement per unit of injected leaf error.
	pub gamma_sensitivity: f32,
	/// Fraction of the variance in the true values the solve accounts for.
	pub r2: f32,
	/// Mean positions estimated at the horizon per solve.
	///
	/// The number that decides whether any of the rest means anything. A 1v1
	/// often finishes inside the horizon, and a solve that never reaches its
	/// horizon never consults the leaf — so it would score a perfect `gamma` and
	/// a `gamma_sensitivity` of zero while demonstrating nothing at all.
	pub leaves_used: f32,
}

pub struct ContractionReport {
	pub positions: usize,
	pub discarded: usize,
	pub sd_truth: f32,
	/// `ε` — RMS error of the raw heuristic against the exact value.
	pub leaf_rms: f32,
	/// Fraction of the variance in the true values the raw heuristic accounts for.
	pub leaf_r2: f32,
	/// The RMS of the error actually injected, which is what `γ_s` divides by.
	pub noise_rms: f32,
	pub rows: Vec<DepthRow>,
}

fn rms(values: &[f32]) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	(values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt()
}

fn sd(values: &[f32]) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	let mean = values.iter().sum::<f32>() / values.len() as f32;
	let centred: Vec<f32> = values.iter().map(|v| v - mean).collect();
	rms(&centred)
}

/// The movesets used here: the designed set minus anything that can stall.
fn moveset(species: u32) -> Vec<MoveId> {
	Registry::default_moveset(SpeciesId(species))
		.into_iter()
		.filter(|id| !STALLING_MOVES.contains(&id.0))
		.collect()
}

fn duel(registry: &Registry, zero: u32, one: u32) -> BattleState {
	let build = |species: u32| {
		vec![CreatureState::from_species(registry, SpeciesId(species), moveset(species))]
	};
	BattleState::from(build(zero), build(one), vec![0, 1])
}

/// Every ordered species pairing, walked a few turns under random play.
///
/// Turn zero is included as well as the positions after it: a full-health start
/// is where the heuristic has least to say (both sides read as exactly even) and
/// a damaged mid-game is where it has most, so both belong in the sample.
pub fn sample_positions(
	registry: &Registry,
	config: &ContractionConfig,
) -> Vec<(BattleState, StepRequest)> {
	let mut out = Vec::new();

	for zero in ROSTER {
		for one in ROSTER {
			let mut rng = StdRng::seed_from_u64(config.seed ^ (zero as u64) << 8 ^ one as u64);
			let mut state = duel(registry, zero, one);
			let mut request = StepRequest::NeedsActions;

			for _ in 0..=config.walk {
				let actors = match DecisionNode::from(&state, &request) {
					DecisionNode::Terminal(_) => break,
					DecisionNode::Decision { actors } => actors,
				};
				out.push((state.clone(), request.clone()));

				let commands: Vec<_> = actors
					.iter()
					.map(|actor| {
						let choice = actor.mask.get_random_valid(&mut rng).unwrap();
						actor.command(choice.to_number(), &state, registry)
					})
					.collect();
				let StepResult { battle_state, step_request } =
					engine::step(state, commands, registry, &mut rng);
				state = battle_state;
				request = step_request;
			}
		}
	}

	out
}

/// One solve, reporting the root value and whether anything was estimated.
fn solve_value(
	registry: &Registry,
	state: &BattleState,
	request: &StepRequest,
	depth: usize,
	iterations: usize,
	max_nodes: u64,
	leaf: Box<dyn LeafEvaluator>,
	seed: u64,
) -> Option<Solved> {
	let mut rng = StdRng::seed_from_u64(seed);
	let mut solver = Solver::with_leaf(
		registry,
		SolverConfig { iterations, max_depth: depth, max_nodes, ..SolverConfig::default() },
		leaf,
	);
	solver.log_values(0.5);
	solver.solve_from(state, request.clone(), &mut rng);
	solver.root_value(state, request, Team::Zero).map(|value| Solved {
		value,
		truncated: solver.truncated_positions(),
		out_of_budget: solver.hit_node_budget(),
	})
}

/// What one solve came back with, and the two reasons not to believe it.
struct Solved {
	value: f32,
	/// Positions estimated at the horizon instead of played out. Nonzero means
	/// the answer rests on the leaf estimate.
	truncated: u64,
	/// The solve stopped on the node budget, so it ran fewer iterations than
	/// asked and has not converged — which a value alone does not reveal.
	out_of_budget: bool,
}

/// Run the whole measurement. Slow — one exact solve plus `4 * depths` limited
/// solves per position.
pub fn measure(registry: &Registry, config: &ContractionConfig) -> ContractionReport {
	let positions = sample_positions(registry, config);
	if config.progress {
		println!(
			"  {} positions, {} depths, noise {:.2}",
			positions.len(),
			config.depths.len(),
			config.noise,
		);
	}
	let started = std::time::Instant::now();

	let mut truths: Vec<f32> = Vec::new();
	let mut leaf_errors: Vec<f32> = Vec::new();
	let mut noise: Vec<f32> = Vec::new();
	// Per depth: solve error, perturbation movement, reseed movement.
	let mut solved: Vec<Vec<f32>> = config.depths.iter().map(|_| Vec::new()).collect();
	let mut moved: Vec<Vec<f32>> = config.depths.iter().map(|_| Vec::new()).collect();
	let mut floor: Vec<Vec<f32>> = config.depths.iter().map(|_| Vec::new()).collect();
	let mut leaves: Vec<Vec<f32>> = config.depths.iter().map(|_| Vec::new()).collect();
	let mut discarded = 0;

	for (index, (state, request)) in positions.iter().enumerate() {
		let seed = config.seed.wrapping_add(index as u64 * 1_009);

		// The reference, searched to the end. Anything estimated at a horizon
		// means this is not exact, so the position is dropped rather than used.
		let truth = match solve_value(
			registry,
			state,
			request,
			SolverConfig::default().max_depth,
			config.truth_iterations,
			config.truth_max_nodes,
			Box::new(HealthHeuristic),
			seed,
		) {
			Some(Solved { value, truncated: 0, out_of_budget: false }) => value,
			_ => {
				discarded += 1;
				if config.progress {
					println!("  [{}/{}] discarded — no exact value", index + 1, positions.len());
				}
				continue;
			}
		};

		let perturbed = || -> Box<dyn LeafEvaluator> {
			Box::new(Perturbed::new(Box::new(HealthHeuristic), config.noise, seed))
		};

		truths.push(truth);
		leaf_errors.push(HealthHeuristic.value(state, Team::Zero, registry) - truth);
		noise.push(perturbed().value(state, Team::Zero, registry)
			- HealthHeuristic.value(state, Team::Zero, registry));

		for (slot, depth) in config.depths.iter().enumerate() {
			// Common random numbers: the baseline and the perturbed run share a
			// seed, so their difference is the leaf talking and not the sampler.
			let budget = config.truth_max_nodes;
			let base = solve_value(
				registry, state, request, *depth, config.solve_iterations, budget,
				Box::new(HealthHeuristic), seed,
			);
			let bumped = solve_value(
				registry, state, request, *depth, config.solve_iterations, budget,
				perturbed(), seed,
			);
			// A different seed with the same leaf: how far apart two runs land
			// for no reason at all.
			let reseed = solve_value(
				registry, state, request, *depth, config.solve_iterations, budget,
				Box::new(HealthHeuristic), seed ^ 0xDEAD_BEEF,
			);

			if let Some(base) = base {
				leaves[slot].push(base.truncated as f32);
				solved[slot].push(base.value - truth);
				if let Some(bumped) = bumped {
					moved[slot].push(bumped.value - base.value);
				}
				if let Some(reseed) = reseed {
					floor[slot].push(reseed.value - base.value);
				}
			}
		}

		if config.progress {
			println!(
				"  [{}/{}] truth {:+.4}  heuristic {:+.4}  {:.0?} elapsed",
				index + 1,
				positions.len(),
				truth,
				truth + leaf_errors[leaf_errors.len() - 1],
				started.elapsed(),
			);
		}
	}

	let sd_truth = sd(&truths);
	let leaf_rms = rms(&leaf_errors);
	let noise_rms = rms(&noise);
	let variance = sd_truth * sd_truth;
	let r2 = |error: f32| if variance > 0.0 { 1.0 - (error * error) / variance } else { f32::NAN };

	let rows = config
		.depths
		.iter()
		.enumerate()
		.map(|(slot, depth)| {
			let solved_rms = rms(&solved[slot]);
			let movement_rms = rms(&moved[slot]);
			DepthRow {
				depth: *depth,
				solved_rms,
				gamma_absolute: if leaf_rms > 0.0 { solved_rms / leaf_rms } else { f32::NAN },
				movement_rms,
				floor_rms: rms(&floor[slot]),
				gamma_sensitivity: if noise_rms > 0.0 { movement_rms / noise_rms } else { f32::NAN },
				r2: r2(solved_rms),
				leaves_used: leaves[slot].iter().sum::<f32>() / leaves[slot].len().max(1) as f32,
			}
		})
		.collect();

	ContractionReport {
		positions: truths.len(),
		discarded,
		sd_truth,
		leaf_rms,
		leaf_r2: r2(leaf_rms),
		noise_rms,
		rows,
	}
}

impl fmt::Display for ContractionReport {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(
			f,
			"{} positions ({} discarded as not searchable to the end)",
			self.positions, self.discarded,
		)?;
		writeln!(f, "spread of the true values (sd):  {:.4}", self.sd_truth)?;
		writeln!(
			f,
			"raw heuristic:  RMS error {:.4}   R^2 {:+.3}    <- this is epsilon",
			self.leaf_rms, self.leaf_r2,
		)?;
		writeln!(f, "injected leaf error, as applied at the horizon: RMS {:.4}", self.noise_rms)?;
		writeln!(f)?;
		writeln!(
			f,
			"{:>6}  {:>9}  {:>7}  {:>7}  {:>9}  {:>7}  {:>9}  {:>10}",
			"depth", "RMS err", "gamma", "R^2", "moved by", "floor", "gamma_s", "leaves/solve",
		)?;
		for row in &self.rows {
			writeln!(
				f,
				"{:>6}  {:>9.4}  {:>7.3}  {:>+7.3}  {:>9.4}  {:>7.4}  {:>9.3}  {:>10.1}",
				row.depth,
				row.solved_rms,
				row.gamma_absolute,
				row.r2,
				row.movement_rms,
				row.floor_rms,
				row.gamma_sensitivity,
				row.leaves_used,
			)?;
		}
		writeln!(f)?;
		writeln!(f, "gamma   = depth-k error / raw heuristic error   (<1: search beats guessing)")?;
		writeln!(f, "gamma_s = root movement / injected leaf error   (<<1: leaf work is wasted)")?;
		writeln!(f, "floor   = movement from reseeding alone; gamma_s is meaningless unless moved by >> floor")?;
		write!(f, "leaves/solve = horizon estimates per solve; a solve that reaches zero never used the leaf")
	}
}


/// How far does the root move when the leaf is perturbed, at a position whose
/// true value is unknown?
///
/// [`measure`] needs an exact reference, which confines it to duels that end by
/// themselves — and a position that ends by itself is exactly one where the
/// horizon may never be reached and the leaf never consulted. That is the wrong
/// place to ask whether the leaf matters.
///
/// Sensitivity does not need the reference. Perturbing the leaf and watching the
/// root is a comparison of two solves against each other, not against truth, so
/// it runs on the positions that actually matter: a 2v2, a full six-a-side, or
/// anything loaded from a file. This is the measurement that speaks to whether
/// better leaf estimates would repay the effort.
///
/// Samples come from re-salting the perturbation rather than from many
/// positions, so one root gives a distribution rather than a single number.
pub struct SensitivityConfig {
	pub iterations: usize,
	pub depths: Vec<usize>,
	pub noise: f32,
	/// Perturbations to draw, and baseline reseeds to match them against.
	pub samples: usize,
	pub seed: u64,
	pub progress: bool,
}

impl Default for SensitivityConfig {
	fn default() -> Self {
		SensitivityConfig {
			iterations: 1_500,
			depths: vec![1, 2, 3, 4],
			noise: 0.5,
			samples: 6,
			seed: 90_210,
			progress: false,
		}
	}
}

pub struct SensitivityRow {
	pub depth: usize,
	pub movement_rms: f32,
	pub floor_rms: f32,
	pub gamma_sensitivity: f32,
	pub leaves_used: f32,
	pub base_value: f32,
}

pub struct SensitivityReport {
	pub noise_rms: f32,
	pub rows: Vec<SensitivityRow>,
}

/// The size of the perturbation *where it is actually applied*.
///
/// Sampling it at the root is wrong twice over. The root is not a leaf, and one
/// state gives one number per salt — with a handful of salts that estimates the
/// spread badly, which is how a first run reported 0.197 for a perturbation whose
/// construction gives 0.289. The ratio it divides is only as good as this.
///
/// It also cannot be taken from the construction, because the estimate is
/// clamped to the result scale: at a leaf where one side is nearly dead the base
/// value is already near -1 or +1 and part of the offset is clipped away. The
/// applied error is therefore smaller than the drawn one, by an amount that
/// depends on the position. So it is measured, over positions reached by walking
/// the root — a rough stand-in for the distribution the horizon actually sees.
fn applied_perturbation(
	registry: &Registry,
	root: &BattleState,
	config: &SensitivityConfig,
) -> f32 {
	let horizon = config.depths.iter().copied().max().unwrap_or(4);
	let mut offsets = Vec::new();

	for sample in 0..config.samples {
		let salt = config.seed.wrapping_add(sample as u64 * 7_919);
		let leaf = Perturbed::new(Box::new(HealthHeuristic), config.noise, salt);

		for walk in 0..24u64 {
			let mut rng = StdRng::seed_from_u64(salt ^ walk.wrapping_mul(2_654_435_761));
			let mut state = root.clone();
			let mut request = StepRequest::NeedsActions;

			for _ in 0..horizon {
				let actors = match DecisionNode::from(&state, &request) {
					DecisionNode::Terminal(_) => break,
					DecisionNode::Decision { actors } => actors,
				};
				let commands: Vec<_> = actors
					.iter()
					.map(|actor| {
						let choice = actor.mask.get_random_valid(&mut rng).unwrap();
						actor.command(choice.to_number(), &state, registry)
					})
					.collect();
				let StepResult { battle_state, step_request } =
					engine::step(state, commands, registry, &mut rng);
				state = battle_state;
				request = step_request;
			}

			offsets.push(
				leaf.value(&state, Team::Zero, registry)
					- HealthHeuristic.value(&state, Team::Zero, registry),
			);
		}
	}

	rms(&offsets)
}

pub fn sensitivity(
	registry: &Registry,
	root: &BattleState,
	request: &StepRequest,
	config: &SensitivityConfig,
) -> SensitivityReport {
	let noise_rms = applied_perturbation(registry, root, config);
	let mut rows = Vec::new();
	let started = std::time::Instant::now();
	if config.progress {
		println!("  perturbation as applied at the horizon: RMS {noise_rms:.4}");
	}

	for depth in &config.depths {
		let Some(base) = solve_value(
			registry, root, request, *depth, config.iterations,
			SolverConfig::default().max_nodes, Box::new(HealthHeuristic), config.seed,
		) else {
			continue;
		};

		let mut moved = Vec::new();
		let mut floor = Vec::new();
		for sample in 0..config.samples {
			let salt = config.seed.wrapping_add(sample as u64 * 7_919);
			let leaf = Perturbed::new(Box::new(HealthHeuristic), config.noise, salt);

			// Same solver seed as the baseline, so the difference is the leaf.
			if let Some(bumped) = solve_value(
				registry, root, request, *depth, config.iterations,
				SolverConfig::default().max_nodes, Box::new(leaf), config.seed,
			) {
				moved.push(bumped.value - base.value);
			}
			// Same leaf, different solver seed: the floor this has to clear.
			if let Some(reseed) = solve_value(
				registry, root, request, *depth, config.iterations,
				SolverConfig::default().max_nodes, Box::new(HealthHeuristic),
				salt ^ 0xDEAD_BEEF,
			) {
				floor.push(reseed.value - base.value);
			}
		}

		let movement_rms = rms(&moved);
		if config.progress {
			println!(
				"  depth {depth}: moved {movement_rms:.4}, floor {:.4}, {} leaves, {:.0?}",
				rms(&floor), base.truncated, started.elapsed(),
			);
		}
		rows.push(SensitivityRow {
			depth: *depth,
			movement_rms,
			floor_rms: rms(&floor),
			gamma_sensitivity: if noise_rms > 0.0 { movement_rms / noise_rms } else { f32::NAN },
			leaves_used: base.truncated as f32,
			base_value: base.value,
		});
	}

	SensitivityReport { noise_rms, rows }
}

impl fmt::Display for SensitivityReport {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(f, "injected leaf error, as applied at the horizon: RMS {:.4}", self.noise_rms)?;
		writeln!(f)?;
		writeln!(
			f,
			"{:>6}  {:>9}  {:>9}  {:>7}  {:>9}  {:>12}",
			"depth", "value", "moved by", "floor", "gamma_s", "leaves used",
		)?;
		for row in &self.rows {
			writeln!(
				f,
				"{:>6}  {:>+9.4}  {:>9.4}  {:>7.4}  {:>9.3}  {:>12}",
				row.depth,
				row.base_value,
				row.movement_rms,
				row.floor_rms,
				row.gamma_sensitivity,
				row.leaves_used as u64,
			)?;
		}
		writeln!(f)?;
		write!(
			f,
			"gamma_s far below 1 means leaf improvements are discarded before they reach the root;\n\
			 trust a row only where 'moved by' clearly exceeds 'floor' and 'leaves used' is not zero",
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::mirror_duel;

	fn registry() -> Registry {
		Registry::load()
	}

	/// The perturbed estimate has to stay zero-sum. If it does not, the solver is
	/// no longer solving a zero-sum game and the root value means nothing — which
	/// would quietly invalidate every sensitivity number rather than failing.
	#[test]
	fn a_perturbed_estimate_is_still_zero_sum() {
		let registry = registry();
		let leaf = Perturbed::new(Box::new(HealthHeuristic), 0.5, 11);

		for (index, (state, _)) in
			sample_positions(&registry, &ContractionConfig { walk: 2, ..Default::default() })
				.iter()
				.enumerate()
				.take(40)
		{
			let zero = leaf.value(state, Team::Zero, &registry);
			let one = leaf.value(state, Team::One, &registry);
			assert!(
				(zero + one).abs() < 1e-5,
				"position {index}: {zero} and {one} do not cancel",
			);
		}
	}

	/// A value function's error is a function of the position: it is wrong about
	/// the same position the same way every time. Were the offset re-rolled per
	/// visit the search would average it away, and the measurement would report
	/// that leaf error does not matter — for the wrong reason.
	#[test]
	fn the_same_position_is_always_perturbed_the_same_way() {
		let registry = registry();
		let state = mirror_duel(&registry);
		let leaf = Perturbed::new(Box::new(HealthHeuristic), 0.5, 3);

		let first = leaf.value(&state, Team::Zero, &registry);
		for _ in 0..8 {
			assert_eq!(first, leaf.value(&state, Team::Zero, &registry));
		}
	}

	#[test]
	fn the_perturbation_respects_its_scale_and_moves_with_its_salt() {
		let registry = registry();
		let positions =
			sample_positions(&registry, &ContractionConfig { walk: 2, ..Default::default() });

		let quiet = Perturbed::new(Box::new(HealthHeuristic), 0.1, 1);
		let loud = Perturbed::new(Box::new(HealthHeuristic), 0.9, 1);
		let resalted = Perturbed::new(Box::new(HealthHeuristic), 0.9, 2);

		let mut differs = 0;
		for (state, _) in positions.iter().take(40) {
			assert!(quiet.offset(state).abs() <= 0.1 + 1e-6);
			assert!(loud.offset(state).abs() <= 0.9 + 1e-6);
			if (loud.offset(state) - resalted.offset(state)).abs() > 1e-6 {
				differs += 1;
			}
		}
		assert!(differs > 30, "a new salt should redraw the error, only {differs}/40 moved");
	}

	/// The reference is only exact if the duel ends, so the movesets must carry
	/// nothing that can stall. *sap seed* is the one that is easy to miss: unlike
	/// every other move it restores health, to the seeder rather than the user.
	#[test]
	fn the_sampled_movesets_cannot_stall() {
		for species in ROSTER {
			let moves = moveset(species);
			assert!(!moves.is_empty(), "species {species} has nothing left to play");
			for id in moves {
				assert!(
					!STALLING_MOVES.contains(&id.0),
					"species {species} kept stalling move {}",
					id.0,
				);
			}
		}
	}

	#[test]
	fn every_sampled_position_is_live_and_a_real_duel() {
		let registry = registry();
		let positions =
			sample_positions(&registry, &ContractionConfig { walk: 2, ..Default::default() });

		assert!(positions.len() >= 36, "one per pairing at least, got {}", positions.len());
		for (state, request) in &positions {
			assert!(
				matches!(DecisionNode::from(state, request), DecisionNode::Decision { .. }),
				"a finished position has nothing to solve",
			);
		}
	}

	/// End to end, small enough to run by default. The claim being pinned is the
	/// weak one that holds whatever the answer turns out to be: searching a turn
	/// and then estimating beats estimating immediately.
	#[test]
	fn searching_beats_guessing() {
		let registry = registry();
		let report = measure(
			&registry,
			&ContractionConfig {
				truth_iterations: 200,
				solve_iterations: 150,
				depths: vec![1],
				walk: 1,
				..Default::default()
			},
		);

		assert!(report.positions > 20, "too few positions survived: {}", report.positions);
		assert!(report.sd_truth > 0.0, "the true values are all the same");
		let row = &report.rows[0];
		assert!(row.leaves_used > 0.0, "the leaf was never consulted, so this proves nothing");
		assert!(
			row.gamma_absolute < 1.0,
			"a one-turn search did worse than the raw estimate: gamma {}",
			row.gamma_absolute,
		);
	}
}
