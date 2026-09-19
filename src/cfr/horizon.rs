//! Does a lead's value depend on where the search stops?
//!
//! The six-a-side preview run turned up something the lead matrix alone cannot
//! settle. Team zero's two stonewardens share only half a moveset — one carries
//! *iron press* and *guard*, the other *rock smash* and *blade dance* — and yet
//! their rows differ by a mean of 0.0028 across the whole matrix, with the blade
//! dance set the worse of the two in every single column.
//!
//! Two readings fit that, and they are not close to equivalent:
//!
//! * **Blade dance is simply bad here.** Then the solver is right and there is
//!   nothing to fix.
//! * **Four turns is too short to price it.** A setup move pays its cost
//!   immediately and returns it over the turns that follow, so a horizon that
//!   sees the stat boost and not the sweep it buys will price setup at cost and
//!   call it a loss. Then the solver is wrong in a way that gets worse the more
//!   the position rewards patience, and the same error applies to every slow
//!   strategy in the game.
//!
//! What separates them is the *trend*. If the horizon is the problem, the blade
//! dance set's value relative to the other should climb as the search deepens. If
//! the move is simply bad, the gap should sit flat.
//!
//! Two things this has to get right or it reports noise as a finding.
//!
//! **Paired seeds.** The gap being chased is a few thousandths, and two MCCFR
//! runs of the same position disagree by about that much on their own. So both
//! leads are solved from the *same* seed at each depth and it is their difference
//! that is recorded — common random numbers, which cancels most of the sampling
//! variance because both runs draw the same chance outcomes.
//!
//! **A spread, not a point.** The difference is measured over several seeds and
//! its standard deviation reported next to it. A trend inside its own error bars
//! is not a trend, and this project has already published one number that turned
//! out to be noise.

use std::fmt;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::battle::engine::engine::StepRequest;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::Field;
use crate::battle::state::Team;
use crate::cfr::solver::{Solver, SolverConfig};
use crate::model::registry::Registry;

pub struct HorizonConfig {
	pub depths: Vec<usize>,
	pub iterations: usize,
	/// Independent repeats at each depth, which is where the error bar comes from.
	pub seeds: usize,
	pub seed: u64,
	pub max_nodes: u64,
	pub progress: bool,
}

impl Default for HorizonConfig {
	fn default() -> Self {
		HorizonConfig {
			depths: vec![2, 3, 4, 5],
			iterations: 1_500,
			seeds: 3,
			seed: 314_159,
			max_nodes: SolverConfig::default().max_nodes,
			progress: false,
		}
	}
}

pub struct HorizonRow {
	pub depth: usize,
	/// Mean value to team zero when it leads the first creature.
	pub a: f32,
	/// Mean value to team zero when it leads the second.
	pub b: f32,
	/// Mean of the paired `b - a`, which is the quantity with the small error bar.
	pub difference: f32,
	/// Standard deviation of that difference across seeds. A gap smaller than
	/// this is not a gap.
	pub spread: f32,
	pub leaves_used: f32,
	pub seconds: f32,
}

pub struct HorizonReport {
	pub label_a: String,
	pub label_b: String,
	pub rows: Vec<HorizonRow>,
}

/// Put `zero`'s creature and `one`'s creature on the field, leaving the benches
/// alone. Roster slots interleave: team zero at `i * 2`, team one at `j * 2 + 1`.
fn with_leads(root: &BattleState, zero: usize, one: usize) -> BattleState {
	let mut state = root.clone();
	state.field = Field::from(vec![zero * 2, one * 2 + 1]);
	state
}

fn value_of(
	registry: &Registry,
	state: &BattleState,
	depth: usize,
	config: &HorizonConfig,
	seed: u64,
) -> Option<(f32, u64)> {
	let mut rng = StdRng::seed_from_u64(seed);
	let mut solver = Solver::new(
		registry,
		SolverConfig {
			iterations: config.iterations,
			max_depth: depth,
			max_nodes: config.max_nodes,
			..SolverConfig::default()
		},
	);
	solver.log_values(0.5);
	solver.solve(state, &mut rng);
	solver
		.root_value(state, &StepRequest::NeedsActions, Team::Zero)
		.map(|value| (value, solver.truncated_positions()))
}

fn mean(values: &[f32]) -> f32 {
	if values.is_empty() {
		return f32::NAN;
	}
	values.iter().sum::<f32>() / values.len() as f32
}

fn sd(values: &[f32]) -> f32 {
	if values.len() < 2 {
		return f32::NAN;
	}
	let mean = mean(values);
	let variance =
		values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (values.len() - 1) as f32;
	variance.sqrt()
}

/// Solve two of team zero's leads against the same opponent, over a sweep of
/// horizons, and report how their difference moves.
pub fn compare_leads(
	registry: &Registry,
	root: &BattleState,
	zero_a: usize,
	zero_b: usize,
	one: usize,
	config: &HorizonConfig,
) -> HorizonReport {
	let name = |slot: usize, team: Team| {
		root.get_mon_from_team(&team, slot)
			.map(|mon| registry.get_species_data(mon.species_id).name.clone())
			.unwrap_or_else(|| String::from("?"))
	};

	let mut rows = Vec::new();
	for depth in &config.depths {
		let started = Instant::now();
		let (mut values_a, mut values_b, mut diffs, mut leaves) =
			(Vec::new(), Vec::new(), Vec::new(), Vec::new());

		for repeat in 0..config.seeds {
			// The same seed for both leads: their difference is then the lead,
			// not the sampler.
			let seed = config.seed.wrapping_add(repeat as u64 * 1_000_003);
			let a = value_of(registry, &with_leads(root, zero_a, one), *depth, config, seed);
			let b = value_of(registry, &with_leads(root, zero_b, one), *depth, config, seed);

			if let (Some((a, leaves_a)), Some((b, _))) = (a, b) {
				values_a.push(a);
				values_b.push(b);
				diffs.push(b - a);
				leaves.push(leaves_a as f32);
			}
		}

		let row = HorizonRow {
			depth: *depth,
			a: mean(&values_a),
			b: mean(&values_b),
			difference: mean(&diffs),
			spread: sd(&diffs),
			leaves_used: mean(&leaves),
			seconds: started.elapsed().as_secs_f32(),
		};
		if config.progress {
			println!(
				"  depth {:>2}: {:+.4} vs {:+.4}   difference {:+.4} +/- {:.4}   {:.0}s",
				row.depth, row.a, row.b, row.difference, row.spread, row.seconds,
			);
		}
		rows.push(row);
	}

	HorizonReport { label_a: name(zero_a, Team::Zero), label_b: name(zero_b, Team::Zero), rows }
}

impl fmt::Display for HorizonReport {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(f, "A = {} (slot one)", self.label_a)?;
		writeln!(f, "B = {} (slot two)", self.label_b)?;
		writeln!(f)?;
		writeln!(
			f,
			"{:>6}  {:>9}  {:>9}  {:>11}  {:>9}  {:>11}  {:>7}",
			"depth", "A", "B", "B - A", "+/- sd", "leaves", "secs",
		)?;
		for row in &self.rows {
			writeln!(
				f,
				"{:>6}  {:>+9.4}  {:>+9.4}  {:>+11.4}  {:>9.4}  {:>11.0}  {:>7.0}",
				row.depth,
				row.a,
				row.b,
				row.difference,
				row.spread,
				row.leaves_used,
				row.seconds,
			)?;
		}
		writeln!(f)?;
		write!(
			f,
			"B - A climbing with depth means the horizon was underpricing B.\n\
			 Flat means B is simply the weaker lead. Ignore any move smaller than the sd."
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::six_asymmetric;

	#[test]
	fn swapping_the_lead_changes_who_is_out_and_nothing_else() {
		let registry = Registry::load();
		let root = six_asymmetric(&registry);

		let first = with_leads(&root, 0, 0);
		let second = with_leads(&root, 3, 0);

		// The benches are untouched — only the creature on the field moves.
		assert_eq!(first.roster, second.roster, "changing the lead rebuilt the team");
		assert_ne!(first.field, second.field, "the lead did not actually change");
		assert!(first.outcome().is_none() && second.outcome().is_none());
	}

	/// The paired design is the whole point: solving one lead twice from the same
	/// seed has to give exactly the same number, or the difference carries the
	/// sampler's noise rather than the lead's effect.
	#[test]
	fn the_same_seed_gives_the_same_answer() {
		let registry = Registry::load();
		let root = six_asymmetric(&registry);
		let state = with_leads(&root, 0, 0);
		let config = HorizonConfig { iterations: 60, ..Default::default() };

		let first = value_of(&registry, &state, 2, &config, 7).unwrap();
		let second = value_of(&registry, &state, 2, &config, 7).unwrap();
		assert_eq!(first.0, second.0, "the solve is not reproducible from its seed");
	}

	#[test]
	fn a_sweep_reports_one_row_per_depth_with_an_error_bar() {
		let registry = Registry::load();
		let root = six_asymmetric(&registry);
		let config = HorizonConfig {
			depths: vec![1, 2],
			iterations: 80,
			seeds: 3,
			..Default::default()
		};

		let report = compare_leads(&registry, &root, 0, 1, 0, &config);
		assert_eq!(report.rows.len(), 2);
		for row in &report.rows {
			assert!(row.spread.is_finite(), "three seeds should give a spread");
			assert!(row.leaves_used > 0.0, "nothing was estimated, so depth is not binding");
		}
	}
}
