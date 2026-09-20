//! Long solves, and lead selection.
//!
//! `solve` runs one position for as long as you let it, reporting as it goes.
//! `leads` answers the team-preview question: given two teams, which creature
//! should each side send out, and how often?
//!
//! Both print a running trace rather than only a final answer, because the
//! interesting question about a long run is usually whether it had settled.

use std::error::Error;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use strat_optimizer::battle::engine::engine::StepRequest;
use strat_optimizer::battle::state::battle_state::BattleState;
use strat_optimizer::battle::state::{Team, TEAM_SIZE};
use strat_optimizer::cfr::contraction::{self, ContractionConfig, SensitivityConfig};
use strat_optimizer::cfr::critic::{self, TrainingConfig};
use strat_optimizer::cfr::horizon::{self, HorizonConfig};
use strat_optimizer::cfr::leaf::LeafSource;
use strat_optimizer::cfr::matrix::MatrixGame;
use strat_optimizer::cfr::node::DecisionNode;
use strat_optimizer::cfr::position;
use strat_optimizer::cfr::solver::{action_label, Solver, SolverConfig, DEFAULT_QUIESCENCE};
use strat_optimizer::model::registry::Registry;
use strat_optimizer::rl::moveslot::MAX_DECISION;

fn usage() -> String {
	String::from(
		"usage:\n  \
		 deep solve <position> <lookahead> <iterations> [report-every]\n  \
		 deep leads <position> <lookahead> <iterations-per-matchup>\n  \
		 deep contraction [walk] [truth-iters] [solve-iters] [noise]\n  \
		 deep sensitivity <position> <iterations> [noise] [samples]\n  \
		 deep horizon <position> <zero-a> <zero-b> <one> [iters] [seeds] [max-depth]\n  \
		 deep train-critic <out-path> [rounds]\n\n\
		 any solving command takes --leaf health | resource | critic:<path>\n  \
		 and --quiescence <n> to carry loud positions past the horizon\n\n\
		 <position> is a builtin name or a path to a battle JSON:\n  \
		 switch_prediction_2v2 | delayed_setup | setup_duel | full_team_mirror | ...",
	)
}

fn load(name: &str, registry: &Registry) -> Result<BattleState, Box<dyn Error>> {
	Ok(match name {
		"switch_prediction_2v2" => position::switch_prediction_2v2(registry),
		"delayed_setup" => position::delayed_setup(registry),
		"guard_probe" => position::guard_probe(registry),
		"plain_probe" => position::plain_probe(registry),
		"full_team_mirror" => position::full_team_mirror(registry),
		"known_answer_duel" => position::known_answer_duel(registry),
		"mirror_duel" => position::mirror_duel(registry),
		"setup_duel" => position::setup_duel(registry),
		path => BattleState::from_file(path)?,
	})
}

/// Pull `--leaf <spec>` out of the argument list wherever it appears.
///
/// Removed rather than read in place so that the positional arguments after it
/// keep their numbering — otherwise adding the flag would silently shift every
/// index a command reads and change what it solves.
fn take_leaf(args: &mut Vec<String>) -> Result<LeafSource, Box<dyn Error>> {
	let Some(at) = args.iter().position(|arg| arg == "--leaf") else {
		return Ok(LeafSource::default());
	};
	if at + 1 >= args.len() {
		return Err("--leaf needs a value: health | resource | critic:<path>".into());
	}
	let spec = args.remove(at + 1);
	args.remove(at);
	LeafSource::parse(&spec)
}

/// Pull `--quiescence <n>` out of the argument list, like [`take_leaf`].
///
/// Absent means [`DEFAULT_QUIESCENCE`], not zero. Returning zero here would have
/// the flag's *default* silently disable a fix the solver turns on by itself —
/// which it did, and the giveaway was a sweep whose numbers came back identical
/// to four decimal places. Pass `--quiescence 0` to actually turn it off.
fn take_quiescence(args: &mut Vec<String>) -> Result<usize, Box<dyn Error>> {
	let Some(at) = args.iter().position(|arg| arg == "--quiescence") else {
		return Ok(DEFAULT_QUIESCENCE);
	};
	if at + 1 >= args.len() {
		return Err("--quiescence needs a number of extra turns".into());
	}
	let turns = args.remove(at + 1).parse()?;
	args.remove(at);
	Ok(turns)
}

fn main() -> Result<(), Box<dyn Error>> {
	let mut args: Vec<String> = std::env::args().collect();
	let leaf = take_leaf(&mut args)?;
	let quiescence = take_quiescence(&mut args)?;
	let args = args;

	if args.get(1).map(String::as_str) == Some("train-critic") && args.len() >= 3 {
		return train_critic(&args);
	}

	// Takes no position: it builds its own, because the reference value has to be
	// exact and only a 1v1 searched to the end gives one.
	if args.get(1).map(String::as_str) == Some("contraction") {
		contraction(&Registry::load(), &args, &leaf);
		return Ok(());
	}

	// Needs a position but not a lookahead: it sweeps the depths itself.
	if args.get(1).map(String::as_str) == Some("sensitivity") && args.len() >= 4 {
		let registry = Registry::load();
		let root = load(&args[2], &registry)?;
		sensitivity(&registry, &root, &args, &leaf);
		return Ok(());
	}

	// Compares two of team zero's leads across a sweep of horizons, so it takes
	// three slot numbers rather than a single lookahead.
	if args.get(1).map(String::as_str) == Some("horizon") && args.len() >= 6 {
		let registry = Registry::load();
		let root = load(&args[2], &registry)?;
		horizon_sweep(&registry, &root, &args, &leaf, quiescence)?;
		return Ok(());
	}

	if args.len() < 5 {
		println!("{}", usage());
		return Ok(());
	}

	let registry = Registry::load();
	let root = load(&args[2], &registry)?;
	let lookahead: usize = args[3].parse()?;
	let iterations: usize = args[4].parse()?;

	match args[1].as_str() {
		"solve" => {
			let report_every: usize =
				args.get(5).map(|a| a.parse()).transpose()?.unwrap_or(iterations / 20).max(1);
			solve(&registry, &root, lookahead, iterations, report_every, &args[2], quiescence);
		}
		"leads" => leads(&registry, &root, lookahead, iterations, quiescence),
		_ => println!("{}", usage()),
	}
	Ok(())
}

/// How much of a leaf estimate's error does the search remove?
fn contraction(registry: &Registry, args: &[String], leaf: &LeafSource) {
	let number = |index: usize| args.get(index).and_then(|arg| arg.parse().ok());
	let config = ContractionConfig {
		walk: number(2).unwrap_or(2),
		truth_iterations: number(3).unwrap_or(4_000),
		solve_iterations: number(4).unwrap_or(1_500),
		noise: args.get(5).and_then(|arg| arg.parse().ok()).unwrap_or(0.5),
		progress: true,
		..ContractionConfig::default()
	};

	println!("=== contraction ===");
	println!("  leaf under test: {}", leaf.label());
	println!(
		"  reference: 1v1 solved to the end, {} iterations",
		config.truth_iterations,
	);
	println!(
		"  depth-limited solves: {} iterations at depths {:?}",
		config.solve_iterations, config.depths,
	);

	let start = Instant::now();
	let report = contraction::measure(registry, &config, leaf);
	println!("\n{report}");
	println!("\ntotal {:.1?}", start.elapsed());
}

/// Does the leaf estimate reach the root at a position that actually needs one?
fn sensitivity(registry: &Registry, root: &BattleState, args: &[String], leaf: &LeafSource) {
	let number = |index: usize| args.get(index).and_then(|arg| arg.parse().ok());
	let config = SensitivityConfig {
		iterations: number(3).unwrap_or(1_500),
		noise: args.get(4).and_then(|arg| arg.parse().ok()).unwrap_or(0.5),
		samples: number(5).unwrap_or(6),
		progress: true,
		..SensitivityConfig::default()
	};

	println!("=== sensitivity: {} ===", args[2]);
	println!("  leaf under test: {}", leaf.label());
	describe(registry, root);
	println!(
		"  {} iterations per solve, {} perturbations of {:.2} at depths {:?}\n",
		config.iterations, config.samples, config.noise, config.depths,
	);

	let start = Instant::now();
	let report =
		contraction::sensitivity(registry, root, &StepRequest::NeedsActions, &config, leaf);
	println!("\n{report}");
	println!("\ntotal {:.1?}", start.elapsed());
}

/// Is a lead underpriced because the search stops too early?
fn horizon_sweep(
	registry: &Registry,
	root: &BattleState,
	args: &[String],
	leaf: &LeafSource,
	quiescence: usize,
) -> Result<(), Box<dyn Error>> {
	let number = |index: usize| args.get(index).and_then(|arg| arg.parse().ok());
	let (zero_a, zero_b, one) = (args[3].parse()?, args[4].parse()?, args[5].parse()?);
	let config = HorizonConfig {
		iterations: number(6).unwrap_or(1_500),
		seeds: number(7).unwrap_or(3),
		// Cost multiplies about fivefold per extra turn, so the last depth
		// dominates the run and is worth being able to drop.
		depths: (2..=number(8).unwrap_or(6)).collect(),
		quiescence,
		progress: true,
		..HorizonConfig::default()
	};

	println!("=== horizon: {} ===", args[2]);
	println!("  leaf under test: {}", leaf.label());
	describe(registry, root);
	println!(
		"  team zero slot {zero_a} against slot {zero_b}, both versus team one slot {one}",
	);
	println!(
		"  {} iterations, {} paired seeds, depths {:?}",
		config.iterations, config.seeds, config.depths,
	);
	println!("  quiescence: {} extra turns for positions still in motion\n", config.quiescence);

	let start = Instant::now();
	let report = horizon::compare_leads(registry, root, zero_a, zero_b, one, &config, leaf);
	println!("\n{report}");
	println!("\ntotal {:.1?}", start.elapsed());
	Ok(())
}

/// Train a critic and write it out, so the solving commands can load it.
///
/// Nothing else in the project produces one: `train` builds a critic in memory
/// and the callers so far have used it and thrown it away. A leaf estimate that
/// has to be retrained before every measurement cannot be compared against
/// anything, so it gets saved.
fn train_critic(args: &[String]) -> Result<(), Box<dyn Error>> {
	let path = &args[2];
	let rounds: usize = args.get(3).and_then(|arg| arg.parse().ok()).unwrap_or(4);

	let registry = Registry::load();
	let config = TrainingConfig { rounds, ..TrainingConfig::default() };
	let curriculum = critic::default_curriculum(&registry);

	println!("=== train-critic ===");
	println!("  {} training positions, {rounds} rounds", curriculum.len());
	println!("  each round re-solves with what has been learned so far\n");

	let start = Instant::now();
	let mut rng = StdRng::seed_from_u64(20260919);
	let (critic, reports) = critic::train(&registry, &curriculum, &config, &mut rng);

	for report in &reports {
		println!(
			"  round {}: {} positions, {} labels ({:.1} each), mse {:.5}",
			report.round,
			report.positions,
			report.labels,
			report.labels_per_position,
			report.mean_squared_error,
		);
	}

	critic.to_file(path)?;
	println!("\nwrote {path} after {:.1?}", start.elapsed());
	println!("use it with:  deep <command> ... --leaf critic:{path}");
	Ok(())
}

/// Describe the position, so a pasted log is self-contained.
fn describe(registry: &Registry, root: &BattleState) {
	let alive = |team: &Team| {
		(0..TEAM_SIZE)
			.filter_map(|i| root.get_mon_from_team(team, i))
			.map(|mon| registry.get_species_data(mon.species_id).name.clone())
			.collect::<Vec<_>>()
	};
	println!("  team zero: {}", alive(&Team::Zero).join(", "));
	println!("  team one : {}", alive(&Team::One).join(", "));

	if let DecisionNode::Decision { actors } = DecisionNode::from(root, &StepRequest::NeedsActions) {
		for actor in actors {
			let legal: Vec<String> = (0..MAX_DECISION)
				.filter(|i| actor.mask.allowed[*i])
				.map(|i| action_label(i, &actor, root, registry))
				.collect();
			println!("  {:?} options ({}): {}", actor.team, legal.len(), legal.join(" | "));
		}
	}
}

/// The strategy at the root, as a printable line.
fn strategy_line(
	registry: &Registry,
	root: &BattleState,
	solver: &Solver,
	team: Team,
) -> String {
	let actors = match DecisionNode::from(root, &StepRequest::NeedsActions) {
		DecisionNode::Decision { actors } => actors,
		DecisionNode::Terminal(_) => return String::from("(finished)"),
	};
	let Some(actor) = actors.iter().find(|a| a.team == team) else {
		return String::from("(not deciding)");
	};
	let Some(strategy) = solver.root_strategy(root, team) else {
		return String::from("(never reached)");
	};

	let mut parts: Vec<(f32, String)> = (0..MAX_DECISION)
		.filter(|i| actor.mask.allowed[*i])
		.map(|i| (strategy[i], action_label(i, actor, root, registry)))
		.collect();
	parts.sort_by(|a, b| b.0.total_cmp(&a.0));
	parts
		.iter()
		.map(|(p, name)| format!("{name} {:.0}%", p * 100.0))
		.collect::<Vec<_>>()
		.join("  ")
}

fn solve(
	registry: &Registry,
	root: &BattleState,
	lookahead: usize,
	iterations: usize,
	report_every: usize,
	name: &str,
	quiescence: usize,
) {
	println!("=== solve: {name} ===");
	describe(registry, root);
	println!(
		"  lookahead {lookahead}, {iterations} iterations, reporting every {report_every}"
	);
	println!("  quiescence: {quiescence} extra turns for positions still in motion");
	println!("  (cost grows about fivefold per extra turn of lookahead)\n");

	let mut rng = StdRng::seed_from_u64(20260918);
	let mut solver = Solver::new(
		registry,
		SolverConfig {
			iterations: report_every,
			max_depth: lookahead,
			quiescence,
			..SolverConfig::default()
		},
	);
	solver.log_values(0.5);

	let start = Instant::now();
	let chunks = iterations.div_ceil(report_every);

	for chunk in 1..=chunks {
		solver.solve(root, &mut rng);
		let done = chunk * report_every;

		println!("[{done}/{iterations}] {:.1?} elapsed", start.elapsed());
		println!(
			"    {} nodes, {} infosets, {} estimated at the horizon",
			solver.nodes_visited(),
			solver.table().len(),
			solver.truncated_positions(),
		);
		if let Some(value) = solver.root_value(root, &StepRequest::NeedsActions, Team::Zero) {
			println!("    value to team zero: {value:+.4}");
		}
		println!("    zero: {}", strategy_line(registry, root, &solver, Team::Zero));
		println!("    one : {}", strategy_line(registry, root, &solver, Team::One));
		if solver.hit_node_budget() {
			println!("    STOPPED on the node budget — this answer is not converged");
			break;
		}
		println!();
	}

	println!("=== done in {:.1?} ===", start.elapsed());
	println!("value (playing it out): {:+.4}", solver.evaluate(root, 4_000, &mut rng));
	println!("zero: {}", strategy_line(registry, root, &solver, Team::Zero));
	println!("one : {}", strategy_line(registry, root, &solver, Team::One));
}

/// Which creature should each side lead, and how often?
///
/// Every pairing is solved separately, giving a value for each cell of a team
/// preview matrix. The lead choice itself is then a simultaneous decision over
/// those values — a matrix game, solved exactly — so the answer is a mixture
/// rather than a single best lead, which is what the question actually calls for
/// whenever no lead is safe against everything.
fn leads(
	registry: &Registry,
	root: &BattleState,
	lookahead: usize,
	iterations: usize,
	quiescence: usize,
) {
	println!("=== leads ===");
	describe(registry, root);

	let members = |team: &Team| -> Vec<(usize, String)> {
		(0..TEAM_SIZE)
			.filter_map(|i| {
				root.get_mon_from_team(team, i)
					.filter(|mon| mon.current_hp > 0)
					.map(|mon| (i, registry.get_species_data(mon.species_id).name.clone()))
			})
			.collect()
	};
	let zero = members(&Team::Zero);
	let one = members(&Team::One);
	println!(
		"  {} x {} = {} matchups, {iterations} iterations each at lookahead {lookahead}",
		zero.len(),
		one.len(),
		zero.len() * one.len(),
	);
	println!("  quiescence: {quiescence} extra turns for positions still in motion\n");

	let start = Instant::now();
	let mut payoff: Vec<Vec<f32>> = Vec::new();

	for (i, zero_name) in &zero {
		let mut row = Vec::new();
		for (j, one_name) in &one {
			// Roster slots interleave: team zero at i*2, team one at j*2+1.
			let mut state = root.clone();
			state.field = strat_optimizer::battle::state::field::Field::from(vec![i * 2, j * 2 + 1]);

			let mut rng = StdRng::seed_from_u64(20260918);
			let mut solver = Solver::new(
				registry,
				SolverConfig { iterations, max_depth: lookahead, ..SolverConfig::default() },
			);
			solver.log_values(0.5);
			solver.solve(&state, &mut rng);

			let value = solver
				.root_value(&state, &StepRequest::NeedsActions, Team::Zero)
				.unwrap_or_else(|| solver.evaluate(&state, 2_000, &mut rng));
			row.push(value);

			println!(
				"  {zero_name} vs {one_name}: {value:+.4}   ({} infosets, {:.1?} elapsed)",
				solver.table().len(),
				start.elapsed(),
			);
		}
		payoff.push(row);
	}

	println!("\n=== lead matrix (value to team zero) ===");
	print!("{:>14}", "");
	for (_, name) in &one {
		print!("{name:>14}");
	}
	println!();
	for (index, (_, name)) in zero.iter().enumerate() {
		print!("{name:>14}");
		for value in &payoff[index] {
			print!("{value:>+14.4}");
		}
		println!();
	}

	if zero.len() > MAX_DECISION || one.len() > MAX_DECISION {
		println!("\n(too many leads to solve the preview game here)");
		return;
	}

	let game = MatrixGame::new(payoff);
	let (zero_mix, one_mix) = game.solve(200_000);

	println!("\n=== how often to lead each ===");
	for (index, (_, name)) in zero.iter().enumerate() {
		println!("  zero  {name:>14}: {:>6.1}%", zero_mix[index] * 100.0);
	}
	for (index, (_, name)) in one.iter().enumerate() {
		println!("  one   {name:>14}: {:>6.1}%", one_mix[index] * 100.0);
	}
	println!(
		"\nvalue of the preview game to team zero: {:+.4}",
		game.value(&zero_mix, &one_mix),
	);
	println!(
		"lead-choice exploitability: {:.5}   (0 means this mixture is unbeatable *given the matrix*)",
		game.exploitability(&zero_mix, &one_mix),
	);
	println!("\ntotal {:.1?}", start.elapsed());
}
