//! Solve a 1v1 position and print the equilibrium strategies.
//!
//! This is the shape the matchup solver takes: point it at a position, get back
//! how often each action should be played. Unlike `strat-optimizer`, nothing is
//! trained and nothing is saved — a solve is about one position and is thrown
//! away afterwards.

use std::error::Error;

use rand::rngs::StdRng;
use rand::SeedableRng;

use strat_optimizer::battle::engine::engine::StepRequest;
use strat_optimizer::battle::state::battle_state::BattleState;
use strat_optimizer::cfr::exploit::{measure, ExploitConfig};
use strat_optimizer::cfr::node::DecisionNode;
use strat_optimizer::cfr::position;
use strat_optimizer::cfr::solver::{action_label, Solver, SolverConfig};
use strat_optimizer::model::registry::Registry;

fn main() -> Result<(), Box<dyn Error>> {
	let registry = Registry::load();

	let iterations: usize = std::env::args()
		.nth(1)
		.and_then(|arg| arg.parse().ok())
		.unwrap_or(5_000);

	// A 1v1 ends on its own, so it is searched to the end and owes nothing to the
	// leaf estimate. A 2v2 cannot be: with a switch on top of the moves, the tree
	// grows about fivefold per turn, so it is cut off at a horizon and estimated.
	let positions: Vec<(&str, BattleState, Option<usize>)> = vec![
		("known-answer duel, 1v1 (flame lash dominates)", position::known_answer_duel(&registry), None),
		("mirror duel, 1v1 (value must be zero)", position::mirror_duel(&registry), None),
		("switch-prediction, 2v2 (mirror)", position::switch_prediction_2v2(&registry), Some(6)),
	];

	for (name, root, lookahead) in positions {
		println!("\n=== {name} ===");
		match lookahead {
			Some(turns) => println!("solving {iterations} iterations, {turns} turns of lookahead..."),
			None => println!("solving {iterations} iterations, searched to the end..."),
		}

		let mut rng = StdRng::seed_from_u64(20260916);
		let config = match lookahead {
			Some(turns) => SolverConfig::for_lookahead(iterations, turns),
			None => SolverConfig { iterations, ..SolverConfig::default() },
		};
		let mut solver = Solver::new(&registry, config);
		solver.solve(&root, &mut rng);

		println!(
			"{} decision points, {} nodes expanded",
			solver.table().len(),
			solver.nodes_visited(),
		);
		if solver.hit_node_budget() {
			println!("  WARNING: stopped on the node budget — this answer is not converged");
		}
		if solver.truncated_positions() > 0 {
			println!(
				"  {} positions estimated at the horizon rather than played out",
				solver.truncated_positions(),
			);
		}

		let value = solver.evaluate(&root, 5_000, &mut rng);
		println!("value to Team Zero: {value:+.3}  (-1 always loses, +1 always wins)");
		if value.abs() > 0.98 {
			println!(
				"  note: one side wins from here whatever it does, so the losing side's"
			);
			println!(
				"        frequencies below are arbitrary — every action is worth the same"
			);
		}

		report(&root, &solver, &registry);

		// The one check that does not need the answer known in advance: how much
		// would a perfect opponent get out of this?
		//
		// Skipped where the best response cannot search as far as the solve did.
		// A best response enumerates both sides' actions, so it branches harder
		// than the solve does and cannot always follow it to the end; measuring a
		// shorter game than was solved gives a confident, meaningless number.
		let exploit_config = ExploitConfig::matching(&solver);
		if exploit_config.max_depth > 20 {
			println!(
				"\nexploitability: not measured — this position is searched to the end,\n\
				 and a best response cannot follow it that far"
			);
		} else {
			println!("\nexploitability:");
			println!("{}", measure(&solver, &root, exploit_config, &mut rng));
		}
	}

	Ok(())
}

/// Print each side's equilibrium frequencies at the position handed in.
fn report(root: &BattleState, solver: &Solver, registry: &Registry) {
	let actors = match DecisionNode::from(root, &StepRequest::NeedsActions) {
		DecisionNode::Decision { actors } => actors,
		DecisionNode::Terminal(outcome) => {
			println!("position is already finished: {outcome:?}");
			return;
		}
	};

	for actor in actors {
		let name = root
			.get_mon(actor.position)
			.map(|mon| registry.get_species_data(mon.species_id).name.clone())
			.unwrap_or_else(|| String::from("?"));

		println!("\n{:?} — {}", actor.team, name);

		let strategy = match solver.root_strategy(root, actor.team) {
			Some(strategy) => strategy,
			None => {
				println!("  (never reached)");
				continue;
			}
		};

		let mut ranked: Vec<(usize, f32)> = strategy
			.iter()
			.enumerate()
			.filter(|(action, _)| actor.mask.allowed[*action])
			.map(|(action, probability)| (action, *probability))
			.collect();
		ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

		for (action, probability) in ranked {
			println!(
				"  {:>6.2}%  {}",
				probability * 100.0,
				action_label(action, &actor, root, registry),
			);
		}
	}
}
