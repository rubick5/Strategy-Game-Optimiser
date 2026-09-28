use strat_optimizer::rl::agent::ppo_agent::PPOAgent;
use strat_optimizer::rl::train;
use rand::prelude::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
	// The new roster. `normal_battle.json` still loads, but both its species are
	// the original test dummies — one of which has 1000 Defense — so it is a poor
	// training signal now that there is a real roster to learn against.
	//
	// A battle state is picked at random per battle, so listing several exposes
	// the agent to more than one match-up.
	let battle_states = vec![
		"example_battles/fair_start_battle.json",
		/*
		"example_battles/ability_showcase.json",
		"example_battles/sand_vs_levitate.json",
		"example_battles/status_duel.json",
		"example_battles/coverage_test.json", */
	];

	// usage: strat-optimizer [batches] [out-path]
	//
	// Both default, so a bare `cargo run --release` is the full run it always
	// was. A short run is worth having because the exploit probe downstream can
	// only be rehearsed against a file that exists — and the schedules scale
	// with the run, so a short one exercises the whole curriculum.
	let args: Vec<String> = std::env::args().collect();
	let batches: usize = args
		.get(1)
		.and_then(|arg| arg.parse().ok())
		.unwrap_or(train::BATCH_COUNT);
	let out_path = args.get(2).map(String::as_str).unwrap_or("agent.json");

	println!("training {batches} batches, saving the best to {out_path}");

	let mut rng = StdRng::seed_from_u64(6767);
	//let agent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let agent = PPOAgent::init_random(&mut rng);
	train::main_loop(agent, &mut rng, &battle_states, batches, out_path)
}
