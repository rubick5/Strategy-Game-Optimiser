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

	let mut rng = StdRng::seed_from_u64(6767);
	//let agent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let agent = PPOAgent::init_random(&mut rng);
	train::main_loop(agent, &mut rng, &battle_states)
}
