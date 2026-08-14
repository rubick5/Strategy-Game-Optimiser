use strat_optimizer::rl::train;
use rand::prelude::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
	let battle_states = vec![
		"example_battles/normal_battle.json",
	];
	let mut rng = StdRng::seed_from_u64(10);
	train::main_loop(&mut rng, &battle_states)
}