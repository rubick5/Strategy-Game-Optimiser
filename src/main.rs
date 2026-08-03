use poke_sim::rl::train;
use rand::prelude::SeedableRng;
use rand::rngs::StdRng;

fn main() {
	
	let mut rng = StdRng::seed_from_u64(67);
	train::main_loop(&mut rng);
}