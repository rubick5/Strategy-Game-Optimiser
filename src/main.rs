use strat_optimizer::rl::agent::bot_agent::BotAgent;
use strat_optimizer::rl::agent::ppo_agent::PPOAgent;
use strat_optimizer::rl::{encoder, train};
use rand::prelude::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
	let battle_states = vec![
		"example_battles/normal_battle.json",
	];
	let mut rng = StdRng::seed_from_u64(871);
	//let agent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let agent = PPOAgent::init_random(&mut rng);
	train::main_loop(agent, &mut rng, &battle_states)
}