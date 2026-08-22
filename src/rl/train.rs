use std::error::Error;

use rand::{Rng, RngCore, seq::{IndexedMutRandom, IndexedRandom as _}};

use crate::{battle::state::{battle_state::BattleState, field::PositionId}, model::registry::Registry, rl::{agent::{Agent, LearningAgent, bot_agent::BASELINE_LEARNING_RATE, random_agent::RandomAgent, spam_agent::SpamAgent, train_config::TrainConfig}, battle_playout::{PlayedBattle, play_out_battle}, encoder, mask::Mask}};
use crate::battle::state::Team;

pub const EXPLORATION_CHANCE: f32 = 0.00;
pub const LEARNING_RATE: f32 = 0.05;
pub const ENTROPY_REWARD_RATE: f32 = 0.05;

// note that the total number of battles used for training
// will be BATCH_COUNT * BATCH_SIZE
const BATCH_COUNT: usize = 2_000;
const BATCH_SIZE: usize = 32;

const BATCH_PRINT_FREQ: usize = 50;
const BATCH_PRINT_GAP_SIZE: usize = BATCH_SIZE * BATCH_PRINT_FREQ;
const BATCH_HISTORY_FREQ: usize = 50;


fn get_next_opponent<'a>(static_ops: &'a mut Vec<Box<dyn Agent>>, past_ops: &'a mut Vec<Box<dyn Agent>>, rng: &mut dyn RngCore) -> Option<&'a mut Box<dyn Agent>> {
	if past_ops.is_empty() || rng.random_bool(0.7) {
		static_ops.choose_mut(rng)
	} else {
		past_ops.choose_mut(rng)
	}

}

fn decay_train_config(train_config: &mut TrainConfig, batch_num: usize, total_batches: usize) {
	let batch_num_f32 = batch_num as f32;
	let total_batches_f32 = total_batches as f32;
	train_config.learning_rate = LEARNING_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32;
	train_config.entropy_reward_rate = ENTROPY_REWARD_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32 + 0.05;
}

pub fn main_loop(mut agent: impl LearningAgent + 'static, rng: &mut dyn RngCore, battle_state_paths: &[&str]) -> Result<(), Box<dyn Error>> {
	let registry = Registry::load();
	let mut battles_won = 0;

	let mut best_agent = agent.clone();
	let mut max_battles_won: i32 = 0;

	let mut train_config = TrainConfig {
		learning_rate: LEARNING_RATE,
		entropy_reward_rate: ENTROPY_REWARD_RATE,
		baseline_learning_rate: BASELINE_LEARNING_RATE
	};
	
	let mut static_opponents: Vec<Box<dyn Agent>> = vec![
		Box::new(RandomAgent{}),
		Box::new(SpamAgent{ index: 1 }),
		Box::new(SpamAgent{index: 0}),
		Box::new(SpamAgent{index: 2}),
		];

	let mut past_self_opponents: Vec<Box<dyn Agent>> = Vec::new();

	let battle_states: Vec<BattleState> = battle_state_paths.iter()
		.map(|s| BattleState::from_file(s)).collect::<Result<Vec<_>, _>>()?;


	for batch_num in 0..BATCH_COUNT {
		let mut current_batch: Vec<PlayedBattle> = Vec::new();
		for _ in 0..BATCH_SIZE {
			let opponent = get_next_opponent(&mut static_opponents, &mut past_self_opponents, rng).ok_or("no opponents available...")?;
			let battle: BattleState = battle_states.choose(rng).ok_or("no battle states available....")?.clone();

			let played_battle = play_out_battle(battle, &registry, &mut agent, opponent, rng);

			if played_battle.battle_reward > 0.0 {
				battles_won += 1;
			}
			current_batch.push(played_battle);
		}
		if batch_num % BATCH_PRINT_FREQ == 0 {
			decay_train_config(&mut train_config, batch_num, BATCH_COUNT);
			println!("batch {}: battles won: {} out of {}", batch_num, battles_won, BATCH_PRINT_GAP_SIZE);
			if battles_won > max_battles_won {
				best_agent = agent.clone();
				max_battles_won = battles_won;
			}
			battles_won = 0;
		}
		agent.learn_from_batch(&current_batch, 1.0, &train_config);

		if batch_num % BATCH_HISTORY_FREQ == 0 {
			past_self_opponents.push(Box::new(agent.clone()));
		}
	}
	println!("final state of agent:");
	final_agent_checks(&mut agent, &registry, &battle_states[0]);
	println!("highest winrate agent: ({} wins)", max_battles_won);
	final_agent_checks(&mut best_agent, &registry, &battle_states[0]);

	println!("Saving best agent to file: agent.json...");
	best_agent.to_file("agent.json")
}

fn final_agent_checks(agent: &mut impl LearningAgent, registry: &Registry, battle_state: &BattleState) {
	let agent_mask_normal = Mask::from_battle_state(&Team::Zero, PositionId(0), battle_state);

	println!("agent: {:?}", agent.move_probs(&encoder::encode(battle_state, registry, false), &agent_mask_normal));
}




