use crate::rl::{agent::{bot_agent::{BASELINE_LEARNING_RATE, BotAgent}, train_config::TrainConfig}, battle_playout::{PlayedBattle, Step}};

const DAMPING_CONSTANT: f32 = 0.95;
const BATCH_STANDARD_CONST: f32 = 0.05;
const WEIGHT_DECAY_AMOUNT: f32 = 0.99995;

pub fn learn_from_batch(agent: &mut BotAgent, batch: &Vec<PlayedBattle>, gt: f32, train_config: &TrainConfig) {
	if batch.is_empty() {
		return;
	}
	let initial_advantages: Vec<f32> = batch.iter().map(|b| {
		(b.battle_reward - agent.baseline) * gt
	}).collect();
	let initial_advantages_mean: f32 = initial_advantages.iter().sum::<f32>() / initial_advantages.len() as f32;

	let initial_advantages_squared_mean: f32 = initial_advantages.iter().map(|f| f * f).sum::<f32>() / initial_advantages.len() as f32;
	let initial_advantages_sigma: f32 = 
		(initial_advantages_squared_mean - initial_advantages_mean * initial_advantages_mean).max(0.0).sqrt();

	let standardised: Vec<f32> = (0..initial_advantages.len()).map(|index| {
		(initial_advantages[index] - initial_advantages_mean) / (initial_advantages_sigma + BATCH_STANDARD_CONST)
	}).collect();

	for battle_index in 0..batch.len() {
		learn_from_battle(&train_config, agent, &batch[battle_index], standardised[battle_index]);
	}

	pub fn learn_from_battle(train_config: &TrainConfig, agent: &mut BotAgent, played_battle: &PlayedBattle, scalar: f32) {
		let mut gt = scalar;
		agent.baseline += (played_battle.battle_reward - agent.baseline) * BASELINE_LEARNING_RATE;
		for Step { encoding, move_chosen, probabilities } in played_battle.steps.iter() {
			agent.backprop(
				*move_chosen,
				gt,
				&encoding,
				&probabilities,
				train_config.learning_rate,
				train_config.entropy_reward_rate,
				WEIGHT_DECAY_AMOUNT,
			);
			gt = gt * DAMPING_CONSTANT;
		}
	}
}

