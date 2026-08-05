use crate::rl::{agent::{BASELINE_LEARNING_RATE, BotAgent}, battle_playout::Step};

const DAMPING_CONSTANT: f32 = 0.95;

pub fn learn_from_battle(agent: &mut BotAgent, battle_reward: f32, steps: Vec<Step>) {
	let mut gt = 1.0;
	agent.baseline += (battle_reward - agent.baseline) * BASELINE_LEARNING_RATE;
	for Step { encoding, move_chosen, probabilities } in steps {
			// don't forget to use 'reward' in here somewhere
		agent.backprop(move_chosen, battle_reward, gt, &encoding, &probabilities);
		gt = gt * DAMPING_CONSTANT;
	}
}