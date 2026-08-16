use crate::{battle::state::battle_state::BattleState, rl::{agent::{Agent, mean_squared_error}, battle_playout::PlayedBattle, mask::Mask, moveslot::Moveslot, nn::neural_net::NeuralNet}};
use crate::rl::encoder::TOTAL_ENCODING_LEN;
use crate::rl::moveslot::MAX_DECISION;
use rand::RngCore;

struct PPOStep {
	reward_to_go: f32,
	action: Moveslot,
	action_probability: f32,
	critic_value: f32,
	encoding: Vec<f32>,
}

// for ppo agent, we will model the battle's rewards as follows:
// -1 point per turn, +30 for win, -30 for lose
pub struct PPOAgent {
	pub actor: NeuralNet<fn(f32) -> f32>,
	pub critic: NeuralNet<fn(f32) -> f32>,
}

impl PPOAgent {
	pub fn gen_random(rng: &mut dyn RngCore) -> Self {
		let actor_layer_sizes: Vec<usize> = vec![TOTAL_ENCODING_LEN, 64, 128, 32, MAX_DECISION];
		let critic_layer_sizes: Vec<usize> = vec![TOTAL_ENCODING_LEN, 64, 128, 32, 1];
		Self {
			actor: NeuralNet::gen_random(rng, &actor_layer_sizes, super::relu, super::relu_prime),
			critic: NeuralNet::gen_random(rng, &critic_layer_sizes, super::relu, super::relu_prime),
		}
	}

	pub fn use_batch(&mut self, batch: Vec<PlayedBattle>, start_state: &BattleState) {
		for b in batch {

			let rewards_to_go = calc_reward_to_go(&b);

			let ppo_steps: Vec<PPOStep> = rewards_to_go.into_iter().zip(b.steps).map(
				|(rtg, step)| {
					PPOStep {
						reward_to_go: rtg,
						action: step.move_chosen,
						action_probability: step.chosen_prob(),
						critic_value: self.critic.forward(&step.encoding)[0],
						encoding: step.encoding,
					}
				}).collect();

			// step 1: update the critic
			for step in ppo_steps.iter() {
				let grad = 2.0 * (step.critic_value - step.reward_to_go);
				self.critic.backward(vec![grad], &step.encoding, 0.05);
			}

			// step 2: normalise the advantages:
		}
	}

	fn consume_ppo_step(&mut self, ppo_step: PPOStep) {
		let critic_error = ppo_step.critic_value - ppo_step.reward_to_go;
		self.critic.backward(vec![critic_error], &ppo_step.encoding, 0.05);
	}
}

impl Agent for PPOAgent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		todo!()
	}
}

fn calc_reward_to_go(battle: &PlayedBattle) -> Vec<f32> {
	let mut running = battle.battle_reward * 30.0;
	let gamma = 0.95;

	let mut returns: Vec<f32> = vec![0.0; battle.steps.len()];
	for t in (0..battle.steps.len()).rev() {
		running = -1.0 + gamma * running;
		returns[t] = running;
	}
	returns
}