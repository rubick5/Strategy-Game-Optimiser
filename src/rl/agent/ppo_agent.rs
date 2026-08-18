use std::error::Error;

use crate::{rl::{agent::{Agent, LearningAgent}, battle_playout::PlayedBattle, mask::Mask, moveslot::Moveslot, nn::neural_net::NeuralNet}};
use crate::rl::encoder::TOTAL_ENCODING_LEN;
use crate::rl::moveslot::MAX_DECISION;
use rand::RngCore;

// NEXT THING TO DOOOOOOOOOOOo
// STORE MASK IN THE STEP STRUCT, ATM MASK DOESNT EXIST AND WE NEED IT
// WHEN LEARNING IN THIS AGENT.......

const EPSILON: f32 = 0.2;
const T: usize = 5;

struct PPOStep {
	reward_to_go: f32,
	action: Moveslot,
	action_probability: f32,
	critic_value: f32,
	encoding: Vec<f32>,
	mask: Mask,
}

// for ppo agent, we will model the battle's rewards as follows:
// -1 point per turn, +30 for win, -30 for lose

#[derive(Clone)]
pub struct PPOAgent {
	pub actor: NeuralNet<fn(f32) -> f32>,
	pub critic: NeuralNet<fn(f32) -> f32>,
}

impl LearningAgent for PPOAgent {
	fn learn_from_batch(&mut self, batch: &Vec<PlayedBattle>, _: f32, _: &super::train_config::TrainConfig) {
		self.use_batch(batch.to_vec());
	}

	fn choose_move_with_probs(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> (Moveslot, Vec<f32>) {
		let mut probs = self.actor.forward(representation);
		mask.apply(&mut probs);
		(Moveslot::from_number(super::softmax_then_select(&probs, rng)), probs)
	}

	fn move_probs(&mut self, representation: &[f32], mask: &Mask) -> Vec<f32> {
		let mut probs = self.actor.forward(representation);
		mask.apply(&mut probs);
		super::softmax(&probs)
	}

	fn to_file(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
		self.actor.to_file(file_name)
	}
	
}

impl Agent for PPOAgent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		let mut probs = self.actor.forward(representation);
		mask.apply(&mut probs);
		Moveslot::from_number(super::softmax_then_select(&probs, rng))
	}
}

impl PPOAgent {
	pub fn init_random(rng: &mut dyn RngCore) -> Self {
		let actor_layer_sizes: Vec<usize> = vec![TOTAL_ENCODING_LEN, 32, 32, MAX_DECISION];
		let critic_layer_sizes: Vec<usize> = vec![TOTAL_ENCODING_LEN, 32, 32, 1];
		Self {
			actor: NeuralNet::gen_random(rng, &actor_layer_sizes, super::relu, super::relu_prime),
			critic: NeuralNet::gen_random(rng, &critic_layer_sizes, super::relu, super::relu_prime),
		}
	}

	fn use_batch(&mut self, batch: Vec<PlayedBattle>) {
		let mut all_steps: Vec<PPOStep> = Vec::new();
		for b in batch {

			let rewards_to_go = calc_reward_to_go(&b);

			let mut ppo_steps = rewards_to_go.into_iter().zip(b.steps).map(
				|(rtg, step)| {
					PPOStep {
						reward_to_go: rtg,
						action: step.move_chosen,
						action_probability: step.chosen_prob(),
						critic_value: self.critic.forward(&step.encoding)[0],
						encoding: step.encoding,
						mask: step.mask,
					}
				}).collect();
			all_steps.append(&mut ppo_steps);
		}

		let raw_advantages: Vec<f32> = all_steps.iter().map(|step| {
				step.reward_to_go - step.critic_value
			}).collect();

		for _ in 0..T {
			let normalised_advantages = super::normalise_floats(&raw_advantages);

			for (advantage, step) in normalised_advantages.iter().zip(all_steps.iter()) {
				let v = self.critic.forward(&step.encoding)[0];
				let probs = self.move_probs(&step.encoding, &step.mask);
				let p_current = probs[step.action.to_number()];

				let grad = (2.0 * (v - step.reward_to_go)).clamp(-10.0, 10.0);
				//println!("critic 'grad': {}", grad);
				
				self.critic.backward(vec![grad], &step.encoding, 0.005);

				let coeff = Self::lclip_prime(p_current, step.action_probability, *advantage, EPSILON);
				let h: f32 = probs.iter().filter(|x| **x != 0.0).map(|x| -x * x.ln()).sum();
				let error = (0..probs.len()).map(|i| {
					let indicator = if i == step.action.to_number() { 1.0 } else { 0.0 };
					let entropy_term = if probs[i] == 0.0 { 0.0 } else {
						probs[i] * (probs[i].ln() + h)
					};
					coeff * (probs[i] - indicator) + entropy_term * 0.3
				}).collect();
				self.actor.backward(error, &step.encoding, 0.01);
			}
		}
	}

	fn lclip_prime(p_current: f32, p_old: f32, advantage: f32, epsilon: f32) -> f32 {
		let rt = (p_current.ln() - p_old.ln()).exp();
		let needs_clipping =
			(advantage > 0.0 && rt > 1.0 + epsilon)
			|| (advantage < 0.0 && rt < 1.0 - epsilon);

		if needs_clipping { 0.0 } else { advantage * rt }
	}

	fn _lclip(p_current: f32, p_old: f32, advantage: f32, epsilon: f32) -> f32 {
		let rt = p_current / p_old;
		let left = advantage * rt;
		let right = rt.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage;
		left.min(right)
	}
}

fn calc_reward_to_go(battle: &PlayedBattle) -> Vec<f32> {
	let mut running = battle.battle_reward * 3.0;
	let gamma = 0.95;

	let mut returns: Vec<f32> = vec![0.0; battle.steps.len()];
	for t in (0..battle.steps.len()).rev() {
		running = gamma * running; // -1.0
		returns[t] = running;
	}
	returns
}