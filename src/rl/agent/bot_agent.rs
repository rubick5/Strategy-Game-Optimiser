use crate::rl::{agent::{Agent, LearningAgent, softmax, softmax_then_select, train_config::TrainConfig}, battle_playout::{PlayedBattle, Step}, mask::Mask, moveslot::{MAX_DECISION, Moveslot}, nn::neural_net::NeuralNet};
use rand::RngCore;

use std::error::Error;

pub const ENTROPY_REWARD_RATE: f32 = 0.01;

pub const BASELINE_LEARNING_RATE: f32 = 0.05;

const DAMPING_CONSTANT: f32 = 0.95;
const BATCH_STANDARD_CONST: f32 = 0.05;
const WEIGHT_DECAY_AMOUNT: f32 = 0.99;

#[derive(Debug, Clone)]
pub struct BotAgent
{
	pub net: NeuralNet<fn(f32) -> f32>,
	pub baseline: f32,
}

impl Agent for BotAgent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		self.choose_move_with_probs(representation, mask, rng).0
	}

	
}

impl LearningAgent for BotAgent {
	fn learn_from_batch(&mut self, batch: &Vec<PlayedBattle>, gt: f32, train_config: &TrainConfig) {
		if batch.is_empty() {
			return;
		}
		let initial_advantages: Vec<f32> = batch.iter().map(|b| {
			(b.battle_reward - self.baseline) * gt
		}).collect();
		let initial_advantages_mean: f32 = initial_advantages.iter().sum::<f32>() / initial_advantages.len() as f32;

		let initial_advantages_squared_mean: f32 = initial_advantages.iter().map(|f| f * f).sum::<f32>() / initial_advantages.len() as f32;
		let initial_advantages_sigma: f32 = 
			(initial_advantages_squared_mean - initial_advantages_mean * initial_advantages_mean).max(0.0).sqrt();

		let standardised: Vec<f32> = (0..initial_advantages.len()).map(|index| {
			(initial_advantages[index] - initial_advantages_mean) / (initial_advantages_sigma + BATCH_STANDARD_CONST)
		}).collect();

		for battle_index in 0..batch.len() {
			learn_from_battle(&train_config, self, &batch[battle_index], standardised[battle_index]);
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

	fn choose_move_with_probs(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> (Moveslot, Vec<f32>) {
		let mut logits = self.net.forward(representation);
		mask.apply(&mut logits); // edits them in place, remember

		let index_selected = softmax_then_select(&logits, rng);
		assert!(mask.allowed[index_selected],
			"illegal action selected!\n  index: {}\n  allowed: {:?}\n  probs: {:?}\n  logits: {:?}",
			index_selected,
			mask.allowed,
			softmax(&logits),
			logits,);

		(Moveslot::from_number(index_selected), softmax(&logits))
	}



	fn move_probs(&mut self, representation: &[f32], mask: &Mask) -> Vec<f32> {
		let mut output = self.net.forward(representation);
		mask.apply(&mut output);
		output
	}
}

impl BotAgent
{
	pub fn relu_from_file(file_name: &str) -> Result<Self, Box<dyn Error>> {
		Self::from_file(file_name, super::relu, super::relu_prime)
	}

	pub fn from_file(file_name: &str, hidden_activation: fn(f32) -> f32, hidden_activation_prime: fn(f32) -> f32) -> Result<Self, Box<dyn Error>> {
		let net_from_file = 
			NeuralNet::from_file(file_name, hidden_activation, hidden_activation_prime)?;
		Ok(Self {
			net: net_from_file,
			baseline: 0.0,
		})	
	}

	pub fn to_file(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
		self.net.to_file(file_name)
	}

	/**
	 * Runs the encoding through the network then applies the mask to the logits.
	 *
	 * Returns a pair (x, y) where x is a randomly selected move from the probabilities
	 * and y is the calculated probabilities.
	 */
		/**
	 * Initialises the agent neural network with random weights everywhere.

	 * Size and number of hidden layers are defined in a magic number vector here, at some point
	 * that should be moved out...
	 */
	pub fn init_random(weight_count: usize, rng: &mut dyn RngCore) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, MAX_DECISION];
		BotAgent {
			net: NeuralNet::gen_random(rng, &layer_sizes, super::relu, super::relu_prime),
			baseline: 0.0,
		}
	}

	/**
	 * Learns from previous mistakes or successes. Needs a bunch of weird arguments for the maths to work out.
	 */
	pub fn backprop(&mut self, move_slot: Moveslot, gt: f32, encoding: &[f32], last_probabilities: &[f32], learning_rate: f32, entropy_rate: f32, weight_decay: f32) {
		//let encoding = encoder::encode(battle_state, registry);
		//let mask = Mask::from_battle_state(team, pos, battle_state);

		let move_chosen_index = move_slot.to_number();
		let prob_entropy: f32 = last_probabilities.iter().filter(|p| **p != 0.0).map(|p| {
			- p * p.ln()
		}).sum();
		let current_errors: Vec<f32> = (0..last_probabilities.len()).map ( |index| {
			let indicator = if index == move_chosen_index { 1.0 } else { 0.0 };
			let entropy_reward = if last_probabilities[index] == 0.0 {
				0.0
			} else {
				entropy_rate * last_probabilities[index] * (last_probabilities[index].ln() + prob_entropy)
			};
			gt * (last_probabilities[index] - indicator) + entropy_reward
		}).collect();

		self.net.backward_with_decay(current_errors, &encoding, learning_rate, weight_decay);
	}
}
