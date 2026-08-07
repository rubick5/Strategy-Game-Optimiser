/// WE NEED TO ADD SOME ENCOURAGEMENT FOR ENTROPY INTO THE CODE
/// SO THAT IT DOESNT JUST CONVERGE LIKE CRAZY
// * Technique 1: x% of the time just pick a random action instead of the model's one
// * Technique 2: use a entropy reward in the output layer's error to encourage the model to stay
// * versatile. Also note that we can reduce the weight of this as we get further into training
// * once the correct strategies have actually been figured out.

use crate::rl::{mask::Mask, moveslot::{MAX_DECISION, Moveslot}, neural_net::NeuralNet};
use rand::{Rng, RngCore};

use std::{error::Error, f32::consts::E};

// max moveslot discriminant

pub const RELU_LEAK: f32 = 0.01;


pub const ENTROPY_REWARD_RATE: f32 = 0.01;

pub const BASELINE_LEARNING_RATE: f32 = 0.05;



/**
 * Computes the softmax of the logits given
 * 
 * Currently panics on empty logits
 */
fn softmax(logits: &[f32]) -> Vec<f32> {
	let m = logit_max(logits).unwrap();
	let divisor: f32 = logits.iter().map(|x| E.powf(*x - m)).sum();
	logits.iter().map(|x| E.powf(*x - m) / divisor).collect()
}

pub fn relu(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK * x } else { x }
}

pub fn relu_prime(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK } else { 1.0 }
}
/**
 * Finds the largest element in a slice of f32s
 * 
 * Returns none for empty slice
 */
fn logit_max(logits: &[f32]) -> Option<f32> {
	logits.iter().copied().reduce(f32::max)
}

/**
 * Performs softmax then randomly chooses an index based
 * on the probabilities calculated
 * 
 * Inherits panicking behaviour from softmax function on
 * empty logits
 */
fn softmax_then_select(logits: &[f32], rng: &mut dyn RngCore) -> usize {
	let probabilities = softmax(logits);
	let random_selection = rng.random();
	let mut counter = 0.0;
	for (index, prob) in probabilities.into_iter().enumerate() {
		if prob <= 0.0 {
			continue;
		}
		counter += prob;
		if counter >= random_selection {
			return index;
		}
	}
	0
}

pub trait Agent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot;
}

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

impl BotAgent
{
	pub fn relu_from_file(file_name: &str) -> Result<Self, Box<dyn Error>> {
		Self::from_file(file_name, relu, relu_prime)
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


	pub fn just_logits(&mut self, representation: &[f32], mask: &Mask) -> Vec<f32> {
		let mut output = self.net.forward(representation);
		mask.apply(&mut output);
		output
	}
	/**
	 * Runs the encoding through the network then applies the mask to the logits.
	 *
	 * Returns a pair (x, y) where x is a randomly selected move from the probabilities
	 * and y is the calculated probabilities.
	 */
	pub fn choose_move_with_probs(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> (Moveslot, Vec<f32>) {
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

	/**
	 * Initialises the agent neural network with random weights everywhere.

	 * Size and number of hidden layers are defined in a magic number vector here, at some point
	 * that should be moved out...
	 */
	pub fn init_random(weight_count: usize, rng: &mut dyn RngCore) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, MAX_DECISION];
		BotAgent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu, relu_prime),
			baseline: 0.0,
		}
	}

	/**
	 * Learns from previous mistakes or successes. Needs a bunch of weird arguments for the maths to work out.
	 */
	pub fn backprop(&mut self, move_slot: Moveslot, battle_reward: f32, gt: f32, encoding: &[f32], last_probabilities: &[f32]) {
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
				ENTROPY_REWARD_RATE * last_probabilities[index] * (last_probabilities[index].ln() + prob_entropy)
			};
			(battle_reward - self.baseline) * gt * (last_probabilities[index] - indicator) + entropy_reward
		}).collect();

		self.net.backward(current_errors, &encoding)
	}
}

