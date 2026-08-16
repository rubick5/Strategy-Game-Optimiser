pub mod train_config;
pub mod bot_agent;
pub mod random_agent;
pub mod spam_agent;
pub mod ppo_agent;

use std::f32::consts::E;

use crate::rl::{mask::Mask, moveslot::Moveslot};
use rand::{Rng as _, RngCore};

pub const RELU_LEAK: f32 = 0.01;


pub trait Agent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot;
}

impl <A: Agent + ?Sized> Agent for Box<A> {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		(**self).choose_move(representation, mask, rng)
	}
}

pub fn mean_squared_error(guess: f32, values: &[f32]) -> f32 {
	values.iter().map(|v| (guess - v) * (guess - v)).sum()
}


pub fn relu(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK * x } else { x }
}

pub fn relu_prime(x: f32) -> f32 {
	if x < 0.0 { RELU_LEAK } else { 1.0 }
}

/**
 * Computes the softmax of the logits given
 * 
 * Currently panics on empty logits
 */
pub fn softmax(logits: &[f32]) -> Vec<f32> {
	let m = logit_max(logits).unwrap();
	let divisor: f32 = logits.iter().map(|x| E.powf(*x - m)).sum();
	logits.iter().map(|x| E.powf(*x - m) / divisor).collect()
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

