pub mod train_config;
pub mod bot_agent;
pub mod random_agent;
pub mod spam_agent;
pub mod ppo_agent;

use std::{error::Error, f32::consts::E};

use crate::battle::engine::engine::StepRequest;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::Team;
use crate::rl::{agent::train_config::TrainConfig, battle_playout::PlayedBattle, mask::Mask, moveslot::Moveslot};
use rand::{Rng as _, RngCore};

pub const RELU_LEAK: f32 = 0.01;
pub const EPS: f32 = 0.05;


pub trait Agent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot;

	/// The position the next `choose_move` will be about, before it is encoded.
	///
	/// A network needs only the encoding, which is why that is all `choose_move`
	/// gets. A *search* needs the position itself — it cannot look ahead from a
	/// vector of floats. Without this an agent that thinks by searching cannot be
	/// played against the learners here at all.
	///
	/// Does nothing by default, so every existing agent is unaffected.
	fn observe(&mut self, _state: &BattleState, _request: &StepRequest, _team: Team) {}
}

impl <A: Agent + ?Sized> Agent for Box<A> {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		(**self).choose_move(representation, mask, rng)
	}

	fn observe(&mut self, state: &BattleState, request: &StepRequest, team: Team) {
		(**self).observe(state, request, team)
	}
}

pub trait LearningAgent: Agent + Clone {
	fn learn_from_batch(&mut self, batch: &Vec<PlayedBattle>, gt: f32, train_config: &TrainConfig);

	fn move_probs(&mut self, representation: &[f32], mask: &Mask) -> Vec<f32>;
	
	fn choose_move_with_probs(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> (Moveslot, Vec<f32>);

	fn to_file(&self, file_name: &str) -> Result<(), Box<dyn Error>>;
}

pub fn mean_squared_error(guess: f32, values: &[f32]) -> f32 {
	values.iter().map(|v| (guess - v) * (guess - v)).sum()
}

pub fn normalise_floats(floats: &[f32]) -> Vec<f32> {
	match floats {
		[] => vec![],
		floats => {
			let mean = floats.iter().sum::<f32>() / floats.len() as f32;
			let std: f32 = floats.iter().map(|f| {
				(f - mean) * (f - mean)
			}).sum::<f32>() / floats.len() as f32;
			floats.into_iter().map(|f| (f - mean) / (std + EPS)).collect()
		}
	}
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
		if prob.is_nan() {
			panic!("NANANANAN ANANNANANANA ANAN");
		}
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

