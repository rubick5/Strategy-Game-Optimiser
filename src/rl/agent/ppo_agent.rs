use crate::{battle::state::battle_state::BattleState, rl::{battle_playout::PlayedBattle, moveslot::Moveslot, nn::neural_net::NeuralNet}};
use crate::rl::encoder::TOTAL_ENCODING_LEN;
use crate::rl::moveslot::MAX_DECISION;
use rand::RngCore;

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

	pub fn use_batch(batch: Vec<PlayedBattle>, start_state: &BattleState) {
		for b in batch {
			let state = start_state;
			let actions: Vec<Moveslot> = b.steps.iter().map(|s| s.move_chosen).collect();
			let log_probabilities: Vec<f32> = b.steps.iter()
				.map(|s| s.probabilities[s.move_chosen.to_number()].ln()).collect();
			//let rewards: Vec<f32> = b.

		}
	}
}