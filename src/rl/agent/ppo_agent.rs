use crate::{battle::state::battle_state::BattleState, rl::{agent::Agent, battle_playout::PlayedBattle, mask::Mask, moveslot::Moveslot, nn::neural_net::NeuralNet}};
use crate::rl::encoder::TOTAL_ENCODING_LEN;
use crate::rl::moveslot::MAX_DECISION;
use rand::RngCore;

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

	pub fn use_batch(batch: Vec<PlayedBattle>, start_state: &BattleState) {
		for b in batch {
			let state = start_state;
			let actions: Vec<Moveslot> = b.steps.iter().map(|s| s.move_chosen).collect();
			let log_probabilities: Vec<f32> = b.steps.iter()
				.map(|s| s.probabilities[s.move_chosen.to_number()].ln()).collect();

			let rewards_to_go = calc_reward_to_go(&b);

		}
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
	// still need to add the win/loss bonus!!!
	returns
}