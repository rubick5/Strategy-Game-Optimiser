use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, Team}}, model::registry::Registry, rl::encoder};
use rand::{Rng, distr::{self, Uniform}};

use std::f32::consts::E;

pub enum Moveslot {
	Slot(usize),
	Switch,
}

impl Moveslot {
	pub fn to_command(&self, team: Team, user: PositionId, battle_state: &BattleState) -> Command {
		let (_, target) = match team {
			Team::One  => ( 1, 0 ),
			Team::Zero => ( 0, 1 ),
		};
		match self {
			Moveslot::Slot(index) => {
				Command::MoveAction(MoveCommand {
					user,
					targets: vec![PositionId(target)],
					move_id: battle_state.mons.get(user).unwrap().moves[*index]
				})
			},
			Moveslot::Switch => todo!()
		}
	}
}

pub struct Agent {
	// a bunch of weights telling us what to do
	pub neuron1: Neuron,
	pub neuron2: Neuron,
}

impl Agent {
	pub fn choose_move(&self, representation: &[f32], registry: &Registry) -> Moveslot {
		let move1choice = self.neuron1.forward(&representation);
		let move2choice = self.neuron2.forward(&representation);
		if move1choice > move2choice {
			return Moveslot::Slot(0);
		} else {
			return Moveslot::Slot(1);
		}
	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		Agent {
			neuron1: Neuron::gen_random(weight_count, rng),
			neuron2: Neuron::gen_random(weight_count, rng),
		}
	}

	pub fn backprop(&self, move_slot: Moveslot, state: BattleState, battle_reward: f32) {
		todo!()
	}
}


struct Neuron {
	weights: Vec<f32>,
	bias: f32,
}

impl Neuron {
	pub fn forward(&self, inputs: &[f32]) -> f32 {
		if inputs.len() != self.weights.len() {
			panic!("bad neural net setup: inputs; {} weights: {}", inputs.len(), self.weights.len());
			
		}
		let mut result = self.bias;
		for i in 0..inputs.len() {
			result += inputs[i] * self.weights[i];
		}
		return result;
	}

	pub fn gen_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let distribution = Uniform::new(-1.0, 1.0).unwrap();
		let weights = rng.sample_iter(distribution).take(weight_count).collect();
		let bias = 0.0; //rng.random_range(-2.0..2.0);

		Neuron {
			weights,
			bias
		}
	}

	pub fn backprop(&mut self, gt: f32, ) {
		// we need to backprop on each weight in the neuron
		for weight in self.weights.iter() {

		}
	}

}

fn softmax(logits: Vec<f32>) -> Vec<f32> {
	let divisor: f32 = logits.iter().map(|x| E.powf(*x)).sum();
	logits.iter().map(|x| E.powf(*x) / divisor).collect()
}