use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, Team}}, model::registry::Registry, rl::encoder};
use rand::{Rng, distr::{self, Uniform}, random};

use std::f32::consts::E;

const LEARNING_RATE: f32 = 0.01;

#[derive(Debug)]
pub enum Moveslot {
	Slot(usize),
	Switch(usize),
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
					move_id: battle_state.get_mon(user).unwrap().moves[*index]
				})
			},
			Moveslot::Switch(n) => todo!()
		}
	}
}

#[derive(Debug, Clone)]
pub struct Agent {
	// a bunch of weights telling us what to do
	pub neuron1: Neuron,
	pub neuron2: Neuron,
}

impl Agent {
	pub fn choose_move(&self, representation: &[f32], registry: &Registry) -> Moveslot {
		let move1choice = self.neuron1.forward(&representation);
		let move2choice = self.neuron2.forward(&representation);

		return Moveslot::Slot(softmax_then_select(vec![move1choice, move2choice]));
	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		Agent {
			neuron1: Neuron::gen_random(weight_count, rng),
			neuron2: Neuron::gen_random(weight_count, rng),
		}
	}

	pub fn backprop(&mut self, move_slot: Moveslot, encoding: Vec<f32>, battle_reward: f32, gt: f32) {
		let v = self.forward(&encoding);
		let gt = gt * battle_reward;
		match move_slot {
			Moveslot::Slot(0) => {
				self.neuron1.backprop(gt, &encoding, true, v[0]);
				self.neuron2.backprop(gt, &encoding, false, v[1]);
			},
			Moveslot::Slot(1) => {
				self.neuron2.backprop(gt, &encoding, true, v[1]);
				self.neuron1.backprop(gt, &encoding, false, v[0]);
			}
			Moveslot::Switch(n) => todo!(),
			Moveslot::Slot(_) => assert!(0 == 1),
		}
	}

	pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
		let n1 = self.neuron1.forward(&inputs);
		let n2 = self.neuron2.forward(&inputs);
		softmax(vec![n1, n2])
	}
}

pub struct NeuronLayer {
	pub neurons: Vec<Neuron>
}

impl NeuronLayer {
	pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
		self.neurons.iter().map(|neuron| neuron.forward(&input)).collect()
	}

	fn backward(&mut self, incoming_error: &[f32], layer_input: &[f32]) -> Vec<f32> {
		vec![0.0]
	}

}

#[derive(Debug, Clone)]
pub struct Neuron {
	pub weights: Vec<f32>,
	pub bias: f32,
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

	pub fn backprop(&mut self, gt: f32, inputs: &[f32], our_action_taken: bool, p_ours: f32) {
		assert!(inputs.len() == self.weights.len());

		// we need to backprop on each weight in the neuron
		let indicator = if our_action_taken { 1.0 } else { 0.0 };

		for index in 0..self.weights.len() {
			self.weights[index] += LEARNING_RATE * gt * inputs[index] * (indicator - p_ours);
		}
		self.bias += LEARNING_RATE * gt * (indicator - p_ours);
	}

}

fn softmax(logits: Vec<f32>) -> Vec<f32> {
	let divisor: f32 = logits.iter().map(|x| E.powf(*x)).sum();
	logits.iter().map(|x| E.powf(*x) / divisor).collect()
}

fn softmax_then_select(logits: Vec<f32>) -> usize {
	let probabilities = softmax(logits);
	let random_selection = random();
	let mut counter = 0.0;
	for (index, prob) in probabilities.into_iter().enumerate() {
		counter += prob;
		if counter >= random_selection {
			return index;
		}
	}
	0
}