use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, Team}}, model::registry::Registry, rl::{encoder, neural_net::NeuralNet}};
use rand::{Rng, distr::{self, Uniform}, random};

use std::f32::consts::E;


#[derive(Debug)]
pub enum Moveslot {
	Slot(usize),
	Switch(usize),
}

impl Moveslot {
	pub fn from_number(n: usize) -> Self {
		if n < 4 {
			Self::Slot(n)
		} else {
			Self::Switch(n - 4)
		}
	}
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
pub struct Agent
{
	// a bunch of weights telling us what to do
	pub net: NeuralNet<fn(f32) -> f32>,
}

impl Agent
{
	pub fn choose_move(&self, representation: &[f32]) -> Moveslot {
		self.net.forward_and_choose(representation)
	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, 4];
		let relu = |x: f32| if x < 0.0 { 0.0 } else { x };
		Agent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu)
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
			Moveslot::Slot(_) => panic!("invalid moveslot"),
		}
	}

	pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
		let mut outputs = inputs.to_owned();
		for layer in self.neuron_layers.iter() {
			outputs = layer.forward(&outputs);			
		}
		softmax(&outputs)
	}
}

