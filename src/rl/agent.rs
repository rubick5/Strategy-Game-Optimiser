use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, Team}}, model::registry::Registry, rl::{encoder, neural_net::NeuralNet}};
use rand::{Rng, distr::{self, Uniform}, random};

use std::{error::Error, f32::consts::E};

// max moveslot discriminant
const MAX_MOVESLOT: usize = 9;


#[derive(Debug)]
pub enum Moveslot {
	Slot1,
	Slot2,
	Slot3,
	Slot4,
	Switch1,
	Switch2,
	Switch3,
	Switch4,
	Switch5,
	Switch6,
}
use Moveslot::*;

impl Moveslot {
	pub fn from_number(n: usize) -> Self {
		match n {
			0 => Slot1,
			1 => Slot2,
			2 => Slot3,
			3 => Slot4,
			4 => Switch1,
			5 => Switch2,
			6 => Switch3,
			7 => Switch4,
			8 => Switch5,
			9 => Switch6,
			_ => panic!("INVALID MOVESLOT SELECTED")
		}
	}
	pub fn to_command(&self, team: Team, user: PositionId, battle_state: &BattleState) -> Command {
		// get the right pokemon
		// choose the right moveslot / switch
		// make and return the command
		// handle targetings
		// all moves target something
		todo!()
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
	pub fn choose_move(&mut self, representation: &[f32]) -> Moveslot {
		self.net.forward_and_choose(representation)
	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, 4];
		let relu = |x: f32| if x < 0.0 { 0.0 } else { x };
		Agent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu)
		}
	}

	pub fn backprop(&mut self, move_slot: Moveslot, battle_reward: f32, gt: f32) {
		self.net.backward(gt, battle_reward, move_slot)
	}
}

