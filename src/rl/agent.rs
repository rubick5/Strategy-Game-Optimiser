use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, RosterId, Team}}, model::registry::Registry, rl::{encoder, neural_net::NeuralNet}};
use rand::{Rng, distr::{self, Uniform}, random};

use std::{error::Error, f32::consts::E};

// max moveslot discriminant
const MAX_MOVESLOT: usize = 9;


#[derive(Debug)]
pub enum Moveslot {
	Slot(usize),
	Switch(usize)
}
use Moveslot::*;

impl Moveslot {
	pub fn from_number(n: usize) -> Self {
		match n {
			n if n < 4 => Slot(n),
			n if n < 9 => Slot(n - 4),
			_ => panic!("INVALID MOVESLOT SELECTED")
		}
	}
	pub fn to_number(&self) -> usize {
		match self {
			Switch(n) => n + 4,
			Slot(n) => *n,
		}
	}

	pub fn to_command(&self, team: Team, user: PositionId, battle_state: &BattleState) -> Command {
		// get the right pokemon
		// choose the right moveslot / switch
		// make and return the command
		// handle targetings
		// all moves target something
		
		// for now, if we are pos 0 we target 1 and if pos 1 we target 0:
		let target = match user {
			PositionId(0) => PositionId(1),
			PositionId(1) => PositionId(0),
			_ => panic!("someone tried to use a move, but they don't exist!")
		};
		
		match self {
			Switch(n) => {
				let new: RosterId = RosterId(
					n * 2 + (if team == Team::One { 1 } else { 0 })
				);
				Command::Switch {
					current: user,
					new,
				}
			},
			Slot(n) => {
				let move_id = battle_state.get_mon(user).unwrap().moves[*n];
				Command::MoveAction(
					MoveCommand {
						move_id,
						user,
						targets: vec![target],
					}
				)
			}
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
	pub fn choose_move(&mut self, representation: &[f32]) -> Moveslot {
		self.net.forward_and_choose(representation)
	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, 4];
		let relu = |x: f32| if x < 0.0 { 0.0 } else { x };
		let relu_prime = |x: f32| if x < 0.0 { 0.0 } else { 1.0 };
		Agent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu, relu_prime)
		}
	}

	pub fn backprop(&mut self, move_slot: Moveslot, battle_reward: f32, gt: f32, encoding: &[f32]) {
		self.net.backward(gt, battle_reward, move_slot, encoding)
	}
}

