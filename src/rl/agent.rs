use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, RosterId, TEAM_SIZE, Team}}, rl::{mask::Mask, neural_net::NeuralNet}};
use rand::Rng;

use std::f32::consts::E;

// max moveslot discriminant
pub const MAX_DECISION: usize = MOVESLOT_COUNT + TEAM_SIZE;

pub const MOVESLOT_COUNT: usize = 4;

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

fn softmax(logits: &[f32]) -> Vec<f32> {
	let divisor: f32 = logits.iter().map(|x| E.powf(*x)).sum();
	logits.iter().map(|x| E.powf(*x) / divisor).collect()
}

fn softmax_then_select(logits: &[f32]) -> usize {
	let probabilities = softmax(logits);
	let random_selection = rand::random();
	let mut counter = 0.0;
	for (index, prob) in probabilities.into_iter().enumerate() {
		counter += prob;
		if counter >= random_selection {
			return index;
		}
	}
	0
}

#[derive(Debug, Clone)]
pub struct Agent
{
	// a bunch of weights telling us what to do
	pub net: NeuralNet<fn(f32) -> f32>,
}

impl Agent
{
	pub fn choose_move(&mut self, representation: &[f32], mask: Mask) -> Moveslot {
		let mut logits = self.net.forward(representation);
		mask.apply(&mut logits); // edits them in place, remember

		Moveslot::from_number(softmax_then_select(&logits))

	}

	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, 4];
		let relu = |x: f32| if x < 0.0 { 0.0 } else { x };
		let relu_prime = |x: f32| if x < 0.0 { 0.0 } else { 1.0 };
		Agent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu, relu_prime)
		}
	}

	pub fn backprop(&mut self, move_slot: Moveslot, battle_reward: f32, gt: f32, encoding: &[f32], last_probabilities: &[f32]) {
		//let encoding = encoder::encode(battle_state, registry);
		//let mask = Mask::from_battle_state(team, pos, battle_state);

		let move_chosen_index = move_slot.to_number();
		let current_errors: Vec<f32> = (0..last_probabilities.len()).map ( |index| {
			let indicator = if index == move_chosen_index { 1.0 } else { 0.0 };
			battle_reward * gt * (last_probabilities[index] - indicator)
		}).collect();

		self.net.backward(current_errors, &encoding)
	}
}

