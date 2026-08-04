/// WE NEED TO ADD SOME ENCOURAGEMENT FOR ENTROPY INTO THE CODE
/// SO THAT IT DOESNT JUST CONVERGE LIKE CRAZY
// * Technique 1: x% of the time just pick a random action instead of the model's one
// * Technique 2: use a entropy reward in the output layer's error to encourage the model to stay
// * versatile. Also note that we can reduce the weight of this as we get further into training
// * once the correct strategies have actually been figured out.

use crate::{battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, RosterId, TEAM_SIZE, Team}}, rl::{mask::Mask, neural_net::NeuralNet}};
use rand::Rng;

use std::f32::consts::E;

// max moveslot discriminant
pub const MAX_DECISION: usize = MOVESLOT_COUNT + TEAM_SIZE;

pub const RELU_LEAK: f32 = 0.01;

pub const MOVESLOT_COUNT: usize = 4;

pub const BASELINE_LEARNING_RATE: f32 = 0.01;

#[derive(Debug)]
pub enum Moveslot {
	Slot(usize),
	Switch(usize)
}
use Moveslot::*;

impl Moveslot {
	/**
	 * Creates a moveslot from a number

	 * Great for using indexes of probability vectors
	 */
	pub fn from_number(n: usize) -> Self {
		match n {
			n if n < 4 => Slot(n),
			n if n <= 9 => Switch(n - 4),
			_ => panic!("INVALID MOVESLOT SELECTED")
		}
	}

	/**
	 * Translates a moveslot back into a number

	 * Inverse of from_number function
	 */
	pub fn to_number(&self) -> usize {
		match self {
			Switch(n) => n + 4,
			Slot(n) => *n,
		}
	}

	/**
	 * Uses the context provided by the battlestate and who is using the move to translate
	 * itself (a moveslot) into an engine-approved command.

	 * This will need significant changes later as we add different types of moves
	 */
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
				let mon = battle_state.get_mon(user).unwrap();
				if *n > 1 {
					println!("n: {}", n);
					println!("mon: {:?}", mon);
				}

				let move_id = mon.moves[*n];
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

/**
 * Computes the softmax of the logits given
 * 
 * Currently panics on empty logits
 */
fn softmax(logits: &[f32]) -> Vec<f32> {
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
fn softmax_then_select(logits: &[f32], rng: &mut impl Rng) -> usize {
	let probabilities = softmax(logits);
	let random_selection = rng.random();
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
	pub net: NeuralNet<fn(f32) -> f32>,
	pub baseline: f32,
}

impl Agent
{
	/**
	 * Runs the encoding through the network then applies the mask to the logits.
	 *
	 * Returns a pair (x, y) where x is a randomly selected move from the probabilities
	 * and y is the calculated probabilities.
	 */
	pub fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut impl Rng) -> (Moveslot, Vec<f32>) {
		let mut logits = self.net.forward(representation);
		mask.apply(&mut logits); // edits them in place, remember

		(Moveslot::from_number(softmax_then_select(&logits, rng)), softmax(&logits))

	}

	/**
	 * Initialises the agent neural network with random weights everywhere.

	 * Size and number of hidden layers are defined in a magic number vector here, at some point
	 * that should be moved out...
	 */
	pub fn init_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let layer_sizes = vec![weight_count, 64, 128, MAX_DECISION];
		let relu = |x: f32| if x < 0.0 { RELU_LEAK * x } else { x };
		let relu_prime = |x: f32| if x < 0.0 { RELU_LEAK } else { 1.0 };
		Agent {
			net: NeuralNet::gen_random(rng, &layer_sizes, relu, relu_prime),
			baseline: 0.0,
		}
	}

	/**
	 * Learns from previous mistakes or successes. Needs a bunch of weird arguments for the maths to work out.
	 */
	pub fn backprop(&mut self, move_slot: Moveslot, battle_reward: f32, gt: f32, encoding: &[f32], last_probabilities: &[f32]) {
		//let encoding = encoder::encode(battle_state, registry);
		//let mask = Mask::from_battle_state(team, pos, battle_state);

		let move_chosen_index = move_slot.to_number();
		let current_errors: Vec<f32> = (0..last_probabilities.len()).map ( |index| {
			let indicator = if index == move_chosen_index { 1.0 } else { 0.0 };
			(battle_reward - self.baseline) * gt * (last_probabilities[index] - indicator)
		}).collect();

		self.net.backward(current_errors, &encoding)
	}
}

