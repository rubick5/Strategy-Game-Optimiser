use crate::battle::{command::{Command, MoveCommand}, state::{BattleState, PositionId, RosterId, TEAM_SIZE, Team}};

pub const MAX_DECISION: usize = MOVESLOT_COUNT + TEAM_SIZE;
pub const MOVESLOT_COUNT: usize = 4;

#[derive(Debug, Copy, Clone)]
pub enum Moveslot {
	Slot(usize),
	Switch(usize)
}
use Moveslot::*;

impl Moveslot {

	pub fn all_moveslots() -> Vec<Self> {
		(0..MAX_DECISION).map(|n| Self::from_number(n)).collect()
	}
	/**
	 * Creates a moveslot from a number

	 * Great for using indexes of probability vectors
	 */
	pub fn from_number(n: usize) -> Self {
		match n {
			n if n < MOVESLOT_COUNT => Slot(n),
			n if n < MAX_DECISION => Switch(n - 4),
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
	pub fn to_command(&self, user: PositionId, battle_state: &BattleState) -> Command {
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
		let team = user.team();

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