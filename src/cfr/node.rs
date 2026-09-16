//! Reading a decision point out of the engine's `StepRequest`.
//!
//! The engine asks for commands in two different situations, and they are not
//! the same decision:
//!
//! * [`StepRequest::NeedsActions`] — every active creature picks a move or a
//!   switch, simultaneously.
//! * [`StepRequest::NeedsReplacements`] — one *or both* sides send something in
//!   after a faint.
//!
//! Both come out of here as a list of actors, which collapses the awkward case:
//! a replacement that only one side owes is a single-player decision, and a list
//! of one actor is exactly that, with no branch anywhere in the solver. It is
//! worth being explicit that this is deliberate, because treating a one-sided
//! replacement as if both players were choosing would invent a decision for the
//! player who is not making one, and quietly corrupt their regrets.

use crate::battle::command::Command;
use crate::battle::engine::engine::StepRequest;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::{Outcome, Team};
use crate::model::registry::Registry;
use crate::rl::mask::Mask;
use crate::rl::moveslot::Moveslot;

/// One creature that owes a command.
#[derive(Clone, Copy)]
pub struct Actor {
	pub team: Team,
	pub position: PositionId,
	pub mask: Mask,
}

impl Actor {
	/// Turn a chosen action index into something the engine accepts.
	pub fn command(&self, action: usize, state: &BattleState, registry: &Registry) -> Command {
		Moveslot::from_number(action).to_command(self.position, state, registry)
	}
}

/// What is happening at one point in the battle.
pub enum DecisionNode {
	/// Nobody decides anything — the battle is over.
	Terminal(Outcome),
	/// One or more creatures owe a command. Never empty.
	Decision { actors: Vec<Actor> },
}

impl DecisionNode {
	pub fn from(state: &BattleState, request: &StepRequest) -> Self {
		match request {
			StepRequest::Finished(outcome) => DecisionNode::Terminal(*outcome),

			StepRequest::NeedsActions => DecisionNode::Decision {
				actors: state
					.field
					.all_field_positions()
					.into_iter()
					.map(|position| actor_at(position, state))
					.collect(),
			},

			// Sorted so the command order handed to the engine does not depend on
			// the order faints happened to be collected in.
			StepRequest::NeedsReplacements(positions) => {
				let mut positions = positions.clone();
				positions.sort_by_key(|position| position.0);
				DecisionNode::Decision {
					actors: positions
						.into_iter()
						.map(|position| actor_at(position, state))
						.collect(),
				}
			}
		}
	}
}

fn actor_at(position: PositionId, state: &BattleState) -> Actor {
	let team = position.team();
	Actor {
		team,
		position,
		mask: Mask::from_battle_state(&team, position, state),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::known_answer_duel;

	#[test]
	fn an_ordinary_turn_asks_both_sides_at_once() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);

		match DecisionNode::from(&state, &StepRequest::NeedsActions) {
			DecisionNode::Decision { actors } => {
				assert_eq!(actors.len(), 2, "a 1v1 turn is a two-player decision");
				assert_eq!(actors[0].team, Team::Zero);
				assert_eq!(actors[1].team, Team::One);
			}
			DecisionNode::Terminal(_) => panic!("the battle has not started"),
		}
	}

	/// The case that would be easy to get wrong: only one side owes a
	/// replacement, so only one player is deciding.
	#[test]
	fn a_one_sided_replacement_is_a_single_player_decision() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);
		let request = StepRequest::NeedsReplacements(vec![PositionId(1)]);

		match DecisionNode::from(&state, &request) {
			DecisionNode::Decision { actors } => {
				assert_eq!(actors.len(), 1);
				assert_eq!(actors[0].team, Team::One);
			}
			DecisionNode::Terminal(_) => panic!("expected a decision"),
		}
	}

	#[test]
	fn a_finished_battle_is_terminal() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);
		let request = StepRequest::Finished(Outcome::Win { team: Team::Zero });

		match DecisionNode::from(&state, &request) {
			DecisionNode::Terminal(Outcome::Win { team }) => assert_eq!(team, Team::Zero),
			_ => panic!("expected a terminal node"),
		}
	}

	#[test]
	fn actions_map_to_the_commands_the_engine_expects() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);
		let actor = actor_at(PositionId(0), &state);

		match actor.command(0, &state, &registry) {
			Command::MoveAction(move_command) => {
				assert_eq!(move_command.user, PositionId(0));
				assert_eq!(move_command.targets, vec![PositionId(1)]);
			}
			Command::Switch { .. } => panic!("slot 0 is a move, not a switch"),
		}
	}
}
