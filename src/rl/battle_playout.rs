use rand::{Rng, RngCore};

use crate::battle::state::Team;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::rl::agent::{Agent, LearningAgent, softmax};
use crate::rl::train::EXPLORATION_CHANCE;
use crate::{
	battle::{
		command::Command,
		engine::engine::{self, StepRequest, StepResult},
		state::Outcome,
	},
	model::registry::Registry,
	rl::{
		encoder,
		mask::Mask,
		moveslot::Moveslot,
	},
};

/// Hard stop on a single battle. A battle that hits this produced no usable
/// training signal, so it is reported separately rather than being quietly
/// folded in with the losses.
pub const MAX_TURNS: usize = 1000;

#[derive(Clone)]
pub struct Step {
	pub encoding: Vec<f32>,
	pub mask: Mask,
	pub move_chosen: Moveslot,
	pub probabilities: Vec<f32>,
}
impl Step {
	pub fn chosen_prob(&self) -> f32 {
		softmax(&self.probabilities)[self.move_chosen.to_number()]
	}
}

/// How a battle finished, from the learner's point of view.
///
/// `Timeout` used to be indistinguishable from a loss: the old code returned
/// `battle_reward: 0.0` with no steps, so it counted as a not-win *and*
/// contributed nothing to learning, with no way to see how often it happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BattleEnd {
	Win,
	Loss,
	Draw,
	Timeout,
}

#[derive(Clone)]
pub struct PlayedBattle {
	pub steps: Vec<Step>,
	pub battle_reward: f32,
	pub outcome: BattleEnd,
	/// Engine steps taken, including replacement requests.
	pub turns: usize,
}

/**
 * Makes the agent calculate replacements for all positions handed to the function
 *
 * Please only give positions aligned with the agent's team :)
 */
fn learner_team_replacements(agent: &mut impl LearningAgent, positions: &[PositionId], battle_state: &BattleState, registry: &Registry, rng: &mut dyn RngCore) -> (Vec<Step>, Vec<Command>) {
	positions
		.iter()
		.map(|pos| {
			// Encoded from the replacing side's own point of view.
			let encoding = encoder::encode(battle_state, registry, true, &pos.team());
			let mask = Mask::from_battle_state(&pos.team(), *pos, &battle_state);
			let (move_chosen, probabilities) = agent.choose_move_with_probs(&encoding, &mask, rng);

			let command = move_chosen.to_command(*pos, battle_state, registry);

			let agent_step =
				Step {
					encoding,
					move_chosen,
					probabilities,
					mask,
				};
			(agent_step, command)
		})
		.unzip()
}

/**
 * Makes the opponent calculate... we don't need the probabilities because we aren't learning
 */
fn opponent_team_replacements(agent: &mut dyn Agent, positions: &[PositionId], battle_state: &BattleState, registry: &Registry, rng: &mut dyn RngCore) -> Vec<Command> {
	positions
		.iter()
		.map(|pos| {
			let encoding = encoder::encode(battle_state, registry, true, &pos.team());
			let mask = Mask::from_battle_state(&pos.team(), *pos, battle_state);
			agent.choose_move(&encoding, &mask, rng).to_command(*pos, battle_state, registry)
		})
		.collect()
}

/**
 * Returns the battle reward for playing out the battle, plus the actions and states that took place
 *
 * The learner is Team Zero and the opponent Team One, and each now receives the
 * board encoded from its *own* side. Before this, both read a Team-Zero-first
 * encoding, so the opponent was evaluating the learner's position rather than its
 * own — which quietly corrupted every self-play game.
 */
pub fn play_out_battle(
	mut battle: BattleState,
	registry: &Registry,
	agent: &mut impl LearningAgent,
	opponent: &mut dyn Agent,
	rng: &mut dyn RngCore,
) -> PlayedBattle {
	let mut actions_and_states: Vec<Step> = Vec::new();
	let mut turn_count = 0;
	let mut step_request = StepRequest::NeedsActions;
	while turn_count < MAX_TURNS {
		turn_count += 1;

		match step_request {
			StepRequest::NeedsActions => {
				let agent_mask = Mask::from_battle_state(&Team::Zero, PositionId(0), &battle);
				let opponent_mask = Mask::from_battle_state(&Team::One, PositionId(1), &battle);

				// Both views in one pass: the per-creature blocks are the same for
				// either side, so this costs barely more than a single encoding.
				let (agent_encoding, opponent_encoding) =
					encoder::encode_both(&battle, registry, false);

				let (mut agent_moveslot, probabilities) =
					agent.choose_move_with_probs(&agent_encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let opponent_moveslot = opponent.choose_move(&opponent_encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(PositionId(0), &battle, registry),
					opponent_moveslot.to_command(PositionId(1), &battle, registry),
				];
				actions_and_states.push(Step {
					encoding: agent_encoding,
					move_chosen: agent_moveslot,
					probabilities,
					mask: agent_mask,
				});
				StepResult {
					battle_state: battle,
					step_request,
				} = engine::step(battle, actions, &registry, rng);
			}
			StepRequest::NeedsReplacements(positions) => {
				// need to get the agent to tell us who to swap to
				let agent_replacement_pos: Vec<PositionId> = positions.iter().filter(|PositionId(p)| p % 2 == 0).map(|p| *p).collect();
				let opponent_replacement_pos: Vec<PositionId> = positions.iter().filter(|PositionId(p)| p % 2 == 1).map(|p| *p).collect();
				let (mut agent_steps, agent_commands) = learner_team_replacements(agent, &agent_replacement_pos, &battle, registry, rng);
				let opponent_replacements = opponent_team_replacements(opponent, &opponent_replacement_pos, &battle, registry, rng);

				actions_and_states.append(&mut agent_steps);

				let all_commands = vec![agent_commands, opponent_replacements].concat();


				StepResult {
					battle_state: battle,
					step_request,
				} = engine::step(battle, all_commands, registry, rng);
			}
			engine::StepRequest::Finished(Outcome::Win { team: Team::Zero }) => {
				return PlayedBattle {
					steps: actions_and_states,
					battle_reward: 1.0,
					outcome: BattleEnd::Win,
					turns: turn_count,
				};
			}
			engine::StepRequest::Finished(Outcome::Win { team: Team::One }) => {
				return PlayedBattle {
					steps: actions_and_states,
					battle_reward: -1.0,
					outcome: BattleEnd::Loss,
					turns: turn_count,
				};
			}
			engine::StepRequest::Finished(Outcome::Draw) => {
				return PlayedBattle {
					steps: actions_and_states,
					battle_reward: -0.1,
					outcome: BattleEnd::Draw,
					turns: turn_count,
				};
			}
		}
	}
	PlayedBattle {
		steps: vec![],
		battle_reward: 0.0,
		outcome: BattleEnd::Timeout,
		turns: turn_count,
	}
}
