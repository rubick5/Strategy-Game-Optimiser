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

/// Play a battle between two fixed policies, recording nothing.
///
/// [`play_out_battle_as`] exists to produce training trajectories, so it carries
/// all the machinery for that. This one is for *measurement*, where both
/// policies are frozen and the only thing wanted is the outcome — so it takes
/// two plain `dyn Agent`s rather than requiring a `LearningAgent`, which lets a
/// random or spam baseline sit in either seat.
///
/// Returns the result from `first`'s point of view, plus the turn count.
pub fn play_headless_as(
	mut battle: BattleState,
	registry: &Registry,
	first: &mut dyn Agent,
	second: &mut dyn Agent,
	first_team: Team,
	rng: &mut dyn RngCore,
) -> (BattleEnd, usize) {
	let second_team = first_team.other();
	let first_pos = battle.field.team_positions(&first_team)[0];
	let second_pos = battle.field.team_positions(&second_team)[0];

	let mut turn_count = 0;
	let mut step_request = StepRequest::NeedsActions;
	while turn_count < MAX_TURNS {
		turn_count += 1;
		match step_request {
			StepRequest::NeedsActions => {
				let first_mask = Mask::from_battle_state(&first_team, first_pos, &battle);
				let second_mask = Mask::from_battle_state(&second_team, second_pos, &battle);
				let (zero_view, one_view) = encoder::encode_both(&battle, registry, false);
				let (first_view, second_view) = match first_team {
					Team::Zero => (zero_view, one_view),
					Team::One => (one_view, zero_view),
				};

				let actions = vec![
					first.choose_move(&first_view, &first_mask, rng).to_command(first_pos, &battle, registry),
					second.choose_move(&second_view, &second_mask, rng).to_command(second_pos, &battle, registry),
				];
				StepResult { battle_state: battle, step_request } =
					engine::step(battle, actions, registry, rng);
			}
			StepRequest::NeedsReplacements(positions) => {
				let mut commands: Vec<Command> = Vec::new();
				for pos in positions {
					let team = pos.team();
					let encoding = encoder::encode(&battle, registry, true, &team);
					let mask = Mask::from_battle_state(&team, pos, &battle);
					let actor: &mut dyn Agent = if team == first_team { first } else { second };
					commands.push(actor.choose_move(&encoding, &mask, rng).to_command(pos, &battle, registry));
				}
				StepResult { battle_state: battle, step_request } =
					engine::step(battle, commands, registry, rng);
			}
			StepRequest::Finished(Outcome::Win { team }) => {
				let won = team == first_team;
				return (if won { BattleEnd::Win } else { BattleEnd::Loss }, turn_count);
			}
			StepRequest::Finished(Outcome::Draw) => return (BattleEnd::Draw, turn_count),
		}
	}
	(BattleEnd::Timeout, turn_count)
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
 * The learner takes Team Zero here; use [`play_out_battle_as`] to put it on the
 * other side. Each player receives the board encoded from its *own* side.
 * Before that, both read a Team-Zero-first encoding, so the opponent was
 * evaluating the learner's position rather than its own — which quietly
 * corrupted every self-play game.
 */
pub fn play_out_battle(
	battle: BattleState,
	registry: &Registry,
	agent: &mut impl LearningAgent,
	opponent: &mut dyn Agent,
	rng: &mut dyn RngCore,
) -> PlayedBattle {
	play_out_battle_as(battle, registry, agent, opponent, rng, Team::Zero)
}

/// As [`play_out_battle`], but the learner takes `learner_team`.
///
/// Worth having because with deterministic damage a fixed roster pairing is a
/// foregone conclusion — one side simply wins. Measuring or training on one side
/// only therefore measures the match-up, not the policy. Playing both sides
/// cancels that out exactly.
pub fn play_out_battle_as(
	mut battle: BattleState,
	registry: &Registry,
	agent: &mut impl LearningAgent,
	opponent: &mut dyn Agent,
	rng: &mut dyn RngCore,
	learner_team: Team,
) -> PlayedBattle {
	let opponent_team = learner_team.other();
	let learner_pos = battle.field.team_positions(&learner_team)[0];
	let opponent_pos = battle.field.team_positions(&opponent_team)[0];

	let mut actions_and_states: Vec<Step> = Vec::new();
	let mut turn_count = 0;
	let mut step_request = StepRequest::NeedsActions;
	while turn_count < MAX_TURNS {
		turn_count += 1;

		match step_request {
			StepRequest::NeedsActions => {
				let agent_mask = Mask::from_battle_state(&learner_team, learner_pos, &battle);
				let opponent_mask = Mask::from_battle_state(&opponent_team, opponent_pos, &battle);

				// Both views in one pass: the per-creature blocks are the same for
				// either side, so this costs barely more than a single encoding.
				let (zero_view, one_view) = encoder::encode_both(&battle, registry, false);
				let (agent_encoding, opponent_encoding) = match learner_team {
					Team::Zero => (zero_view, one_view),
					Team::One => (one_view, zero_view),
				};

				let (mut agent_moveslot, probabilities) =
					agent.choose_move_with_probs(&agent_encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let opponent_moveslot = opponent.choose_move(&opponent_encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(learner_pos, &battle, registry),
					opponent_moveslot.to_command(opponent_pos, &battle, registry),
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
				let agent_replacement_pos: Vec<PositionId> = positions.iter().filter(|p| p.team() == learner_team).map(|p| *p).collect();
				let opponent_replacement_pos: Vec<PositionId> = positions.iter().filter(|p| p.team() == opponent_team).map(|p| *p).collect();
				let (mut agent_steps, agent_commands) = learner_team_replacements(agent, &agent_replacement_pos, &battle, registry, rng);
				let opponent_replacements = opponent_team_replacements(opponent, &opponent_replacement_pos, &battle, registry, rng);

				actions_and_states.append(&mut agent_steps);

				let all_commands = vec![agent_commands, opponent_replacements].concat();


				StepResult {
					battle_state: battle,
					step_request,
				} = engine::step(battle, all_commands, registry, rng);
			}
			engine::StepRequest::Finished(Outcome::Win { team }) => {
				let won = team == learner_team;
				return PlayedBattle {
					steps: actions_and_states,
					battle_reward: if won { 1.0 } else { -1.0 },
					outcome: if won { BattleEnd::Win } else { BattleEnd::Loss },
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
