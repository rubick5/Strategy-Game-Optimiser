use rand::{Rng, RngCore};

use crate::battle::state::Team;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::rl::agent::Agent;
use crate::rl::agent::bot_agent::BotAgent;
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

pub struct Step {
	pub encoding: Vec<f32>,
	pub move_chosen: Moveslot,
	pub probabilities: Vec<f32>,
}

pub struct PlayedBattle {
	pub steps: Vec<Step>,
	pub battle_reward: f32,
}

/**
 * Makes the agent calculate replacements for all positions handed to the function
 * 
 * Please only give positions aligned with the agent's team :)
 */
fn learner_team_replacements(agent: &mut BotAgent, positions: &[PositionId], battle_state: &BattleState, registry: &Registry, rng: &mut dyn RngCore) -> Vec<(Moveslot, Command, Vec<f32>)> {
	positions
		.iter()
		.map(|pos| {
			let encoding = encoder::encode(battle_state, registry, true);
			let mask = Mask::from_battle_state(&pos.team(), *pos, &battle_state);
			let (replacement, probabilities) = agent.choose_move_with_probs(&encoding, &mask, rng);

			let command = replacement.to_command(*pos, battle_state, registry);
			
			(replacement, command, probabilities)
		})
		.collect()
}

/**
 * Makes the opponent calculate... we don't need the probabilities because we aren't learning
 */
fn opponent_team_replacements(agent: &mut dyn Agent, positions: &[PositionId], battle_state: &BattleState, registry: &Registry, rng: &mut dyn RngCore) -> Vec<Command> {
	positions
		.iter()
		.map(|pos| {
			let encoding = encoder::encode(&battle_state, registry, true);
			let mask = Mask::from_battle_state(&pos.team(), *pos, battle_state);
			agent.choose_move(&encoding, &mask, rng).to_command(*pos, battle_state, registry)
		})
		.collect()
}

/**
 * Returns the battle reward for playing out the battle, plus the actions and states that took place
 */
pub fn play_out_battle(
	mut battle: BattleState,
	registry: &Registry,
	agent: &mut BotAgent,
	opponent: &mut dyn Agent,
	rng: &mut dyn RngCore,
) -> PlayedBattle {
	let mut actions_and_states: Vec<Step> = Vec::new();
	let mut turn_count = 0;
	let mut step_request = StepRequest::NeedsActions;
	while turn_count < 1000 {
		turn_count += 1;

		match step_request {
			StepRequest::NeedsActions => {
				let agent_mask = Mask::from_battle_state(&Team::Zero, PositionId(0), &battle);
				let opponent_mask = Mask::from_battle_state(&Team::One, PositionId(1), &battle);

				let encoding = encoder::encode(&battle, registry, false);
				let (mut agent_moveslot, probabilities) =
					agent.choose_move_with_probs(&encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let opponent_moveslot = opponent.choose_move(&encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(PositionId(0), &battle, registry),
					opponent_moveslot.to_command(PositionId(1), &battle, registry),
				];
				actions_and_states.push(Step {
					encoding,
					move_chosen: agent_moveslot,
					probabilities,
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
				let learner_results = learner_team_replacements(agent, &agent_replacement_pos, &battle, registry, rng);
				let opponent_replacements = opponent_team_replacements(opponent, &opponent_replacement_pos, &battle, registry, rng);

				let mut agent_steps = learner_results.iter().map(|(moveslot, _, probabilities)| {
					Step {
						encoding: encoder::encode(&battle, registry, true),
						move_chosen: *moveslot,
						probabilities: probabilities.clone(),
					}
				}).collect();
				actions_and_states.append(&mut agent_steps);

				let agent_commands: Vec<Command> = learner_results.iter().map(|(_, c, _)| c.clone()).collect();

				let all_commands = vec![agent_commands, opponent_replacements].concat();


				StepResult {
					battle_state: battle,
					step_request,
				} = engine::step(battle, all_commands, registry, rng);
			}
			engine::StepRequest::Finished(Outcome::Win { team: Team::Zero }) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: 1.0 };
			}
			engine::StepRequest::Finished(Outcome::Win { team: Team::One }) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: -1.0 };
			}
			engine::StepRequest::Finished(Outcome::Draw) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: -0.1 };
			}
		}
	}
	PlayedBattle { steps: vec![], battle_reward: 0.0 }
}
