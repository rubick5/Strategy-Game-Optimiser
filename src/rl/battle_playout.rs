use rand::Rng;

use crate::battle::state::Team;
use crate::rl::train::EXPLORATION_CHANCE;
use crate::{
	battle::{
		command::Command,
		engine::{self, StepRequest, StepResult},
		state::{BattleState, Outcome, PositionId},
	},
	model::registry::Registry,
	rl::{
		agent::{BotAgent, Moveslot},
		encoder,
		mask::Mask,
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
 * Returns the battle reward for playing out the battle, plus the actions and states that took place
 */
pub fn play_out_battle(
	mut battle: BattleState,
	registry: &Registry,
	agent: &mut BotAgent,
	opponent: &mut BotAgent,
	rng: &mut impl Rng,
) -> PlayedBattle {
	let mut actions_and_states: Vec<Step> = Vec::new();
	let mut turn_count = 0;
	let mut step_request = StepRequest::NeedsActions;
	while turn_count < 1000 {
		turn_count += 1;

		match step_request {
			StepRequest::NeedsActions => {
				let agent_mask = Mask::from_battle_state(Team::Zero, PositionId(0), &battle);
				let opponent_mask = Mask::from_battle_state(Team::One, PositionId(1), &battle);

				let encoding = encoder::encode(&battle, registry, false);
				let (mut agent_moveslot, probabilities) =
					agent.choose_move(&encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let (opponent_moveslot, _) = opponent.choose_move(&encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(PositionId(0), &battle),
					opponent_moveslot.to_command(PositionId(1), &battle),
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
				let commands: Vec<Command> = positions
					.iter()
					.map(|pos| {
						let encoding = encoder::encode(&battle, registry, true);
						let mask = Mask::from_battle_state(pos.team(), *pos, &battle);
						let replacement = agent.choose_move(&encoding, &mask, rng);
						replacement.0.to_command(*pos, &battle)
					})
					.collect();
				StepResult {
					battle_state: battle,
					step_request,
				} = engine::step(battle, commands, registry, rng);
			}
			engine::StepRequest::Finished(Outcome::Side0Wins) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: 1.0 };
			}
			engine::StepRequest::Finished(Outcome::Side1Wins) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: -1.0 };
			}
			engine::StepRequest::Finished(Outcome::Draw) => {
				return PlayedBattle { steps: actions_and_states, battle_reward: 0.0 };
			}
		}
	}
	PlayedBattle { steps: vec![], battle_reward: 0.0 }
}
