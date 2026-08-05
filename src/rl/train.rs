use rand::Rng;

use crate::{battle::{command::Command, engine::{self, StepRequest, StepResult}, state::{BattleState, Outcome, PokemonState, PositionId}}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::{BotAgent, BASELINE_LEARNING_RATE, Moveslot}, encoder, env, mask::Mask}};
use crate::battle::state::Team;

const DAMPING_CONSTANT: f32 = 0.95;
const EXPLORATION_CHANCE: f32 = 0.05;

// note that the total number of battles used for training
// will be BATCH_COUNT * BATCH_SIZE
const BATCH_COUNT: usize = 3000;
const BATCH_SIZE: usize = 32;

const BATCH_PRINT_FREQ: usize = 50;
const BATCH_PRINT_GAP_SIZE: usize = BATCH_SIZE * BATCH_PRINT_FREQ;

/* We want to:
	* Model a strategy as a net
	* Test the strategy against a random strategy
	* See how well our strategy does against the random one
	* Backpropogate our loss
	(We need to figure out what loss means)
*/

pub fn start_battle_state(registry: &Registry) -> BattleState {
	let ps0 = PokemonState::from_species(registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let ps01 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);
	let ps11 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps1_1hp = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	ps1_1hp.current_hp = 1;
	BattleState::from(vec![ps0, ps01], vec![ps1, ps11], vec![0, 1])
}

pub fn battle_state_1hp(registry: &Registry) -> BattleState {
	let ps0 = PokemonState::from_species(registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let ps01 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);
	let ps11 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps1_1hp = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	ps1_1hp.current_hp = 1;
	BattleState::from(vec![ps0, ps01], vec![ps1, ps11], vec![0, 1])
}

struct Step {
	encoding: Vec<f32>,
	move_chosen: Moveslot,
	probabilities: Vec<f32>,
}

pub fn main_loop(mut rng: &mut impl Rng) {
	let registry = Registry::load();
	let mut agent: BotAgent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let mut battles_won = 0;


	let mut best_agent: BotAgent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, rng);
	let mut max_battles_won: i32 = 0;
	let mut opponent: BotAgent;
	let mut current_batch: Vec<(f32, Vec<Step>)> = Vec::new();
	for batch_num in 0..BATCH_COUNT {
		opponent = agent.clone();
		for _ in 0..BATCH_SIZE {
			//opponent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, rng);
			let battle: BattleState = start_battle_state(&registry);

			let (battle_reward, actions_and_states) = play_out_battle(battle, &registry, &mut agent, &mut opponent, rng);
			
			if battle_reward == 1.0 {
				battles_won += 1;
			}
			current_batch.push((battle_reward, actions_and_states));
		}
		if batch_num % BATCH_PRINT_FREQ == 0 {
			println!("batch {}: battles won: {} out of {}", batch_num, battles_won, BATCH_PRINT_GAP_SIZE);
			if battles_won > max_battles_won {
				best_agent = agent.clone();
				max_battles_won = battles_won;
			}
			battles_won = 0;
		}
		while let Some((battle_reward, steps)) = current_batch.pop() {
			learn_from_battle(&mut agent, battle_reward, steps);
		}
	}
	println!("final state of agent:");
	final_agent_checks(agent, &registry, rng);
	println!("highest winrate agent:");
	final_agent_checks(best_agent, &registry, rng);
}

fn final_agent_checks(mut agent: BotAgent, registry: &Registry, rng: &mut impl Rng) {
	let agent_mask_normal = Mask::from_battle_state(Team::Zero, PositionId(0), &start_battle_state(registry));
	let agent_mask_1hp = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_1hp(registry));
	
	println!("agent normal: {:?}", agent.choose_move(&encoder::encode(&start_battle_state(registry), &registry, false), &agent_mask_normal, rng));
	println!("agent 1hp: {:?}", agent.choose_move(&encoder::encode(&battle_state_1hp(registry), &registry, false), &agent_mask_1hp, rng));

}


/**
 * Returns the battle reward for playing out the battle, plus the actions and states that took place
 */
fn play_out_battle(mut battle: BattleState, registry: &Registry, agent: &mut BotAgent, opponent: &mut BotAgent, rng: &mut impl Rng) -> (f32, Vec<Step>) {
	
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
				let (mut agent_moveslot, probabilities) = agent.choose_move(&encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let (opponent_moveslot, _) = opponent.choose_move(&encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(PositionId(0), &battle),
					opponent_moveslot.to_command(PositionId(1), &battle)
				];
				actions_and_states.push(Step {
					encoding,
					move_chosen: agent_moveslot,
					probabilities,
				});
				StepResult { battle_state: battle, step_request } = engine::step(battle, actions, &registry, rng);
			},
			StepRequest::NeedsReplacements(positions) => {
						// need to get the agent to tell us who to swap to
						let commands: Vec<Command> = positions.iter().map (|pos| {
							let encoding = encoder::encode(&battle, registry, true);
							let mask = Mask::from_battle_state(pos.team(), *pos, &battle);
							let replacement = agent.choose_move(&encoding, &mask, rng);
							replacement.0.to_command(*pos, &battle)
						}).collect();
						StepResult { battle_state: battle, step_request } = engine::step(battle, commands, registry, rng);
					},
			engine::StepRequest::Finished(Outcome::Side0Wins) => {
				return (1.0, actions_and_states);
			}
			engine::StepRequest::Finished(Outcome::Side1Wins) => {
				return (-1.0, actions_and_states);
			},
			engine::StepRequest::Finished(Outcome::Draw) => {
				return (0.0, actions_and_states);
			},
		}
	}
	(0.0, vec![])
}

fn learn_from_battle(agent: &mut BotAgent, battle_reward: f32, steps: Vec<Step>) {
	let mut gt = 1.0;
	agent.baseline += (battle_reward - agent.baseline) * BASELINE_LEARNING_RATE;
	for Step { encoding, move_chosen, probabilities } in steps {
			// don't forget to use 'reward' in here somewhere
		agent.backprop(move_chosen, battle_reward, gt, &encoding, &probabilities);
		gt = gt * DAMPING_CONSTANT;
	}
}