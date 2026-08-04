use rand::Rng;

use crate::{battle::state::{BattleState, PokemonState, PositionId}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::{Agent, BASELINE_LEARNING_RATE, Moveslot}, encoder, env, mask::Mask}};
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

struct Step {
	encoding: Vec<f32>,
	move_chosen: Moveslot,
	probabilities: Vec<f32>,
}

pub fn main_loop(mut rng: &mut impl Rng) {
	let registry = Registry::load();
	let mut agent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let mut battles_won = 0;


	let ps0 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let ps01 = PokemonState::from_species(&registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);
	let ps11 = PokemonState::from_species(&registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps1_1hp = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	ps1_1hp.current_hp = 1;
	let battle_state_normal = BattleState::from(vec![ps0, ps01], vec![ps1, ps11], vec![0, 1]);
	let battle_state_1hp = BattleState::from(vec![ps00], vec![ps1_1hp], vec![0, 1]);

	let mut best_agent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, rng);
	let mut max_battles_won: i32 = 0;
	let mut opponent: Agent;
	let mut current_batch: Vec<(f32, Vec<Step>)> = Vec::new();
	for batch_num in 0..BATCH_COUNT {
		for _ in 0..BATCH_SIZE {
			opponent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, rng);
			let mut battle: BattleState = if rng.random::<bool>() {
				battle_state_normal.clone()
			} else {
				battle_state_1hp.clone()
			};
			let mut done = false;

			let mut actions_and_states: Vec<Step> = Vec::new();
			let mut battle_reward: f32 = -100.0;
			let mut turn_count = 0;
			while !done && turn_count < 1000 {
				turn_count += 1;

				let agent_mask = Mask::from_battle_state(Team::Zero, PositionId(0), &battle);
				let opponent_mask = Mask::from_battle_state(Team::One, PositionId(1), &battle);

				let encoding = encoder::encode(&battle, &registry);
				let (mut agent_moveslot, probabilities) = agent.choose_move(&encoding, &agent_mask, rng);
				if rng.random::<f32>() < EXPLORATION_CHANCE {
					agent_moveslot = agent_mask.get_random_valid(rng).unwrap();
				}
				let (opponent_moveslot, _) = opponent.choose_move(&encoding, &opponent_mask, rng);

				let actions = vec![
					agent_moveslot.to_command(Team::Zero, PositionId(0), &battle),
					opponent_moveslot.to_command(Team::One, PositionId(1), &battle)
				];
				actions_and_states.push(Step {
					encoding,
					move_chosen: agent_moveslot,
					probabilities,
				});
				(battle, battle_reward, done) = env::step(battle, actions, &registry, rng);
			}
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

		// now we have to back propagate on the agent

		// we pop from actions_and_states each time to access the next most recent action and state it came from

		// we call a function from Agent which will update the weights?
		while let Some((battle_reward, steps)) = current_batch.pop() {
			let mut gt = 1.0;
			agent.baseline += (battle_reward - agent.baseline) * BASELINE_LEARNING_RATE;
			for Step { encoding, move_chosen, probabilities } in steps {
					// don't forget to use 'reward' in here somewhere
				agent.backprop(move_chosen, battle_reward, gt, &encoding, &probabilities);
				gt = gt * DAMPING_CONSTANT;
			}
		}
	}

	//println!("neuron one for our agent: {:?}", agent.neuron1.weights);
	//println!("neuron two for our agent: {:?}", agent.neuron2.weights);

	let agent_mask_normal = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_normal);
	let agent_mask_1hp = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_1hp);
	
	println!("{:?}", agent.choose_move(&encoder::encode(&battle_state_normal, &registry), &agent_mask_normal, rng));
	println!("{:?}", agent.choose_move(&encoder::encode(&battle_state_1hp, &registry), &agent_mask_1hp, rng));
	println!("best agent normal: {:?}", best_agent.choose_move(&encoder::encode(&battle_state_normal, &registry), &agent_mask_normal, rng));
	println!("best agent 1hp: {:?}", best_agent.choose_move(&encoder::encode(&battle_state_1hp, &registry), &agent_mask_1hp, rng));
}