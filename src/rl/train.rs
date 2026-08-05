use rand::Rng;

use crate::{battle::{state::{BattleState, PokemonState, PositionId}}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::BotAgent, battle_playout::{Step, play_out_battle}, encoder, learner::learn_from_battle, mask::Mask}};
use crate::battle::state::Team;

pub const EXPLORATION_CHANCE: f32 = 0.05;

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
	final_agent_checks(agent, &registry);
	println!("highest winrate agent:");
	final_agent_checks(best_agent, &registry);
}

fn final_agent_checks(mut agent: BotAgent, registry: &Registry) {
	let agent_mask_normal = Mask::from_battle_state(Team::Zero, PositionId(0), &start_battle_state(registry));
	let agent_mask_1hp = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_1hp(registry));
	
	//println!("agent normal: {:?}", agent.choose_move(&encoder::encode(&start_battle_state(registry), &registry, false), &agent_mask_normal, rng));
	//println!("agent 1hp: {:?}", agent.choose_move(&encoder::encode(&battle_state_1hp(registry), &registry, false), &agent_mask_1hp, rng));
	println!("agent normal: {:?}", agent.just_logits(&encoder::encode(&start_battle_state(registry), registry, false), &agent_mask_normal));
	println!("agent 1hp: {:?}", agent.just_logits(&encoder::encode(&battle_state_1hp(registry), registry, false), &agent_mask_1hp));

}




