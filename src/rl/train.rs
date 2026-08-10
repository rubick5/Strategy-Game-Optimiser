use rand::{RngCore, seq::{IndexedMutRandom, IndexedRandom as _}};

use crate::{battle::state::{battle_state::BattleState, creature_state::CreatureState, field::PositionId}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::{Agent, bot_agent::{BASELINE_LEARNING_RATE, BotAgent}, random_agent::RandomAgent, spam_agent::SpamAgent, train_config::TrainConfig}, battle_playout::{PlayedBattle, play_out_battle}, encoder, learner::learn_from_batch, mask::Mask}};
use crate::battle::state::Team;

pub const EXPLORATION_CHANCE: f32 = 0.05;
pub const LEARNING_RATE: f32 = 0.05;
pub const ENTROPY_REWARD_RATE: f32 = 0.05;

// note that the total number of battles used for training
// will be BATCH_COUNT * BATCH_SIZE
const BATCH_COUNT: usize = 10_000;
const BATCH_SIZE: usize = 64;

const BATCH_PRINT_FREQ: usize = 50;
const BATCH_PRINT_GAP_SIZE: usize = BATCH_SIZE * BATCH_PRINT_FREQ;


pub fn start_battle_state(registry: &Registry) -> BattleState {
	let ps0 = CreatureState::from_species(registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let ps01 = CreatureState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let ps11 = CreatureState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1), MoveId(2)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = CreatureState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let mut ps1_1hp = CreatureState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	ps1_1hp.current_hp = 1;
	BattleState::from(vec![ps0, ps01], vec![ps1, ps11], vec![0, 1])
}

pub fn battle_state_1hp(registry: &Registry) -> BattleState {
	let ps0 = CreatureState::from_species(registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let _ps01 = CreatureState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let _ps11 = CreatureState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1), MoveId(2)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let _ps1 = CreatureState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	let mut ps1_1hp = CreatureState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1), MoveId(2)]);
	ps1_1hp.current_hp = 1;
	BattleState::from(vec![ps00], vec![ps1_1hp], vec![0, 1])
}

fn decay_train_config(train_config: &mut TrainConfig, batch_num: usize, total_batches: usize) {
	let batch_num_f32 = batch_num as f32;
	let total_batches_f32 = total_batches as f32;
	train_config.learning_rate = LEARNING_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32;
	train_config.entropy_reward_rate = ENTROPY_REWARD_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32;
}



pub fn main_loop(mut rng: &mut dyn RngCore) {
	let registry = Registry::load();
	let mut agent: BotAgent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let mut battles_won = 0;


	let mut best_agent: BotAgent = BotAgent::init_random(encoder::TOTAL_ENCODING_LEN, rng);
	let mut max_battles_won: i32 = 0;

	let mut train_config = TrainConfig {
		learning_rate: LEARNING_RATE,
		entropy_reward_rate: ENTROPY_REWARD_RATE,
		baseline_learning_rate: BASELINE_LEARNING_RATE
	};
	
	let mut opponents: Vec<Box<dyn Agent>> = vec![
		Box::new(RandomAgent{}),
		Box::new(SpamAgent{ index: 1 }),
		Box::new(SpamAgent{index: 0}),
		];

	let battle_states: Vec<BattleState> = vec![
		start_battle_state(&registry),
		start_battle_state(&registry),
		battle_state_1hp(&registry),
	];
	for batch_num in 0..BATCH_COUNT {
		let mut current_batch: Vec<PlayedBattle> = Vec::new();
		for _ in 0..BATCH_SIZE {
			let opponent = opponents.choose_mut(rng).unwrap();
			let battle: BattleState = battle_states.choose(rng).unwrap_or(&battle_state_1hp(&registry)).clone();

			let played_battle = play_out_battle(battle, &registry, &mut agent, opponent, rng);

			if played_battle.battle_reward > 0.0 {
				battles_won += 1;
			}
			current_batch.push(played_battle);
		}
		if batch_num % BATCH_PRINT_FREQ == 0 {
			decay_train_config(&mut train_config, batch_num, BATCH_COUNT);
			println!("batch {}: battles won: {} out of {}", batch_num, battles_won, BATCH_PRINT_GAP_SIZE);
			if battles_won > max_battles_won {
				best_agent = agent.clone();
				max_battles_won = battles_won;
			}
			battles_won = 0;
		}
		learn_from_batch(&mut agent, &current_batch, 1.0, &train_config);
	}
	println!("final state of agent:");
	final_agent_checks(&mut agent, &registry);
	println!("highest winrate agent:");
	final_agent_checks(&mut best_agent, &registry);

	println!("Saving final agent to file: agent.json...");
	agent.to_file("agent.json").unwrap() // goofy unwrap
}

fn final_agent_checks(agent: &mut BotAgent, registry: &Registry) {
	let agent_mask_normal = Mask::from_battle_state(&Team::Zero, PositionId(0), &start_battle_state(registry));
	let agent_mask_1hp = Mask::from_battle_state(&Team::Zero, PositionId(0), &battle_state_1hp(registry));
	
	//println!("agent normal: {:?}", agent.choose_move(&encoder::encode(&start_battle_state(registry), &registry, false), &agent_mask_normal, rng));
	//println!("agent 1hp: {:?}", agent.choose_move(&encoder::encode(&battle_state_1hp(registry), &registry, false), &agent_mask_1hp, rng));
	println!("agent normal: {:?}", agent.just_logits(&encoder::encode(&start_battle_state(registry), registry, false), &agent_mask_normal));
	println!("agent 1hp: {:?}", agent.just_logits(&encoder::encode(&battle_state_1hp(registry), registry, false), &agent_mask_1hp));

}




