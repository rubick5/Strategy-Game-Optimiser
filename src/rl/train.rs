use std::error::Error;

use rand::{Rng, RngCore, seq::{IndexedMutRandom, IndexedRandom as _}};

use crate::{battle::state::{battle_state::BattleState, field::PositionId}, model::registry::Registry, rl::{agent::{Agent, LearningAgent, bot_agent::BASELINE_LEARNING_RATE, random_agent::RandomAgent, spam_agent::SpamAgent, train_config::TrainConfig}, battle_playout::{BattleEnd, PlayedBattle, play_out_battle}, encoder, evaluate, mask::Mask}};
use crate::battle::state::Team;

pub const EXPLORATION_CHANCE: f32 = 0.00;
pub const LEARNING_RATE: f32 = 0.05;
pub const ENTROPY_REWARD_RATE: f32 = 0.05;

// note that the total number of battles used for training
// will be BATCH_COUNT * BATCH_SIZE
const BATCH_COUNT: usize = 2_000;
const BATCH_SIZE: usize = 32;

const BATCH_PRINT_FREQ: usize = 50;
const BATCH_HISTORY_FREQ: usize = 50;

/// How often to run the frozen evaluation, and how many battles per opponent.
/// 4 opponents x 50 battles every 200 batches is ~2000 extra battles across a
/// full run, against 64,000 training battles — cheap enough to leave on.
const EVAL_FREQ: usize = 200;
const EVAL_BATTLES_EACH: usize = 50;

/// Rolling counters for the battles played since the last printout.
///
/// Timeouts used to be invisible: a battle that hit the turn cap produced no
/// steps *and* counted as a not-win, so it silently removed training signal and
/// depressed the win rate with no way to notice.
#[derive(Default)]
struct TrainingWindow {
	battles: usize,
	wins: usize,
	losses: usize,
	draws: usize,
	timeouts: usize,
	total_turns: usize,
}

impl TrainingWindow {
	fn record(&mut self, played: &PlayedBattle) {
		self.battles += 1;
		self.total_turns += played.turns;
		match played.outcome {
			BattleEnd::Win => self.wins += 1,
			BattleEnd::Loss => self.losses += 1,
			BattleEnd::Draw => self.draws += 1,
			BattleEnd::Timeout => self.timeouts += 1,
		}
	}

	fn print(&self, batch_num: usize) {
		let battles = self.battles.max(1) as f32;
		println!(
			"batch {}: battles won: {} out of {} ({:.1}%) | draws {} | timeouts {} ({:.1}%) | mean {:.1} turns",
			batch_num,
			self.wins,
			// The denominator is what was actually played. The old code printed a
			// constant 1600 here, which made the very first line (a single batch,
			// 32 battles) look like a 0.4% win rate instead of ~22%.
			self.battles,
			self.wins as f32 / battles * 100.0,
			self.draws,
			self.timeouts,
			self.timeouts as f32 / battles * 100.0,
			self.total_turns as f32 / battles,
		);
	}
}

fn get_next_opponent<'a>(static_ops: &'a mut Vec<Box<dyn Agent>>, past_ops: &'a mut Vec<Box<dyn Agent>>, rng: &mut dyn RngCore) -> Option<&'a mut Box<dyn Agent>> {
	if past_ops.is_empty() || rng.random_bool(0.7) {
		static_ops.choose_mut(rng)
	} else {
		past_ops.choose_mut(rng)
	}

}

fn decay_train_config(train_config: &mut TrainConfig, batch_num: usize, total_batches: usize) {
	let batch_num_f32 = batch_num as f32;
	let total_batches_f32 = total_batches as f32;
	train_config.learning_rate = LEARNING_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32;
	train_config.entropy_reward_rate = ENTROPY_REWARD_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32 + 0.05;
}

pub fn main_loop(mut agent: impl LearningAgent + 'static, rng: &mut dyn RngCore, battle_state_paths: &[&str]) -> Result<(), Box<dyn Error>> {
	let registry = Registry::load();
	let mut window = TrainingWindow::default();

	let mut best_agent = agent.clone();
	let mut best_eval_score: f32 = -1.0;

	let mut train_config = TrainConfig {
		learning_rate: LEARNING_RATE,
		entropy_reward_rate: ENTROPY_REWARD_RATE,
		baseline_learning_rate: BASELINE_LEARNING_RATE
	};

	let mut static_opponents: Vec<Box<dyn Agent>> = vec![
		Box::new(RandomAgent{}),
		Box::new(SpamAgent{ index: 1 }),
		Box::new(SpamAgent{index: 0}),
		Box::new(SpamAgent{index: 2}),
		];

	let mut past_self_opponents: Vec<Box<dyn Agent>> = Vec::new();

	let battle_states: Vec<BattleState> = battle_state_paths.iter()
		.map(|s| BattleState::from_file(s)).collect::<Result<Vec<_>, _>>()?;

	// Where the agent starts, so later evals have something to be measured against.
	let baseline = evaluate::evaluate(&mut agent, &registry, &battle_states, EVAL_BATTLES_EACH);
	evaluate::print_results(0, &baseline);

	for batch_num in 0..BATCH_COUNT {
		let mut current_batch: Vec<PlayedBattle> = Vec::new();
		for _ in 0..BATCH_SIZE {
			let opponent = get_next_opponent(&mut static_opponents, &mut past_self_opponents, rng).ok_or("no opponents available...")?;
			let battle: BattleState = battle_states.choose(rng).ok_or("no battle states available....")?.clone();

			let played_battle = play_out_battle(battle, &registry, &mut agent, opponent, rng);

			window.record(&played_battle);
			current_batch.push(played_battle);
		}

		if batch_num % BATCH_PRINT_FREQ == 0 {
			decay_train_config(&mut train_config, batch_num, BATCH_COUNT);
			window.print(batch_num);
			window = TrainingWindow::default();
		}

		// The frozen yardstick, and the criterion for what gets saved. The old
		// code kept whichever agent scored highest on the *training* counter,
		// which compares numbers taken against different opponent mixtures.
		if batch_num > 0 && batch_num % EVAL_FREQ == 0 {
			let results = evaluate::evaluate(&mut agent, &registry, &battle_states, EVAL_BATTLES_EACH);
			evaluate::print_results(batch_num, &results);
			let score = evaluate::overall_win_rate(&results);
			if score > best_eval_score {
				best_eval_score = score;
				best_agent = agent.clone();
			}
		}

		agent.learn_from_batch(&current_batch, 1.0, &train_config);

		if batch_num % BATCH_HISTORY_FREQ == 0 {
			past_self_opponents.push(Box::new(agent.clone()));
		}
	}

	println!("\nfinal state of agent:");
	final_agent_checks(&mut agent, &registry, &battle_states[0]);
	let final_results = evaluate::evaluate(&mut agent, &registry, &battle_states, EVAL_BATTLES_EACH);
	evaluate::print_results(BATCH_COUNT, &final_results);
	if evaluate::overall_win_rate(&final_results) > best_eval_score {
		best_eval_score = evaluate::overall_win_rate(&final_results);
		best_agent = agent.clone();
	}

	println!("\nbest agent by frozen eval ({:.1}% overall):", best_eval_score * 100.0);
	final_agent_checks(&mut best_agent, &registry, &battle_states[0]);

	println!("Saving best agent to file: agent.json...");
	best_agent.to_file("agent.json")
}

fn final_agent_checks(agent: &mut impl LearningAgent, registry: &Registry, battle_state: &BattleState) {
	// Both sides now, since the agent can be asked to play either.
	for (team, pos) in [(Team::Zero, PositionId(0)), (Team::One, PositionId(1))] {
		let mask = Mask::from_battle_state(&team, pos, battle_state);
		let encoding = encoder::encode(battle_state, registry, false, &team);
		println!("  as {:?}: {:?}", team, agent.move_probs(&encoding, &mask));
	}
}
