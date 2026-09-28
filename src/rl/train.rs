use std::error::Error;

use rand::{Rng, RngCore, seq::{IndexedMutRandom, IndexedRandom as _}};

use crate::{battle::state::{battle_state::BattleState, field::PositionId}, model::registry::Registry, rl::{agent::{Agent, LearningAgent, bot_agent::BASELINE_LEARNING_RATE, random_agent::RandomAgent, spam_agent::SpamAgent, train_config::TrainConfig}, battle_playout::{BattleEnd, PlayedBattle, play_out_battle_as}, encoder, evaluate, exploit, mask::Mask}};
use crate::battle::state::Team;

pub const EXPLORATION_CHANCE: f32 = 0.00;
pub const LEARNING_RATE: f32 = 0.05;
pub const ENTROPY_REWARD_RATE: f32 = 0.05;

// note that the total number of battles used for training
// will be BATCH_COUNT * BATCH_SIZE
/// The default run length. Overridable, see [`main_loop`].
pub const BATCH_COUNT: usize = 5_000;
const BATCH_SIZE: usize = 32;

const BATCH_PRINT_FREQ: usize = 50;
const BATCH_HISTORY_FREQ: usize = 50;

/// How often to run the frozen evaluation, and how many battles per opponent.
/// 4 opponents x 50 battles every 200 batches is ~2000 extra battles across a
/// full run, against 64,000 training battles — cheap enough to leave on.
const EVAL_FREQ: usize = 200;
const EVAL_BATTLES_EACH: usize = 50;

/// How often a fresh exploiter is trained against the *current* agent.
///
/// This is the league: instead of only ever facing weak fixed policies and its
/// own past selves — none of which are trying to punish it — the agent
/// periodically has an opponent built specifically to beat the version of it
/// that exists right now, and then has to keep playing against that opponent.
/// Past selves alone let a policy drift in a circle; an exploiter closes the
/// loop, because the only way to stop losing to it is to fix the hole it found.
const EXPLOITER_FREQ: usize = 200;

/// How many exploiters to keep.
///
/// Only the most recent few, for two reasons: an exploiter built against a
/// long-superseded version of the agent is attacking a hole that may no longer
/// exist, and an unbounded pool would dilute every other opponent late in
/// training when there are dozens of them.
const EXPLOITER_POOL_SIZE: usize = 5;

/// The chance of drawing from the exploiter pool, ramped across training.
///
/// Early on the agent still needs the basics, and a dedicated exploiter is a
/// brutal and very narrow teacher — nearly all its battles would be losses,
/// which is both discouraging as a gradient and uninformative about the rest of
/// the game. Late on the basics are settled and the holes are the only thing
/// left worth training against, so the pool's share rises.
const EXPLOITER_CHANCE_START: f32 = 0.10;
const EXPLOITER_CHANCE_END: f32 = 0.45;

/// The chance of drawing a *static* opponent — random or spam — ramped **down**
/// across training.
///
/// This was a flat 0.7, which is what made the agent brilliant against spam and
/// helpless against anything that switches. Working the shares through: with a
/// 10% exploiter draw at the start, 63% of every batch was random-or-spam, and
/// even at the end it was still ~38%. Most of a 64,000-battle run was spent
/// against opponents that never pivot, in a game where pivoting is most of the
/// skill — so the agent learned to beat spam rather than to play.
///
/// It starts at the old 0.7, because early on the basics really are what a
/// weak fixed opponent teaches, and decays to 0.15. At the end of a run the mix
/// is roughly 45% exploiters, 47% past selves, 8% statics.
const STATIC_CHANCE_START: f32 = 0.70;
const STATIC_CHANCE_END: f32 = 0.15;

/// Budget for one exploiter.
///
/// Raised from 150x16. At the old budget the pool's exploiters were about a
/// tenth the size of the probe that grades the finished agent, so the agent was
/// never trained against anything as strong as the thing measuring it — and the
/// exploitability numbers said so. This is 9,600 battles each, roughly 40% of a
/// full standalone probe.
///
/// It is not free: at [`EXPLOITER_FREQ`] of 200 over a 2,000-batch run that is
/// nine exploiters and ~86,000 extra battles, more than the training run itself.
/// Expect roughly twice the wall-clock time.
fn exploiter_config() -> exploit::ExploitConfig {
	exploit::ExploitConfig {
		batch_count: 300,
		batch_size: 32,
		measure_every: 100,
		measure_battles: 60,
		// No traces or match-up tables: this probe is being built to be played
		// against, not read.
		trace_battles: 0,
		summarise_battles: 0,
		verbose: false,
		..exploit::ExploitConfig::default()
	}
}

/// How likely the next opponent is drawn from the exploiter pool.
///
/// Linear from [`EXPLOITER_CHANCE_START`] to [`EXPLOITER_CHANCE_END`] over the
/// run. Clamped so an odd batch count can never produce a probability outside
/// the two ends.
fn exploiter_chance(batch_num: usize, total_batches: usize) -> f64 {
	ramp(EXPLOITER_CHANCE_START, EXPLOITER_CHANCE_END, batch_num, total_batches)
}

/// How likely the next opponent is a static one, given it is not an exploiter.
///
/// Ramps *down*, so the pool the agent spends most of its time against shifts
/// from fixed policies to opponents that actually play the game.
fn static_chance(batch_num: usize, total_batches: usize) -> f64 {
	ramp(STATIC_CHANCE_START, STATIC_CHANCE_END, batch_num, total_batches)
}

/// Linear interpolation from `start` to `end` across the run, clamped to the
/// two ends so an odd batch count cannot produce a probability outside them.
fn ramp(start: f32, end: f32, batch_num: usize, total_batches: usize) -> f64 {
	let progress = (batch_num as f32 / total_batches.max(1) as f32).clamp(0.0, 1.0);
	((start + (end - start) * progress) as f64).clamp(start.min(end) as f64, start.max(end) as f64)
}

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

/// Pick the next training opponent.
///
/// The exploiter pool is drawn from *separately*, before the existing
/// static/past-self split, rather than being mixed in with the past selves.
/// Exploiters are a different kind of opponent: there are few of them, they are
/// individually much stronger against this particular agent, and how often they
/// should appear changes over the run. Folding them into `past_ops` would let
/// their share drift with however many past selves happen to have accumulated.
fn get_next_opponent<'a>(
	static_ops: &'a mut Vec<Box<dyn Agent>>,
	past_ops: &'a mut Vec<Box<dyn Agent>>,
	exploiter_ops: &'a mut Vec<Box<dyn Agent>>,
	exploiter_chance: f64,
	static_chance: f64,
	rng: &mut dyn RngCore,
) -> Option<&'a mut Box<dyn Agent>> {
	if !exploiter_ops.is_empty() && rng.random_bool(exploiter_chance) {
		return exploiter_ops.choose_mut(rng);
	}
	if past_ops.is_empty() || rng.random_bool(static_chance) {
		static_ops.choose_mut(rng)
	} else {
		past_ops.choose_mut(rng)
	}

}

/// Train an exploiter against the agent as it stands, and add it to the pool.
///
/// The agent is cloned first, so the probe attacks a frozen snapshot: the thing
/// in the pool should stay aimed at the version it was built against, and the
/// agent must not be perturbed by being probed.
fn refresh_exploiter_pool(
	agent: &impl LearningAgent,
	exploiter_ops: &mut Vec<Box<dyn Agent>>,
	registry: &Registry,
	battle_states: &[BattleState],
	batch_num: usize,
	rng: &mut dyn RngCore,
) {
	let mut frozen = agent.clone();
	let probe = exploit::train_exploit_probe(
		&mut frozen, registry, battle_states, &exploiter_config(), rng,
	);

	// Worth printing: this is a live read on whether the agent currently has a
	// hole, and on which seat. A probe that gets nowhere means the last 500
	// batches left nothing easy to punish.
	println!(
		"batch {}: new exploiter trained - it beats the current agent {:.0}% of the time \
		 (as zero {:.0}%, as one {:.0}%)",
		batch_num,
		probe.peak.win_rate * 100.0,
		probe.peak.as_team_zero * 100.0,
		probe.peak.as_team_one * 100.0,
	);

	exploiter_ops.push(Box::new(probe.prober));
	// Oldest out first, so the pool is always the most recent few.
	while exploiter_ops.len() > EXPLOITER_POOL_SIZE {
		exploiter_ops.remove(0);
	}
}

fn decay_train_config(train_config: &mut TrainConfig, batch_num: usize, total_batches: usize) {
	let batch_num_f32 = batch_num as f32;
	let total_batches_f32 = total_batches as f32;
	train_config.learning_rate = LEARNING_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32;
	train_config.entropy_reward_rate = ENTROPY_REWARD_RATE * (total_batches_f32 - batch_num_f32) / total_batches_f32 + 0.05;
}

/// Train an agent and save the best version seen.
///
/// `batches` is how long the run is, and it is a parameter rather than
/// [`BATCH_COUNT`] because it is not only a stopping point: the exploiter and
/// static opponent mixes and the learning-rate decay are all scheduled as a
/// *fraction* of the run. Passing a shorter run therefore compresses the whole
/// curriculum into it rather than truncating it partway, which is what makes a
/// short run a useful rehearsal of a long one instead of just its first tenth.
///
/// `out_path` is where the best agent by frozen evaluation is written.
pub fn main_loop(
	mut agent: impl LearningAgent + 'static,
	rng: &mut dyn RngCore,
	battle_state_paths: &[&str],
	batches: usize,
	out_path: &str,
) -> Result<(), Box<dyn Error>> {
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

	// The league's pool: agents trained specifically to beat this agent.
	let mut exploiter_opponents: Vec<Box<dyn Agent>> = Vec::new();

	let battle_states: Vec<BattleState> = battle_state_paths.iter()
		.map(|s| BattleState::from_file(s)).collect::<Result<Vec<_>, _>>()?;

	// Where the agent starts, so later evals have something to be measured against.
	let baseline = evaluate::evaluate(&mut agent, &registry, &battle_states, EVAL_BATTLES_EACH);
	evaluate::print_results(0, &baseline);

	for batch_num in 0..batches {
		// Before the batch, so the new exploiter is in the pool for it.
		if batch_num > 0 && batch_num % EXPLOITER_FREQ == 0 {
			refresh_exploiter_pool(
				&agent, &mut exploiter_opponents, &registry, &battle_states, batch_num, rng,
			);
		}

		let draw_exploiter = exploiter_chance(batch_num, batches);
		let draw_static = static_chance(batch_num, batches);

		let mut current_batch: Vec<PlayedBattle> = Vec::new();
		for _ in 0..BATCH_SIZE {
			let opponent = get_next_opponent(
				&mut static_opponents,
				&mut past_self_opponents,
				&mut exploiter_opponents,
				draw_exploiter,
				draw_static,
				rng,
			).ok_or("no opponents available...")?;
			let battle: BattleState = battle_states.choose(rng).ok_or("no battle states available....")?.clone();

			// Coin-flip which side the learner takes. With deterministic damage a
			// fixed pairing is a foregone conclusion, so training only as Team
			// Zero would teach the agent that roster's outcome rather than how to
			// play. The perspective-aware encoder is what makes this possible.
			let side = if rng.random_bool(0.5) { Team::Zero } else { Team::One };
			let played_battle = play_out_battle_as(battle, &registry, &mut agent, opponent, rng, side);

			window.record(&played_battle);
			current_batch.push(played_battle);
		}

		if batch_num % BATCH_PRINT_FREQ == 0 {
			decay_train_config(&mut train_config, batch_num, batches);
			window.print(batch_num);
			// The mix is the thing that most determines what the agent becomes,
			// so it should be visible rather than buried in two constants.
			let exploiters = if exploiter_opponents.is_empty() { 0.0 } else { draw_exploiter };
			let statics = (1.0 - exploiters) * draw_static;
			println!(
				"          opponent mix: {:.0}% static | {:.0}% past selves | {:.0}% exploiters ({} in pool)",
				statics * 100.0,
				(1.0 - exploiters - statics) * 100.0,
				exploiters * 100.0,
				exploiter_opponents.len(),
			);
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
	evaluate::print_results(batches, &final_results);
	if evaluate::overall_win_rate(&final_results) > best_eval_score {
		best_eval_score = evaluate::overall_win_rate(&final_results);
		best_agent = agent.clone();
	}

	println!("\nbest agent by frozen eval ({:.1}% overall):", best_eval_score * 100.0);
	final_agent_checks(&mut best_agent, &registry, &battle_states[0]);

	println!("Saving best agent to file: {out_path}...");
	best_agent.to_file(out_path)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The pool must never grow past its cap, and must keep the *newest*
	/// entries — an exploiter built against a long-superseded agent is aimed at
	/// a hole that may not exist any more.
	#[test]
	fn the_exploiter_pool_keeps_only_the_most_recent() {
		// Stand-ins for exploiters; only the eviction order is under test.
		let mut pool: Vec<usize> = Vec::new();
		for generation in 0..12 {
			pool.push(generation);
			while pool.len() > EXPLOITER_POOL_SIZE {
				pool.remove(0);
			}
		}
		assert_eq!(pool.len(), EXPLOITER_POOL_SIZE);
		assert_eq!(pool, vec![7, 8, 9, 10, 11], "the newest five, oldest evicted first");
	}

	/// The exploiter share rises over the run and never leaves its two ends.
	#[test]
	fn the_exploiter_share_ramps_up_and_stays_in_range() {
		let total = BATCH_COUNT;
		let start = exploiter_chance(0, total);
		let end = exploiter_chance(total, total);

		assert!((start - EXPLOITER_CHANCE_START as f64).abs() < 1e-6);
		assert!((end - EXPLOITER_CHANCE_END as f64).abs() < 1e-6);
		assert!(end > start, "the pool's share has to grow, not shrink");

		let mut previous = 0.0;
		for batch in (0..=total).step_by(total / 20) {
			let chance = exploiter_chance(batch, total);
			assert!(chance >= previous - 1e-9, "the ramp must be monotonic");
			assert!((EXPLOITER_CHANCE_START as f64..=EXPLOITER_CHANCE_END as f64).contains(&chance));
			previous = chance;
		}
	}

	/// The static share falls over the run. This is the fix for an agent that
	/// beats every spam policy and loses to everything that switches.
	#[test]
	fn the_static_share_ramps_down_and_stays_in_range() {
		let total = BATCH_COUNT;
		assert!((static_chance(0, total) - STATIC_CHANCE_START as f64).abs() < 1e-6);
		assert!((static_chance(total, total) - STATIC_CHANCE_END as f64).abs() < 1e-6);

		let mut previous = 1.0;
		for batch in (0..=total).step_by(total / 20) {
			let chance = static_chance(batch, total);
			assert!(chance <= previous + 1e-9, "the static share must only fall");
			assert!((STATIC_CHANCE_END as f64..=STATIC_CHANCE_START as f64).contains(&chance));
			previous = chance;
		}
	}

	/// What the two ramps actually add up to, since that is the number that
	/// decides what the agent learns.
	#[test]
	fn the_effective_mix_shifts_away_from_fixed_policies() {
		let total = BATCH_COUNT;
		let share = |batch: usize| {
			let e = exploiter_chance(batch, total);
			let s = (1.0 - e) * static_chance(batch, total);
			(s, 1.0 - e - s, e)
		};

		let (start_static, _, _) = share(0);
		let (end_static, end_past, end_exploiter) = share(total);

		// Unchanged at the start: a fresh agent really is taught by weak
		// opponents, and the old 0.7 static share was right there.
		assert!((start_static - 0.63).abs() < 0.01, "was {start_static}");
		// By the end fixed policies are a small minority.
		assert!(end_static < 0.10, "was {end_static}");
		assert!(end_past > 0.40 && end_exploiter > 0.40,
			"the end of the run should be dominated by real opponents");
		assert!((end_static + end_past + end_exploiter - 1.0).abs() < 1e-6,
			"the three shares must be a distribution");
	}

	/// A ramp with a degenerate total must not divide by zero or escape its ends.
	#[test]
	fn ramps_survive_a_zero_length_run() {
		// Batch 0 of a zero-length run is still the beginning, so it reads as the
		// start value rather than dividing by zero.
		assert!((ramp(0.7, 0.15, 0, 0) - 0.7).abs() < 1e-6);
		// And running past the end is pinned to the end value, not extrapolated.
		assert!((ramp(0.7, 0.15, 99, 10) - 0.15).abs() < 1e-6);
	}

	/// With an empty pool the exploiter branch must be skipped entirely, or the
	/// first 500 batches would have no opponent to return.
	#[test]
	fn an_empty_exploiter_pool_falls_through_to_the_other_pools() {
		let mut rng = rand::rng();
		let mut statics: Vec<Box<dyn Agent>> = vec![Box::new(RandomAgent {})];
		let mut past: Vec<Box<dyn Agent>> = Vec::new();
		let mut exploiters: Vec<Box<dyn Agent>> = Vec::new();

		for _ in 0..50 {
			// A certainty of drawing an exploiter, with none to draw.
			let opponent = get_next_opponent(&mut statics, &mut past, &mut exploiters, 1.0, 0.7, &mut rng);
			assert!(opponent.is_some(), "must still return an opponent");
		}
	}

	/// And with a pool present, a probability of 1.0 always draws from it.
	#[test]
	fn a_full_exploiter_pool_is_drawn_from_when_the_chance_is_certain() {
		let mut rng = rand::rng();
		let mut statics: Vec<Box<dyn Agent>> = vec![Box::new(SpamAgent { index: 0 })];
		let mut past: Vec<Box<dyn Agent>> = Vec::new();
		// SpamAgent index 3 is not in the static pool, so its choice identifies it.
		let mut exploiters: Vec<Box<dyn Agent>> = vec![Box::new(SpamAgent { index: 3 })];

		let registry = Registry::load();
		let battle = BattleState::from(
			vec![crate::battle::state::creature_state::CreatureState::from_species(
				&registry, crate::model::speciesdata::SpeciesId(2),
				Registry::default_moveset(crate::model::speciesdata::SpeciesId(2)))],
			vec![crate::battle::state::creature_state::CreatureState::from_species(
				&registry, crate::model::speciesdata::SpeciesId(6),
				Registry::default_moveset(crate::model::speciesdata::SpeciesId(6)))],
			vec![0, 1],
		);
		let mask = Mask::from_battle_state(&Team::Zero, PositionId(0), &battle);
		let encoding = encoder::encode(&battle, &registry, false, &Team::Zero);

		for _ in 0..25 {
			let opponent = get_next_opponent(&mut statics, &mut past, &mut exploiters, 1.0, 0.7, &mut rng)
				.expect("an opponent");
			assert_eq!(opponent.choose_move(&encoding, &mask, &mut rng).to_number(), 3,
				"a certain draw must come from the exploiter pool");
		}
	}
}

fn final_agent_checks(agent: &mut impl LearningAgent, registry: &Registry, battle_state: &BattleState) {
	// Both sides now, since the agent can be asked to play either.
	for (team, pos) in [(Team::Zero, PositionId(0)), (Team::One, PositionId(1))] {
		let mask = Mask::from_battle_state(&team, pos, battle_state);
		let encoding = encoder::encode(battle_state, registry, false, &team);
		println!("  as {:?}: {:?}", team, agent.move_probs(&encoding, &mask));
	}
}
