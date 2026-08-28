//! A fixed yardstick for measuring the agent.
//!
//! # Why this exists
//!
//! The win rate printed during training is measured against a pool that *changes
//! as training goes on*: 70% static opponents, 30% past selves, and the past-self
//! pool gains a snapshot every 50 batches. So a rising training win rate could
//! mean the agent improved, or the pool happened to get easier, and there is no
//! way to tell which from the number alone.
//!
//! Evaluation here is deliberately frozen on every axis that can drift:
//!
//! * the opponents are the stateless ones (random, spam), which cannot improve
//! * the battle states are the same ones every time
//! * the RNG is seeded from a constant, so the *same* battles are replayed at
//!   every checkpoint, and a change in the number is a change in the agent
//!
//! It also deliberately does not touch the training RNG, so adding or removing an
//! eval does not shift the training run's random stream.

use rand::prelude::SeedableRng;
use rand::rngs::StdRng;

use crate::battle::state::battle_state::BattleState;
use crate::model::registry::Registry;
use crate::rl::agent::{Agent, LearningAgent, random_agent::RandomAgent, spam_agent::SpamAgent};
use crate::rl::battle_playout::{BattleEnd, play_out_battle};

/// Fixed seed, so every checkpoint plays the same battles.
const EVAL_SEED: u64 = 20_240_601;

#[derive(Debug, Clone)]
pub struct EvalResult {
	pub opponent: String,
	pub battles: usize,
	pub wins: usize,
	pub losses: usize,
	pub draws: usize,
	pub timeouts: usize,
	pub mean_turns: f32,
}

impl EvalResult {
	pub fn win_rate(&self) -> f32 {
		if self.battles == 0 { 0.0 } else { self.wins as f32 / self.battles as f32 }
	}

	pub fn timeout_rate(&self) -> f32 {
		if self.battles == 0 { 0.0 } else { self.timeouts as f32 / self.battles as f32 }
	}
}

/// The frozen opponent set. Stateless by construction — none of these learn, so
/// the yardstick cannot move.
pub fn fixed_opponents() -> Vec<(String, Box<dyn Agent>)> {
	vec![
		(String::from("random"), Box::new(RandomAgent {}) as Box<dyn Agent>),
		(String::from("spam-0"), Box::new(SpamAgent { index: 0 })),
		(String::from("spam-1"), Box::new(SpamAgent { index: 1 })),
		(String::from("spam-2"), Box::new(SpamAgent { index: 2 })),
	]
}

/// Play `battles_each` battles against every fixed opponent, cycling through
/// `battle_states`.
///
/// The agent is taken by `&mut` because a forward pass needs it, but nothing here
/// calls `learn_from_batch` — evaluation never updates weights.
pub fn evaluate(
	agent: &mut impl LearningAgent,
	registry: &Registry,
	battle_states: &[BattleState],
	battles_each: usize,
) -> Vec<EvalResult> {
	let mut rng = StdRng::seed_from_u64(EVAL_SEED);
	let mut results = Vec::new();

	for (name, mut opponent) in fixed_opponents() {
		let mut wins = 0;
		let mut losses = 0;
		let mut draws = 0;
		let mut timeouts = 0;
		let mut total_turns = 0usize;

		for i in 0..battles_each {
			// Cycle rather than sample, so each opponent faces the same spread of
			// starting positions.
			let battle = match battle_states.get(i % battle_states.len().max(1)) {
				Some(battle) => battle.clone(),
				None => break,
			};
			let played = play_out_battle(battle, registry, agent, opponent.as_mut(), &mut rng);
			total_turns += played.turns;
			match played.outcome {
				BattleEnd::Win => wins += 1,
				BattleEnd::Loss => losses += 1,
				BattleEnd::Draw => draws += 1,
				BattleEnd::Timeout => timeouts += 1,
			}
		}

		let battles = wins + losses + draws + timeouts;
		results.push(EvalResult {
			opponent: name,
			battles,
			wins,
			losses,
			draws,
			timeouts,
			mean_turns: if battles == 0 { 0.0 } else { total_turns as f32 / battles as f32 },
		});
	}

	results
}

/// Mean win rate across opponents. Used to pick the agent worth saving, which is
/// a fairer criterion than the training counter — that one compares scores taken
/// against different opponent mixtures.
pub fn overall_win_rate(results: &[EvalResult]) -> f32 {
	if results.is_empty() {
		return 0.0;
	}
	results.iter().map(|r| r.win_rate()).sum::<f32>() / results.len() as f32
}

/// Print the per-opponent breakdown.
///
/// The timeout column matters: a battle that hits the turn cap contributes no
/// steps to learning, so a high number there means training data is being thrown
/// away, not just that the agent is losing.
pub fn print_results(batch_num: usize, results: &[EvalResult]) {
	println!("  [eval @ batch {}] overall {:.1}%", batch_num, overall_win_rate(results) * 100.0);
	for r in results {
		println!(
			"    vs {:<8} {:>5.1}% win  ({}W/{}L/{}D/{}T)  mean {:.0} turns",
			r.opponent,
			r.win_rate() * 100.0,
			r.wins,
			r.losses,
			r.draws,
			r.timeouts,
			r.mean_turns,
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::creature_state::CreatureState;
	use crate::model::speciesdata::SpeciesId;
	use crate::rl::agent::ppo_agent::PPOAgent;

	fn mon(registry: &Registry, id: u32) -> CreatureState {
		CreatureState::from_species(registry, SpeciesId(id), Registry::default_moveset(SpeciesId(id)))
	}

	fn small_battle(registry: &Registry) -> BattleState {
		BattleState::from(
			vec![mon(registry, 2), mon(registry, 3)],
			vec![mon(registry, 6), mon(registry, 5)],
			vec![0, 1],
		)
	}

	/// End-to-end smoke test: a real agent, real opponents, real battles.
	///
	/// This is the one test that drives the whole stack at once — perspective
	/// encoding for both sides, masking, the engine with abilities and weather,
	/// and the outcome accounting. If the two-sided encoding were wired up wrong,
	/// the opponent would be reading the learner's slots and this is where it
	/// would surface.
	#[test]
	fn evaluation_runs_and_accounts_for_every_battle() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut agent = PPOAgent::init_random(&mut rng);
		let battles = vec![small_battle(&registry)];

		let results = evaluate(&mut agent, &registry, &battles, 3);

		assert_eq!(results.len(), fixed_opponents().len());
		for r in &results {
			assert_eq!(r.battles, 3, "every battle must land in exactly one bucket");
			assert_eq!(r.wins + r.losses + r.draws + r.timeouts, r.battles);
			assert!(r.mean_turns > 0.0, "a played battle takes at least one turn");
			assert!((0.0..=1.0).contains(&r.win_rate()));
		}
	}

	/// The eval must be reproducible, or a change in the number cannot be
	/// attributed to a change in the agent.
	#[test]
	fn evaluation_is_deterministic_for_a_fixed_agent() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut agent = PPOAgent::init_random(&mut rng);
		let battles = vec![small_battle(&registry)];

		let first = evaluate(&mut agent, &registry, &battles, 3);
		let second = evaluate(&mut agent, &registry, &battles, 3);

		for (a, b) in first.iter().zip(second.iter()) {
			assert_eq!(a.opponent, b.opponent);
			assert_eq!((a.wins, a.losses, a.draws, a.timeouts), (b.wins, b.losses, b.draws, b.timeouts));
		}
	}

	/// An empty result set must not divide by zero.
	#[test]
	fn overall_win_rate_handles_no_results() {
		assert_eq!(overall_win_rate(&[]), 0.0);
	}
}
