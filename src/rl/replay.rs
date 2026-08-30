//! Watching two agents play, so a win rate can be turned into an explanation.
//!
//! The exploitability probe tells you *that* a policy has a hole. This tells you
//! *what* the hole is: which action the exploiter reaches for in each match-up,
//! what the victim does instead, and which creature falls over first.
//!
//! Two views, because they answer different questions:
//!
//! * [`trace`] prints one battle turn by turn. Good for "show me the line".
//! * [`summarise`] aggregates many battles into per-match-up action
//!   frequencies. Good for "is that line the plan, or did I watch a fluke?".

use std::collections::BTreeMap;

use rand::RngCore;

use crate::battle::command::Command;
use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::{Outcome, Team};
use crate::model::registry::Registry;
use crate::rl::agent::{Agent, LearningAgent};
use crate::rl::battle_playout::MAX_TURNS;
use crate::rl::encoder;
use crate::rl::mask::Mask;
use crate::rl::moveslot::{MOVESLOT_COUNT, Moveslot};

/// Which of the two seats a report is talking about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
	Target,
	Prober,
}

impl Role {
	fn label(self) -> &'static str {
		match self {
			Role::Target => "target",
			Role::Prober => "prober",
		}
	}
}

/// What a creature is, in one word.
fn species_name(registry: &Registry, battle: &BattleState, pos: PositionId) -> String {
	match battle.get_mon(pos) {
		Some(creature) => registry.get_species_data(creature.species_id).name.clone(),
		None => String::from("-"),
	}
}

fn hp_of(battle: &BattleState, pos: PositionId) -> (u32, u32) {
	match battle.get_mon(pos) {
		Some(creature) => (creature.current_hp, creature.max_hp),
		None => (0, 0),
	}
}

/// A chosen action, named the way a player would say it.
pub fn action_name(registry: &Registry, battle: &BattleState, pos: PositionId, slot: Moveslot) -> String {
	match slot {
		Moveslot::Slot(n) => match battle.get_mon(pos).and_then(|c| c.moves.get(n).copied()) {
			Some(move_id) => registry.get_move(move_id).name.clone(),
			None => format!("slot {}", n),
		},
		Moveslot::Switch(n) => match battle.get_mon_from_team(&pos.team(), n) {
			Some(target) => format!("-> {}", registry.get_species_data(target.species_id).name),
			None => format!("-> slot {}", n),
		},
	}
}

/// Print one battle, turn by turn.
///
/// `first` takes `first_team`; the other agent takes the other seat.
pub fn trace(
	mut battle: BattleState,
	registry: &Registry,
	first: &mut dyn Agent,
	second: &mut dyn Agent,
	first_team: Team,
	first_role: Role,
	rng: &mut dyn RngCore,
) {
	let second_team = first_team.other();
	let first_pos = battle.field.team_positions(&first_team)[0];
	let second_pos = battle.field.team_positions(&second_team)[0];
	let second_role = if first_role == Role::Target { Role::Prober } else { Role::Target };

	println!(
		"\n=== {} as {:?} vs {} as {:?} ===",
		first_role.label(), first_team, second_role.label(), second_team
	);

	let mut step_request = StepRequest::NeedsActions;
	let mut turn = 0;
	while turn < MAX_TURNS {
		match step_request {
			StepRequest::NeedsActions => {
				turn += 1;
				let first_mask = Mask::from_battle_state(&first_team, first_pos, &battle);
				let second_mask = Mask::from_battle_state(&second_team, second_pos, &battle);
				let (zero_view, one_view) = encoder::encode_both(&battle, registry, false);
				let (first_view, second_view) = match first_team {
					Team::Zero => (zero_view, one_view),
					Team::One => (one_view, zero_view),
				};

				let first_slot = first.choose_move(&first_view, &first_mask, rng);
				let second_slot = second.choose_move(&second_view, &second_mask, rng);

				let (fc, fm) = hp_of(&battle, first_pos);
				let (sc, sm) = hp_of(&battle, second_pos);
				println!(
					"t{:<3} {:>12} {:>4}/{:<4}   vs   {:<12} {:>4}/{:<4}",
					turn,
					species_name(registry, &battle, first_pos), fc, fm,
					species_name(registry, &battle, second_pos), sc, sm,
				);
				println!(
					"       {:<7} {:<20}  {:<7} {}",
					first_role.label(), action_name(registry, &battle, first_pos, first_slot),
					second_role.label(), action_name(registry, &battle, second_pos, second_slot),
				);

				let actions = vec![
					first_slot.to_command(first_pos, &battle, registry),
					second_slot.to_command(second_pos, &battle, registry),
				];
				StepResult { battle_state: battle, step_request } =
					engine::step(battle, actions, registry, rng);
			}
			StepRequest::NeedsReplacements(ref positions) => {
				for pos in positions.clone() {
					let who = if pos.team() == first_team { first_role } else { second_role };
					println!("       {} lost {}", who.label(), species_name(registry, &battle, pos));
				}
				let mut commands: Vec<Command> = Vec::new();
				for pos in positions.clone() {
					let team = pos.team();
					let encoding = encoder::encode(&battle, registry, true, &team);
					let mask = Mask::from_battle_state(&team, pos, &battle);
					let actor: &mut dyn Agent = if team == first_team { first } else { second };
					let slot = actor.choose_move(&encoding, &mask, rng);
					println!(
						"       {} sends in {}",
						if team == first_team { first_role.label() } else { second_role.label() },
						action_name(registry, &battle, pos, slot).trim_start_matches("-> "),
					);
					commands.push(slot.to_command(pos, &battle, registry));
				}
				StepResult { battle_state: battle, step_request } =
					engine::step(battle, commands, registry, rng);
			}
			StepRequest::Finished(Outcome::Win { team }) => {
				let winner = if team == first_team { first_role } else { second_role };
				println!("       *** {} wins on turn {} ***", winner.label(), turn);
				return;
			}
			StepRequest::Finished(Outcome::Draw) => {
				println!("       *** draw ***");
				return;
			}
		}
	}
	println!("       *** hit the {}-turn cap ***", MAX_TURNS);
}

/// Per-match-up action counts, plus who tends to fall over.
#[derive(Debug, Default, Clone)]
pub struct MatchupReport {
	pub battles: usize,
	pub target_wins: usize,
	pub prober_wins: usize,
	pub draws: usize,
	pub timeouts: usize,
	pub total_turns: usize,
	/// (my active, their active) -> action -> times chosen.
	pub target_choices: BTreeMap<(String, String), BTreeMap<String, usize>>,
	pub prober_choices: BTreeMap<(String, String), BTreeMap<String, usize>>,
	/// Which of the target's creatures faints first, how often.
	pub target_first_loss: BTreeMap<String, usize>,
}

impl MatchupReport {
	pub fn mean_turns(&self) -> f32 {
		if self.battles == 0 { 0.0 } else { self.total_turns as f32 / self.battles as f32 }
	}
}

fn note(
	into: &mut BTreeMap<(String, String), BTreeMap<String, usize>>,
	mine: String,
	theirs: String,
	action: String,
) {
	*into.entry((mine, theirs)).or_default().entry(action).or_insert(0) += 1;
}

/// Play `battles` battles, alternating seats, recording what each side chose.
pub fn summarise(
	battle: &BattleState,
	registry: &Registry,
	target: &mut dyn Agent,
	prober: &mut dyn Agent,
	battles: usize,
	rng: &mut dyn RngCore,
) -> MatchupReport {
	let mut report = MatchupReport::default();

	for i in 0..battles {
		// Alternate which seat the target holds, so the report covers both.
		let target_team = if i % 2 == 0 { Team::Zero } else { Team::One };
		let prober_team = target_team.other();
		let mut state = battle.clone();
		let target_pos = state.field.team_positions(&target_team)[0];
		let prober_pos = state.field.team_positions(&prober_team)[0];

		let mut step_request = StepRequest::NeedsActions;
		let mut turn = 0;
		let mut first_loss_recorded = false;

		loop {
			if turn >= MAX_TURNS {
				report.timeouts += 1;
				break;
			}
			match step_request {
				StepRequest::NeedsActions => {
					turn += 1;
					let target_mask = Mask::from_battle_state(&target_team, target_pos, &state);
					let prober_mask = Mask::from_battle_state(&prober_team, prober_pos, &state);
					let (zero_view, one_view) = encoder::encode_both(&state, registry, false);
					let (target_view, prober_view) = match target_team {
						Team::Zero => (zero_view, one_view),
						Team::One => (one_view, zero_view),
					};

					let target_slot = target.choose_move(&target_view, &target_mask, rng);
					let prober_slot = prober.choose_move(&prober_view, &prober_mask, rng);

					let target_mon = species_name(registry, &state, target_pos);
					let prober_mon = species_name(registry, &state, prober_pos);
					note(&mut report.target_choices, target_mon.clone(), prober_mon.clone(),
						action_name(registry, &state, target_pos, target_slot));
					note(&mut report.prober_choices, prober_mon, target_mon,
						action_name(registry, &state, prober_pos, prober_slot));

					let actions = vec![
						target_slot.to_command(target_pos, &state, registry),
						prober_slot.to_command(prober_pos, &state, registry),
					];
					StepResult { battle_state: state, step_request } =
						engine::step(state, actions, registry, rng);
				}
				StepRequest::NeedsReplacements(ref positions) => {
					if !first_loss_recorded {
						for pos in positions.clone() {
							if pos.team() == target_team {
								*report
									.target_first_loss
									.entry(species_name(registry, &state, pos))
									.or_insert(0) += 1;
								first_loss_recorded = true;
							}
						}
					}
					let mut commands: Vec<Command> = Vec::new();
					for pos in positions.clone() {
						let team = pos.team();
						let encoding = encoder::encode(&state, registry, true, &team);
						let mask = Mask::from_battle_state(&team, pos, &state);
						let actor: &mut dyn Agent =
							if team == target_team { target } else { prober };
						commands.push(actor.choose_move(&encoding, &mask, rng).to_command(pos, &state, registry));
					}
					StepResult { battle_state: state, step_request } =
						engine::step(state, commands, registry, rng);
				}
				StepRequest::Finished(Outcome::Win { team }) => {
					if team == target_team { report.target_wins += 1 } else { report.prober_wins += 1 }
					break;
				}
				StepRequest::Finished(Outcome::Draw) => {
					report.draws += 1;
					break;
				}
			}
		}
		report.battles += 1;
		report.total_turns += turn;
	}

	report
}

/// Show the top few actions per match-up, most-used first.
fn print_choices(title: &str, choices: &BTreeMap<(String, String), BTreeMap<String, usize>>) {
	println!("\n{}", title);
	for ((mine, theirs), actions) in choices {
		let total: usize = actions.values().sum();
		if total < 5 {
			continue; // too rare to read anything into
		}
		let mut ranked: Vec<(&String, &usize)> = actions.iter().collect();
		ranked.sort_by(|a, b| b.1.cmp(a.1));
		let shown: Vec<String> = ranked
			.iter()
			.take(3)
			.map(|(name, count)| format!("{} {:.0}%", name, **count as f32 / total as f32 * 100.0))
			.collect();
		println!("  {:>12} vs {:<12}  {}", mine, theirs, shown.join("   "));
	}
}

pub fn print_summary(report: &MatchupReport) {
	println!("\n=== match-up summary over {} battles ===", report.battles);
	println!(
		"target {} / prober {} / draws {} / timeouts {}   mean {:.1} turns",
		report.target_wins, report.prober_wins, report.draws, report.timeouts, report.mean_turns()
	);

	if !report.target_first_loss.is_empty() {
		let mut ranked: Vec<(&String, &usize)> = report.target_first_loss.iter().collect();
		ranked.sort_by(|a, b| b.1.cmp(a.1));
		let shown: Vec<String> = ranked
			.iter()
			.map(|(name, count)| format!("{} {}x", name, count))
			.collect();
		println!("first of the target's creatures to fall: {}", shown.join(", "));
	}

	print_choices("what the PROBER does (its exploit):", &report.prober_choices);
	print_choices("what the TARGET does in reply:", &report.target_choices);
}

/// The full action distribution a learning agent assigns at one position.
///
/// Useful for "it keeps clicking that - how sure is it?". Note that `move_probs`
/// is not consistent across the two agent types: `PPOAgent` returns softmaxed
/// probabilities while `BotAgent` returns masked logits, so this normalises
/// whatever it gets before printing.
pub fn print_policy(
	agent: &mut impl LearningAgent,
	battle: &BattleState,
	registry: &Registry,
	team: Team,
	label: &str,
) {
	let pos = match battle.field.team_positions(&team).first().copied() {
		Some(pos) => pos,
		None => return,
	};
	let mask = Mask::from_battle_state(&team, pos, battle);
	let encoding = encoder::encode(battle, registry, false, &team);
	let raw = agent.move_probs(&encoding, &mask);
	let probs = as_probabilities(&raw);

	let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
	ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

	println!("\n{} ({:?}, {}):", label, team, species_name(registry, battle, pos));
	for (index, p) in ranked.into_iter().take(6) {
		if p < 0.005 {
			continue;
		}
		let slot = Moveslot::from_number(index);
		let kind = if index < MOVESLOT_COUNT { "move  " } else { "switch" };
		println!("  {:>5.1}%  {} {}", p * 100.0, kind, action_name(registry, battle, pos, slot));
	}
}

/// Turn whatever `move_probs` handed back into a distribution.
fn as_probabilities(raw: &[f32]) -> Vec<f32> {
	let finite_sum: f32 = raw.iter().filter(|v| v.is_finite()).sum();
	let looks_normalised = (finite_sum - 1.0).abs() < 0.01
		&& raw.iter().filter(|v| v.is_finite()).all(|v| (0.0..=1.0).contains(v));
	if looks_normalised {
		raw.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect()
	} else {
		crate::rl::agent::softmax(raw)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::creature_state::CreatureState;
	use crate::model::speciesdata::SpeciesId;
	use crate::rl::agent::random_agent::RandomAgent;
	use crate::rl::agent::ppo_agent::PPOAgent;

	fn battle(registry: &Registry) -> BattleState {
		let mon = |id: u32| {
			CreatureState::from_species(registry, SpeciesId(id), Registry::default_moveset(SpeciesId(id)))
		};
		BattleState::from(vec![mon(2), mon(3)], vec![mon(6), mon(5)], vec![0, 1])
	}

	#[test]
	fn summary_accounts_for_every_battle() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut a = RandomAgent {};
		let mut b = RandomAgent {};
		let report = summarise(&battle(&registry), &registry, &mut a, &mut b, 8, &mut rng);

		assert_eq!(report.battles, 8);
		assert_eq!(
			report.target_wins + report.prober_wins + report.draws + report.timeouts,
			report.battles
		);
		assert!(report.mean_turns() > 0.0);
		assert!(!report.prober_choices.is_empty(), "choices should have been recorded");
	}

	/// Percentages are only meaningful if the counts add up to the turns played.
	#[test]
	fn choice_counts_are_consistent() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut a = RandomAgent {};
		let mut b = RandomAgent {};
		let report = summarise(&battle(&registry), &registry, &mut a, &mut b, 6, &mut rng);

		let target_total: usize = report.target_choices.values().flat_map(|m| m.values()).sum();
		let prober_total: usize = report.prober_choices.values().flat_map(|m| m.values()).sum();
		assert_eq!(target_total, prober_total, "both sides act once per turn");
	}

	/// `move_probs` differs between agent types; both must normalise the same way.
	#[test]
	fn probabilities_are_normalised_whatever_the_agent_returns() {
		// Already-softmaxed input is left alone.
		let probs = as_probabilities(&[0.7, 0.2, 0.1]);
		assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
		assert!((probs[0] - 0.7).abs() < 1e-5);

		// Raw logits get softmaxed.
		let logits = as_probabilities(&[5.0, 1.0, 1.0]);
		assert!((logits.iter().sum::<f32>() - 1.0).abs() < 1e-5);
		assert!(logits[0] > logits[1], "ordering must survive");

		// Masked-off entries land at zero rather than NaN.
		let masked = as_probabilities(&[2.0, f32::NEG_INFINITY, 1.0]);
		assert_eq!(masked[1], 0.0);
		assert!((masked.iter().sum::<f32>() - 1.0).abs() < 1e-5);
	}

	#[test]
	fn tracing_runs_without_panicking() {
		let registry = Registry::load();
		let mut rng = rand::rng();
		let mut target = PPOAgent::init_random(&mut rng);
		let mut prober = RandomAgent {};
		trace(battle(&registry), &registry, &mut prober, &mut target, Team::One, Role::Prober, &mut rng);
	}
}
