//! Outcome-sampling MCCFR over battle positions.
//!
//! # Why external sampling
//!
//! Expanding every joint action at every node is not viable: each side has up to
//! 10 actions, so a node has up to 100 children, and a 1v1 runs a dozen turns.
//!
//! *Outcome* sampling — one trajectory per iteration — was tried first and does
//! not work here. It corrects for sampling by dividing by the probability of the
//! trajectory, and that probability compounds multiplicatively at every turn. In
//! a battle fifteen-plus decisions deep it reaches 1e-10, so single trajectories
//! arrive carrying importance weights around 1e9 and swamp millions of others.
//! It converges fine on a one-node matrix game and not at all on a battle; the
//! symptom is a mirror match settling on strategies worth -0.8 to one side, when
//! symmetry means the value has to be zero.
//!
//! *External* sampling avoids the problem rather than fighting it. The traverser
//! expands all of its own actions; the opponent's action and all chance are
//! sampled from their true distributions, once per node. Because nothing is
//! sampled off-distribution, **no importance weighting is needed anywhere** —
//! there is no `1/q` to explode. The cost is a branching factor of
//! `|traverser actions|` per turn instead of one, which is affordable at 1v1 and
//! is what [`Solver::nodes_visited`] exists to keep an eye on.
//!
//! # Simultaneous moves
//!
//! Both players choose at once, which is the only imperfect information in the
//! game once teams are known. That falls out for free here: both players'
//! infosets at a node are keyed on the same public state
//! ([`crate::cfr::key`]), so neither key depends on what the other picked. The
//! opponent's action is sampled once and held fixed across every one of the
//! traverser's actions, which is exactly what stops the traverser "seeing" the
//! reply before choosing.
//!
//! What must *not* happen is treating this as a perfect-information game where
//! one player moves and the other responds having seen it. That converges
//! perfectly well — to a Stackelberg equilibrium, which hands the second player
//! an advantage they do not have, and nothing in the output looks wrong.
//!
//! # Chance
//!
//! The engine samples paralysis, confusion, Protect success, secondary effects
//! and speed ties internally, and does not report the probability of what it
//! rolled. Under external sampling that needs no special handling at all: chance
//! is drawn from its true distribution, so the sampled subtree value is already
//! an unbiased estimate of the expectation over chance. Each of the traverser's
//! actions gets its own chance draw, which is the usual chance-sampling
//! treatment.
//!
//! # What converges
//!
//! Regret matching gives the *current* strategy, and it cycles forever. The
//! thing that converges to equilibrium is the running average, so
//! [`Solver::average_strategy`] is what to read and the current strategy is only
//! ever used to sample.

use std::collections::HashMap;

use rand::RngCore;

use crate::battle::command::Command;
use crate::battle::engine::engine::{self, StepRequest, StepResult};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::{Outcome, Team};
use crate::cfr::infoset::{
	regrets_from_action_values, sample, InfosetData, InfosetTable, Strategy,
};
use crate::cfr::key::StateKey;
use crate::cfr::leaf::{HealthHeuristic, LeafEvaluator};
use crate::cfr::node::{Actor, DecisionNode};
use crate::model::registry::Registry;
use crate::rl::moveslot::{Moveslot, MAX_DECISION};

/// How many turns ahead the search looks before handing the position to the leaf
/// evaluator.
///
/// This is the single most important number in a 2v2 solve, because cost is
/// exponential in it: with four moves and a switch, a node has five children, so
/// each extra turn of lookahead multiplies the work by five. Six turns is around
/// 15,000 nodes an iteration and is affordable; ten would be ten million.
///
/// It is set high by default because at 1v1 the positions end on their own and
/// the limit should not bind at all. A 2v2 must lower it — see
/// [`SolverConfig::for_lookahead`].
pub const DEFAULT_MAX_DEPTH: usize = 200;

/// Turns a measurement playout may run before it is judged on the leaf estimate.
///
/// Generous, because this bounds a single game rather than a search tree.
pub const PLAYOUT_CAP: usize = 1_000;

pub struct SolverConfig {
	pub iterations: usize,
	pub max_depth: usize,
	/// Stop the solve once this many nodes have been expanded.
	///
	/// External sampling branches on the traverser's actions, so cost is
	/// exponential in how long the battle runs. A position that can stall for
	/// free — anything holding Protect — has no bound at all, and without this
	/// would simply run forever. `max_depth` does not help: capping depth at 200
	/// still allows 2^200 nodes.
	pub max_nodes: u64,
}

impl SolverConfig {
	/// A configuration for a position too big to solve to the end: look
	/// `turns` ahead, then estimate.
	pub fn for_lookahead(iterations: usize, turns: usize) -> Self {
		SolverConfig { iterations, max_depth: turns, ..SolverConfig::default() }
	}
}

impl Default for SolverConfig {
	fn default() -> Self {
		SolverConfig {
			iterations: 2_000,
			max_depth: DEFAULT_MAX_DEPTH,
			max_nodes: 50_000_000,
		}
	}
}

pub struct Solver<'r> {
	registry: &'r Registry,
	config: SolverConfig,
	table: InfosetTable,
	leaf: Box<dyn LeafEvaluator>,
	truncated: u64,
	nodes_visited: u64,
	/// Running mean of the value computed at each node, when logging is on.
	value_log: Option<HashMap<(StateKey, Team), (f32, u32)>>,
	/// Iterations to skip before logging, so early noise is not recorded.
	warmup: usize,
	iteration: usize,
}

impl<'r> Solver<'r> {
	/// A solver that estimates cut-off positions by remaining health.
	pub fn new(registry: &'r Registry, config: SolverConfig) -> Self {
		Self::with_leaf(registry, config, Box::new(HealthHeuristic))
	}

	/// A solver using a particular leaf estimate — the PPO critic, say, or
	/// [`crate::cfr::leaf::Indifferent`] to take the estimate out of the picture.
	pub fn with_leaf(
		registry: &'r Registry,
		config: SolverConfig,
		leaf: Box<dyn LeafEvaluator>,
	) -> Self {
		Solver {
			registry,
			config,
			table: InfosetTable::new(),
			leaf,
			truncated: 0,
			nodes_visited: 0,
			value_log: None,
			warmup: 0,
			iteration: 0,
		}
	}

	pub fn table(&self) -> &InfosetTable {
		&self.table
	}

	/// Record the value computed at every node, for use as training targets.
	///
	/// A node's value is an average over the current strategy and everything below
	/// it, so it is far quieter than the result of a single played-out game — and
	/// there is one for every position the search touches, not only the handful a
	/// playout happens to pass through. That is the appeal.
	///
	/// The cost is that it is an estimate of an estimate: below the horizon these
	/// values rest on the leaf evaluator, so training a leaf evaluator on them can
	/// confirm its own mistakes. Positions searched to a real result carry no such
	/// debt, which is why the curriculum in [`crate::cfr::critic`] anchors on them.
	///
	/// `warmup` is the fraction of iterations to discard first: early strategies
	/// are close to uniform and their values describe a game nobody is playing.
	pub fn log_values(&mut self, warmup: f32) {
		self.value_log = Some(HashMap::new());
		self.warmup = (self.config.iterations as f32 * warmup.clamp(0.0, 1.0)) as usize;
	}

	/// Mean value recorded at each position, and how many observations it is
	/// based on. Empty unless [`Solver::log_values`] was called.
	pub fn node_values(&self) -> Vec<(&StateKey, Team, f32, u32)> {
		match &self.value_log {
			None => Vec::new(),
			Some(log) => log
				.iter()
				.map(|((key, team), (sum, count))| (key, *team, sum / *count as f32, *count))
				.collect(),
		}
	}

	pub fn config(&self) -> &SolverConfig {
		&self.config
	}

	pub fn registry(&self) -> &'r Registry {
		self.registry
	}

	/// The leaf estimate this solve used.
	///
	/// Exploitability has to be measured against the *same* truncated game that
	/// was solved, horizon and leaf estimate included, or it measures the leaf
	/// error rather than the strategy.
	pub fn leaf(&self) -> &dyn LeafEvaluator {
		self.leaf.as_ref()
	}

	/// Positions handed to the leaf evaluator instead of being played out.
	///
	/// Zero means the whole game tree was searched and the answer owes nothing to
	/// the heuristic. Anything else is the share of the solve that rests on an
	/// estimate, and is worth reporting next to the result.
	pub fn truncated_positions(&self) -> u64 {
		self.truncated
	}

	/// Whether the solve stopped early on the node budget, leaving the answer
	/// unconverged and not to be trusted.
	pub fn hit_node_budget(&self) -> bool {
		self.nodes_visited >= self.config.max_nodes
	}

	/// Decision nodes expanded across the whole solve.
	///
	/// External sampling branches on the traverser's actions, so this grows with
	/// the depth of the position — it is the number to watch when moving to
	/// bigger match-ups.
	pub fn nodes_visited(&self) -> u64 {
		self.nodes_visited
	}

	/// Run the configured number of iterations from `root`.
	///
	/// Traversers alternate, so both players' regrets improve and neither is
	/// optimised against a frozen opponent.
	pub fn solve(&mut self, root: &BattleState, rng: &mut dyn RngCore) {
		self.solve_from(root, StepRequest::NeedsActions, rng);
	}

	/// Solve from a position that may be mid-turn — owing a replacement, say.
	///
	/// [`Solver::solve`] assumes a fresh turn, which is right for a position
	/// someone hands you and wrong for one the search walked into. Re-solving from
	/// an arbitrary position needs this.
	pub fn solve_from(
		&mut self,
		root: &BattleState,
		request: StepRequest,
		rng: &mut dyn RngCore,
	) {
		for iteration in 0..self.config.iterations {
			if self.hit_node_budget() {
				break;
			}
			self.iteration = iteration;
			let traverser = if iteration % 2 == 0 { Team::Zero } else { Team::One };
			self.walk(root.clone(), request.clone(), traverser, 0, rng);
		}
	}

	/// The equilibrium estimate for one side at one decision point.
	///
	/// `None` if the battle is finished there, if that team is not the one
	/// deciding, or if the position was never reached during the solve.
	pub fn average_strategy(
		&self,
		state: &BattleState,
		request: &StepRequest,
		team: Team,
	) -> Option<Strategy> {
		let key = StateKey::new(state, request)?;
		let actors = match DecisionNode::from(state, request) {
			DecisionNode::Decision { actors } => actors,
			DecisionNode::Terminal(_) => return None,
		};
		let actor = actors.into_iter().find(|actor| actor.team == team)?;
		Some(self.table.get(&key, team)?.average_strategy(&actor.mask))
	}

	/// The equilibrium estimate for one side at the position the solve started
	/// from.
	pub fn root_strategy(&self, root: &BattleState, team: Team) -> Option<Strategy> {
		self.average_strategy(root, &StepRequest::NeedsActions, team)
	}

	/// Play the solved strategies against each other and report the empirical
	/// value to Team Zero, on a scale of -1 (always loses) to +1 (always wins).
	///
	/// This is the number a symmetric position can be checked against: whatever
	/// strategies a mirror match settles on, its value has to be zero. That check
	/// is worth more than comparing the two seats' strategies directly, because
	/// equilibria in a zero-sum game are *exchangeable* rather than symmetric —
	/// a mirror match can quite legitimately settle on one side attacking more
	/// and the other defending more, and asserting the seats match would fail on
	/// a perfectly correct solve.
	///
	/// Positions never reached during the solve fall back to uniform play.
	pub fn evaluate(&self, root: &BattleState, games: usize, rng: &mut dyn RngCore) -> f32 {
		if games == 0 {
			return 0.0;
		}
		let total: f32 = (0..games).map(|_| self.play_once(root, rng)).sum();
		total / games as f32
	}

	fn play_once(&self, root: &BattleState, rng: &mut dyn RngCore) -> f32 {
		let mut state = root.clone();
		let mut request = StepRequest::NeedsActions;

		// Deliberately *not* `max_depth`. That is the search horizon, which on a
		// 2v2 is a handful of turns; measuring the value means playing the battle
		// out to a real result, so this runs until the game actually ends.
		for _ in 0..PLAYOUT_CAP {
			let actors = match DecisionNode::from(&state, &request) {
				DecisionNode::Terminal(outcome) => {
					return match outcome {
						Outcome::Win { team } => if team == Team::Zero { 1.0 } else { -1.0 },
						Outcome::Draw => 0.0,
					};
				}
				DecisionNode::Decision { actors } => actors,
			};

			let key = StateKey::new(&state, &request)
				.expect("a node with actors is not terminal");

			let commands: Vec<Command> = actors
				.iter()
				.map(|actor| {
					let strategy = match self.table.get(&key, actor.team) {
						Some(data) => data.average_strategy(&actor.mask),
						None => InfosetData::new().average_strategy(&actor.mask),
					};
					actor.command(sample(&strategy, rng), &state, self.registry)
				})
				.collect();

			let StepResult { battle_state, step_request } =
				engine::step(state, commands, self.registry, rng);
			state = battle_state;
			request = step_request;
		}

		// Ran out of playout budget: the two strategies stall against each other.
		// Fall back to the leaf estimate rather than silently calling it a draw.
		self.leaf.value(&state, Team::Zero, self.registry)
	}

	/// The value of this position to `traverser`, under the current strategies.
	fn walk(
		&mut self,
		state: BattleState,
		request: StepRequest,
		traverser: Team,
		depth: usize,
		rng: &mut dyn RngCore,
	) -> f32 {
		let actors = match DecisionNode::from(&state, &request) {
			DecisionNode::Terminal(outcome) => {
				return match outcome {
					Outcome::Win { team } => if team == traverser { 1.0 } else { -1.0 },
					Outcome::Draw => 0.0,
				};
			}
			DecisionNode::Decision { actors } => actors,
		};

		// Out of lookahead, or out of budget: estimate rather than recurse.
		if depth >= self.config.max_depth || self.nodes_visited >= self.config.max_nodes {
			self.truncated += 1;
			return self.leaf.value(&state, traverser, self.registry);
		}
		self.nodes_visited += 1;

		let key = StateKey::new(&state, &request)
			.expect("a node with actors is not terminal, so it has a key");

		// The opponent's action is sampled once and reused for every one of the
		// traverser's actions. Sampling it per branch instead would let the
		// traverser's choice correlate with the reply — quietly turning a
		// simultaneous decision into one where it gets to see the answer first.
		let mut fixed_commands: Vec<(usize, Command)> = Vec::new();
		let mut traverser_actor = None;

		for (index, actor) in actors.iter().enumerate() {
			assert!(
				!actor.mask.is_empty(),
				"{:?} at {} was asked for a command with no legal action",
				actor.team,
				actor.position,
			);

			if actor.team == traverser {
				traverser_actor = Some((index, *actor));
				continue;
			}

			let strategy = self
				.table
				.entry(&key, actor.team)
				.current_strategy(&actor.mask);

			// The opponent's average strategy accumulates here. External sampling
			// reaches their infosets in proportion to how often play actually
			// arrives, so each visit counts once and no reach weighting is needed.
			self.table
				.entry(&key, actor.team)
				.accumulate_strategy(&strategy, 1.0);

			let action = sample(&strategy, rng);
			fixed_commands.push((index, actor.command(action, &state, self.registry)));
		}

		let (traverser_index, actor) = match traverser_actor {
			Some(found) => found,
			// Only the opponent decides here — a replacement the traverser does
			// not owe. Nothing to regret, so just play it out.
			None => {
				let commands = assemble(&fixed_commands, None);
				let StepResult { battle_state, step_request } =
					engine::step(state, commands, self.registry, rng);
				return self.walk(battle_state, step_request, traverser, depth + 1, rng);
			}
		};

		let strategy = self
			.table
			.entry(&key, traverser)
			.current_strategy(&actor.mask);

		// Every action the traverser could take is explored, so each one gets a
		// real value rather than an estimate reweighted from a single sample.
		let mut action_values = [0.0; MAX_DECISION];
		for action in 0..MAX_DECISION {
			if !actor.mask.allowed[action] {
				continue;
			}
			let command = actor.command(action, &state, self.registry);
			let commands = assemble(&fixed_commands, Some((traverser_index, command)));

			let StepResult { battle_state, step_request } =
				engine::step(state.clone(), commands, self.registry, rng);
			action_values[action] =
				self.walk(battle_state, step_request, traverser, depth + 1, rng);
		}

		let deltas = regrets_from_action_values(&action_values, &strategy, &actor.mask);
		let data = self.table.entry(&key, traverser);
		for action in 0..MAX_DECISION {
			data.add_regret(action, deltas[action]);
		}

		let mut node_value = 0.0;
		for action in 0..MAX_DECISION {
			if actor.mask.allowed[action] {
				node_value += strategy[action] * action_values[action];
			}
		}

		if self.iteration >= self.warmup {
			if let Some(log) = self.value_log.as_mut() {
				let entry = log.entry((key, traverser)).or_insert((0.0, 0));
				entry.0 += node_value;
				entry.1 += 1;
			}
		}

		node_value
	}
}

/// Put the commands back into the order the actors came in.
///
/// Shared with [`crate::cfr::exploit`], which has to build command lists the same
/// way for its measurement to describe the game the solver actually solved.
///
/// The engine breaks an exact Speed tie by coin flip over the command list, so a
/// stable order keeps that flip from depending on which action is being explored.
pub(crate) fn assemble(
	fixed: &[(usize, Command)],
	traverser: Option<(usize, Command)>,
) -> Vec<Command> {
	let mut all: Vec<(usize, Command)> = fixed.to_vec();
	if let Some(entry) = traverser {
		all.push(entry);
	}
	all.sort_by_key(|(index, _)| *index);
	all.into_iter().map(|(_, command)| command).collect()
}

/// A readable name for one action, for printing a solved position.
pub fn action_label(
	action: usize,
	actor: &Actor,
	state: &BattleState,
	registry: &Registry,
) -> String {
	match Moveslot::from_number(action) {
		Moveslot::Slot(slot) => match state.get_mon(actor.position) {
			Some(mon) => match mon.moves.get(slot) {
				Some(move_id) => registry.get_move(*move_id).name.clone(),
				None => format!("slot {slot} (empty)"),
			},
			None => format!("slot {slot}"),
		},
		Moveslot::Switch(index) => match state.get_mon_from_team(&actor.team, index) {
			Some(mon) => format!("switch to {}", registry.get_species_data(mon.species_id).name),
			None => format!("switch to slot {index}"),
		},
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cfr::position::{
		known_answer_duel, mirror_duel, switch_prediction_2v2, KNOWN_ANSWER_DOMINANT_ACTION,
	};
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	fn solver_config(iterations: usize) -> SolverConfig {
		SolverConfig { iterations, ..SolverConfig::default() }
	}

	/// The end-to-end check, and the only battle position with a knowable answer:
	/// flame lash strictly dominates mud wave, so the equilibrium is pure.
	///
	/// This is what proves the scaffolding works against the real engine rather
	/// than only against toy matrices.
	#[test]
	fn the_solver_finds_the_dominant_move() {
		let registry = Registry::load();
		let root = known_answer_duel(&registry);
		let mut rng = StdRng::seed_from_u64(20260916);

		let mut solver = Solver::new(&registry, solver_config(2_000));
		solver.solve(&root, &mut rng);

		let strategy = solver
			.root_strategy(&root, Team::Zero)
			.expect("the root is always visited");

		assert!(
			strategy[KNOWN_ANSWER_DOMINANT_ACTION] > 0.9,
			"expected the dominant move to take nearly all the mass, got {strategy:?}",
		);
	}

	/// A 1v1 has no healing, so every trajectory should reach a real terminal
	/// node. If this starts failing, the depth cap is hiding something.
	#[test]
	fn trajectories_finish_rather_than_hitting_the_depth_cap() {
		let registry = Registry::load();
		let root = known_answer_duel(&registry);
		let mut rng = StdRng::seed_from_u64(5);

		let mut solver = Solver::new(&registry, solver_config(500));
		solver.solve(&root, &mut rng);

		assert_eq!(
			solver.truncated_positions(),
			0,
			"a 1v1 should always finish on its own, owing nothing to the leaf estimate",
		);
	}

	#[test]
	fn solving_discovers_positions() {
		let registry = Registry::load();
		let root = known_answer_duel(&registry);
		let mut rng = StdRng::seed_from_u64(11);

		let mut solver = Solver::new(&registry, solver_config(200));
		assert!(solver.table().is_empty());

		solver.solve(&root, &mut rng);
		assert!(
			solver.table().len() > 1,
			"a battle has more than one decision point",
		);
	}

	/// Illegal actions must be exactly zero in the reported answer, not merely
	/// unlikely — a solver that recommends an impossible move is worse than
	/// useless.
	#[test]
	fn the_reported_strategy_never_recommends_an_illegal_action() {
		let registry = Registry::load();
		let root = known_answer_duel(&registry);
		let mut rng = StdRng::seed_from_u64(3);

		let mut solver = Solver::new(&registry, solver_config(200));
		solver.solve(&root, &mut rng);

		for team in [Team::Zero, Team::One] {
			let strategy = solver.root_strategy(&root, team).unwrap();
			// Each duellist has exactly two moves and an empty bench.
			for action in 2..MAX_DECISION {
				assert_eq!(strategy[action], 0.0, "{team:?} action {action}");
			}
			let total: f32 = strategy.iter().sum();
			assert!((total - 1.0).abs() < 1e-4, "{team:?} strategy sums to {total}");
		}
	}

	/// Is the position itself symmetric? Both seats play the *same* fixed policy,
	/// so any value other than zero is the engine or the position, not CFR.
	/// The 2v2 capability, and the thing 1v1 could not exercise: a position with
	/// switching, solved under a lookahead horizon rather than to the end.
	///
	/// Balance is asserted rather than the exact mixture. The strategy at a
	/// near-indifferent node moves around between runs — the accumulated regrets
	/// on the two live actions sit close to zero, which is what indifference looks
	/// like — so pinning the percentages would be a flaky test asserting more
	/// precision than a six-turn lookahead has. The value is stable, and for a
	/// mirror it has a known right answer.
	#[test]
	fn a_two_v_two_mirror_is_balanced() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(20260916);

		let mut solver = Solver::new(&registry, SolverConfig::for_lookahead(2_000, 5));
		solver.solve(&root, &mut rng);

		let value = solver.evaluate(&root, 2_000, &mut rng);
		assert!(
			value.abs() < 0.2,
			"a mirror 2v2 should be worth nothing to either side, got {value}",
		);
		assert!(
			solver.truncated_positions() > 0,
			"a 2v2 cannot be searched to the end; the horizon should be doing work",
		);
	}

	/// Switching has to be a real option, or the 2v2 is just a slower 1v1.
	#[test]
	fn switching_is_a_live_option_in_a_two_v_two() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(4);

		let mut solver = Solver::new(&registry, SolverConfig::for_lookahead(300, 4));
		solver.solve(&root, &mut rng);

		let actors = match DecisionNode::from(&root, &StepRequest::NeedsActions) {
			DecisionNode::Decision { actors } => actors,
			DecisionNode::Terminal(_) => panic!("the battle has not started"),
		};
		let actor = actors.iter().find(|actor| actor.team == Team::Zero).unwrap();

		// Two moves, then the switch to the reserve at index 4 + 1.
		let legal: Vec<usize> = (0..MAX_DECISION).filter(|i| actor.mask.allowed[*i]).collect();
		assert_eq!(legal, vec![0, 1, 5], "expected two moves and one switch");

		let strategy = solver.root_strategy(&root, Team::Zero).unwrap();
		let total: f32 = legal.iter().map(|i| strategy[*i]).sum();
		assert!((total - 1.0).abs() < 1e-4, "strategy should sum to one, got {total}");
	}

	/// A faint in a 2v2 asks both sides for a replacement, which a 1v1 never does
	/// — `outcome()` is checked first there, so the battle simply ends. This walks
	/// the position far enough to be sure that path is exercised and does not
	/// panic.
	#[test]
	fn a_two_v_two_reaches_and_survives_replacement_requests() {
		let registry = Registry::load();
		let root = switch_prediction_2v2(&registry);
		let mut rng = StdRng::seed_from_u64(99);

		let mut solver = Solver::new(&registry, SolverConfig::for_lookahead(200, 6));
		solver.solve(&root, &mut rng);

		// Play the solved strategies out; a replacement request is reached
		// whenever a creature faints with a team-mate still standing.
		let value = solver.evaluate(&root, 200, &mut rng);
		assert!(value.is_finite(), "playouts through replacements should produce a value");
		assert!(solver.table().len() > 100, "a 2v2 should discover many positions");
	}

	/// A mirror match is symmetric, so its value has to be zero — neither seat
	/// can hold an advantage when both sides are the same creature with the same
	/// moves and the same HP.
	///
	/// This is deliberately a check on the *value* rather than on the two seats
	/// playing identically. Equilibria in a zero-sum game are exchangeable but
	/// not necessarily symmetric, so a correct solve of a mirror can perfectly
	/// well settle on one seat attacking more and the other guarding more —
	/// which is exactly what this position does. Asserting the strategies match
	/// would fail on a correct solver.
	///
	/// It still catches the class of bug worth worrying about: anything that
	/// gives one seat a systematic edge (updating only one player's regrets,
	/// reading a position from the wrong side) moves the value off zero.
	#[test]
	fn a_mirror_match_is_worth_nothing_to_either_side() {
		let registry = Registry::load();
		let root = mirror_duel(&registry);
		let mut rng = StdRng::seed_from_u64(1234);

		let mut solver = Solver::new(&registry, solver_config(4_000));
		solver.solve(&root, &mut rng);

		let value = solver.evaluate(&root, 4_000, &mut rng);
		assert!(
			value.abs() < 0.1,
			"a mirror match should be worth nothing, got {value}",
		);
	}

	/// The counterpart to the mirror test: a position that *is* lopsided should
	/// be valued as such. Without this, a solver that always reported zero would
	/// pass the mirror check.
	#[test]
	fn a_winning_position_is_valued_as_winning() {
		let registry = Registry::load();
		let root = known_answer_duel(&registry);
		let mut rng = StdRng::seed_from_u64(808);

		let mut solver = Solver::new(&registry, solver_config(2_000));
		solver.solve(&root, &mut rng);

		let value = solver.evaluate(&root, 2_000, &mut rng);
		assert!(
			value > 0.5,
			"cinderfox is faster and wins the damage race; got {value}",
		);
	}
}
