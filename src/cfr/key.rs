//! Identifying a decision point.
//!
//! With known teams there is no private information in this game — only
//! simultaneous action selection — so a player's information set *is* the public
//! battle state. That is what makes a matchup solvable in isolation, and it is
//! why the key here is the position itself rather than anything learned or
//! encoded.
//!
//! Two things have to be got right or the table silently splits one infoset into
//! several, and CFR converges smoothly to the equilibrium of a game that is not
//! this one:
//!
//! 1. **Volatiles are insertion-ordered.** A creature Taunted-then-Seeded and one
//!    Seeded-then-Taunted are the same position, but the underlying `Vec` compares
//!    unequal. [`Volatiles::canonicalised`] fixes the order here, at the key,
//!    rather than in `Volatiles::add` — the engine's own iteration order is a
//!    display and hook-dispatch concern and is deliberately left alone.
//! 2. **The same state can mean two different decisions.** A position awaiting
//!    replacements is not the position awaiting actions, even when every creature
//!    is identical, because the legal actions differ. [`NodeKind`] carries that
//!    distinction into the key.
//!
//! The key owns a whole `BattleState` rather than a hash digest. That is a real
//! memory cost, paid on purpose: a digest collision would merge two unrelated
//! positions and produce a confidently wrong answer with nothing to notice it by.
//! Worth revisiting once the table is large enough to matter.

use crate::battle::engine::engine::StepRequest;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::roster::RosterId;
use crate::battle::state::TEAM_SIZE;

/// Which kind of decision the players are being asked for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
	/// An ordinary turn: every active creature picks a move or a switch.
	Actions,
	/// One or both sides must send something in after a faint. The positions are
	/// sorted, so the engine's collection order cannot produce two keys for one
	/// position.
	Replacements(Vec<PositionId>),
}

impl NodeKind {
	/// `None` at a finished battle, which has no decision and so needs no key.
	pub fn from_request(request: &StepRequest) -> Option<Self> {
		match request {
			StepRequest::NeedsActions => Some(NodeKind::Actions),
			StepRequest::NeedsReplacements(positions) => {
				let mut positions = positions.clone();
				positions.sort_by_key(|p| p.0);
				Some(NodeKind::Replacements(positions))
			}
			StepRequest::Finished(_) => None,
		}
	}
}

/// A decision point: the position, plus what is being decided.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateKey {
	state: BattleState,
	node: NodeKind,
}

impl StateKey {
	/// `None` at a finished battle.
	pub fn new(state: &BattleState, request: &StepRequest) -> Option<Self> {
		Some(StateKey {
			state: canonicalise(state),
			node: NodeKind::from_request(request)?,
		})
	}

	pub fn state(&self) -> &BattleState {
		&self.state
	}

	pub fn node(&self) -> &NodeKind {
		&self.node
	}
}

/// The same position, in a form where structural equality is state equality.
fn canonicalise(state: &BattleState) -> BattleState {
	let mut state = state.clone();
	for index in 0..(TEAM_SIZE * 2) {
		if let Some(mon) = state.roster.get_mut_mon(RosterId(index)) {
			mon.volatiles = mon.volatiles.canonicalised();
		}
	}
	state
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::volatile::{Volatile, VolatileKind};
	use crate::battle::state::Outcome;
	use crate::battle::state::Team;
	use crate::cfr::position::known_answer_duel;
	use crate::model::registry::Registry;

	/// The bug this whole module exists to prevent: two identical positions that
	/// hash apart because the conditions were applied in a different order.
	#[test]
	fn volatile_order_does_not_change_the_key() {
		let registry = Registry::load();
		let base = known_answer_duel(&registry);

		let mut taunt_first = base.clone();
		let mon = taunt_first.get_mut_mon(PositionId(0)).unwrap();
		mon.volatiles.add(Volatile::lasting(VolatileKind::Taunt, 3));
		mon.volatiles.add(Volatile::new(VolatileKind::LeechSeed));

		let mut seed_first = base.clone();
		let mon = seed_first.get_mut_mon(PositionId(0)).unwrap();
		mon.volatiles.add(Volatile::new(VolatileKind::LeechSeed));
		mon.volatiles.add(Volatile::lasting(VolatileKind::Taunt, 3));

		assert_ne!(
			taunt_first, seed_first,
			"the raw states differ by insertion order — otherwise this test proves nothing"
		);
		assert_eq!(
			StateKey::new(&taunt_first, &StepRequest::NeedsActions),
			StateKey::new(&seed_first, &StepRequest::NeedsActions),
			"same position, same key, whatever order the conditions arrived in"
		);
	}

	#[test]
	fn a_different_position_is_a_different_key() {
		let registry = Registry::load();
		let base = known_answer_duel(&registry);

		let mut hurt = base.clone();
		hurt.get_mut_mon(PositionId(0)).unwrap().current_hp -= 1;

		assert_ne!(
			StateKey::new(&base, &StepRequest::NeedsActions),
			StateKey::new(&hurt, &StepRequest::NeedsActions),
			"one point of HP is a different position"
		);
	}

	/// The legal actions differ between these two, so they must not share regrets.
	#[test]
	fn actions_and_replacements_never_collide() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);

		assert_ne!(
			StateKey::new(&state, &StepRequest::NeedsActions),
			StateKey::new(&state, &StepRequest::NeedsReplacements(vec![PositionId(0)])),
		);
	}

	#[test]
	fn replacement_position_order_does_not_change_the_key() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);

		let one_way = StepRequest::NeedsReplacements(vec![PositionId(0), PositionId(1)]);
		let other_way = StepRequest::NeedsReplacements(vec![PositionId(1), PositionId(0)]);

		assert_eq!(
			StateKey::new(&state, &one_way),
			StateKey::new(&state, &other_way),
		);
	}

	#[test]
	fn a_finished_battle_has_no_key() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);
		let finished = StepRequest::Finished(Outcome::Win { team: Team::Zero });

		assert!(StateKey::new(&state, &finished).is_none());
	}
}
