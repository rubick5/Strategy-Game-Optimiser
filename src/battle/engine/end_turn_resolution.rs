//! The end-of-turn phases.
//!
//! This file used to hold a hand-written `END_OF_TURN_ORDER` array and a `match`
//! with one arm per effect — plus a `todo!()` that would have panicked the first
//! time anything called it. Every new end-of-turn effect meant editing both the
//! array and the match.
//!
//! Now there is nothing to edit. An effect declares `TriggerKind::Residual` or
//! `TriggerKind::TurnEnd` with an order constant from
//! [`hooks::order`](crate::battle::hooks::order) and the table sorts it into
//! place. These two functions are all that is left: they announce the phase and
//! let whoever cares respond.

use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::event::Event;
use crate::battle::hooks::{HookTable, Trigger};
use crate::battle::state::battle_state::BattleState;
use crate::model::registry::Registry;

/// The residual phase: chip damage, healing, delayed attacks.
///
/// Ordering within the phase comes from the effects themselves — see the
/// residual constants in [`hooks::order`](crate::battle::hooks::order), which
/// preserve the sequence the old `END_OF_TURN_ORDER` was documenting.
pub fn resolve_residual(
	battle_state: &BattleState,
	registry: &Registry,
	hooks: &HookTable,
	events: &mut VecDeque<Event>,
	rng: &mut dyn RngCore,
) {
	hooks.dispatch(Trigger::Residual, battle_state, registry, events, rng);
}

/// The very end of the turn: countdowns and expiry, after residuals have run.
pub fn resolve_turn_end(
	battle_state: &BattleState,
	registry: &Registry,
	hooks: &HookTable,
	events: &mut VecDeque<Event>,
	rng: &mut dyn RngCore,
) {
	hooks.dispatch(Trigger::TurnEnd, battle_state, registry, events, rng);
}
