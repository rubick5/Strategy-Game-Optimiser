//! Non-volatile statuses, expressed as hooks.
//!
//! Compare with the old `engine::queue_non_volatile_status`, which scanned every
//! creature on the field once per turn and matched on its status. Here, a burned
//! creature *subscribes* to `Residual` and `ModifyStat`; an unburned one
//! subscribes to nothing and costs nothing.

use std::collections::VecDeque;

use rand::{Rng, RngCore};

use crate::battle::event::Event;
use crate::battle::hooks::effects::fraction_of_max;
use crate::battle::hooks::handler::{HookCtx, HookDef};
use crate::battle::hooks::order;
use crate::battle::hooks::query::{Query, QueryKind};
use crate::battle::hooks::trigger::{Trigger, TriggerKind};
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::volatile::{Volatile, VolatileKind};
use crate::model::speciesdata::Stat;

/// Poison costs 1/8 max HP a turn.
const POISON_FRACTION: u32 = 8;
/// Bad poison, for now, matches the old engine's 1/16. Real Bad Poison ramps
/// 1/16, 2/16, 3/16 ...; that needs a per-creature counter to be added first.
const BAD_POISON_FRACTION: u32 = 16;
/// Burn costs 1/16 max HP a turn.
const BURN_FRACTION: u32 = 16;
/// Chance per turn that paralysis costs the creature its move entirely.
const FULL_PARALYSIS_CHANCE: u8 = 25;

/// The hooks a given status installs.
pub fn hooks(status: NonVolatileStatus) -> &'static [HookDef] {
	match status {
		NonVolatileStatus::NoStatus => &[],
		NonVolatileStatus::Poison => POISON,
		NonVolatileStatus::BadPoison => BAD_POISON,
		NonVolatileStatus::Burn => BURN,
		NonVolatileStatus::Paralysis => PARALYSIS,
	}
}

static POISON: &[HookDef] = &[
	HookDef::reactive(TriggerKind::Residual, order::POISON, poison_residual),
	HookDef::query(QueryKind::TryApplyStatus, order::IMMUNITY, block_second_status),
];

static BAD_POISON: &[HookDef] = &[
	HookDef::reactive(TriggerKind::Residual, order::POISON, bad_poison_residual),
	HookDef::query(QueryKind::TryApplyStatus, order::IMMUNITY, block_second_status),
];

static BURN: &[HookDef] = &[
	HookDef::reactive(TriggerKind::Residual, order::BURN, burn_residual),
	HookDef::query(QueryKind::ModifyStat, order::MULTIPLIER, burn_halves_attack),
	HookDef::query(QueryKind::TryApplyStatus, order::IMMUNITY, block_second_status),
];

static PARALYSIS: &[HookDef] = &[
	HookDef::query(QueryKind::ModifyStat, order::MULTIPLIER, paralysis_halves_speed),
	HookDef::reactive(TriggerKind::TurnStart, order::DEFAULT, full_paralysis_roll),
	HookDef::query(QueryKind::TryApplyStatus, order::IMMUNITY, block_second_status),
];

/// Shared body for every "chip a fraction of max HP" status.
fn chip_owner(ctx: &HookCtx, events: &mut VecDeque<Event>, denominator: u32) {
	let pos = match ctx.source.owner() {
		Some(pos) => pos,
		None => return,
	};
	let mon = match ctx.battle_state.get_mon(pos) {
		Some(mon) => mon,
		None => return,
	};
	events.push_back(Event::DealDamage {
		amount: fraction_of_max(mon.max_hp, denominator),
		target: pos,
		source: None,
	});
}

fn poison_residual(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	chip_owner(ctx, events, POISON_FRACTION);
}

fn bad_poison_residual(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	chip_owner(ctx, events, BAD_POISON_FRACTION);
}

fn burn_residual(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	chip_owner(ctx, events, BURN_FRACTION);
}

/// Burn halves the burned creature's Attack.
///
/// This lives in the hook layer rather than in `CreatureState::get_stat` on
/// purpose: `get_stat` is also what the RL encoder reads, and the encoder wants
/// the creature's own stat, not a combat-time modified one.
///
/// Only Attack, not Special Attack — special moves scale off `SpecialAttack`
/// now, so a burn no longer weakens them.
fn burn_halves_attack(ctx: &HookCtx, query: &mut Query) {
	if let Query::ModifyStat { pos, stat: Stat::Attack, value } = query {
		if ctx.source.owner() == Some(*pos) {
			*value = (*value / 2).max(1);
		}
	}
}

/// The other half of paralysis: a flat chance to lose the turn.
///
/// Rolled on `TurnStart` and expressed as the `Immobilised` volatile, exactly as
/// confusion does, so the veto itself stays a pure query.
fn full_paralysis_roll(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, rng: &mut dyn RngCore) {
	let pos = match ctx.source.owner() {
		Some(pos) => pos,
		None => return,
	};
	if rng.random_range(1..=100) <= FULL_PARALYSIS_CHANCE {
		events.push_back(Event::ApplyVolatile {
			target: pos,
			volatile: Volatile::new(VolatileKind::Immobilised),
		});
	}
}

/// Paralysis halves Speed.
fn paralysis_halves_speed(ctx: &HookCtx, query: &mut Query) {
	if let Query::ModifyStat { pos, stat: Stat::Speed, value } = query {
		if ctx.source.owner() == Some(*pos) {
			*value = (*value / 2).max(1);
		}
	}
}

/// A creature that already has a non-volatile status cannot take another one.
///
/// Worth noticing how this reads: the *existing* status is the thing that
/// blocks, so it is the existing status that owns the rule. Nothing has to look
/// up "does the target already have a status?" at the call site.
///
/// `NoStatus` is exempt — applying it is how an effect *cures*, and a status
/// must not block its own removal (Natural Cure depends on this).
fn block_second_status(ctx: &HookCtx, query: &mut Query) {
	if let Query::TryApplyStatus { target, status, allowed } = query {
		if *status == NonVolatileStatus::NoStatus {
			return;
		}
		if ctx.source.owner() == Some(*target) {
			*allowed = false;
		}
	}
}
