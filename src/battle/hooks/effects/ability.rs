//! Abilities, expressed as hooks.
//!
//! Five abilities, deliberately chosen so each one exercises a different part of
//! the hook system. Note that none of them required an engine change — the
//! engine was already broadcasting every moment they needed.

use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::event::Event;
use crate::battle::hooks::effects::fraction_of_max;
use crate::battle::hooks::handler::{HookCtx, HookDef};
use crate::battle::hooks::order;
use crate::battle::hooks::query::{Query, QueryKind};
use crate::battle::hooks::trigger::{Trigger, TriggerKind};
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::weather::{TimedWeather, Weather};
use crate::model::ability::AbilityId;
use crate::model::speciesdata::Stat;
use crate::model::typing::{Effectiveness, Type};

/// Rough Skin costs the attacker 1/8 of their max HP.
const ROUGH_SKIN_FRACTION: u32 = 8;
/// How long a summoned sandstorm lasts.
const SAND_STREAM_TURNS: usize = 5;

/// The hooks a given ability installs.
pub fn hooks(ability: AbilityId) -> &'static [HookDef] {
	match ability {
		AbilityId::Levitate => LEVITATE,
		AbilityId::RoughSkin => ROUGH_SKIN,
		AbilityId::Guts => GUTS,
		AbilityId::SandStream => SAND_STREAM,
		AbilityId::NaturalCure => NATURAL_CURE,
	}
}

// ---------------------------------------------------------------------------
// Levitate — a query that zeroes a value before it is used.
// ---------------------------------------------------------------------------

static LEVITATE: &[HookDef] = &[HookDef::query(
	QueryKind::ModifyEffectiveness,
	order::IMMUNITY,
	levitate_blocks_ground,
)];

/// Ground-type moves do nothing to the holder.
///
/// This hooks the *effectiveness* query rather than the damage one, which is
/// what makes it a true immunity: `execute_move` checks for a zero multiplier
/// before rolling secondary effects, so a Ground move cannot burn or poison a
/// Levitate holder on a hit that did nothing.
///
/// Runs at `IMMUNITY` order, i.e. first, so nothing downstream can resurrect it.
fn levitate_blocks_ground(ctx: &HookCtx, query: &mut Query) {
	if let Query::ModifyEffectiveness { target, move_id, effectiveness, .. } = query {
		if ctx.source.owner() != Some(*target) {
			return;
		}
		if ctx.registry.get_move(*move_id).element == Type::Ground {
			*effectiveness = Effectiveness::IMMUNE;
		}
	}
}

// ---------------------------------------------------------------------------
// Rough Skin — a reactive hook that queues a new event.
// ---------------------------------------------------------------------------

static ROUGH_SKIN: &[HookDef] = &[HookDef::reactive(
	TriggerKind::AfterDamage,
	order::DEFAULT,
	rough_skin_recoil,
)];

/// Attackers that made contact take 1/8 of their own max HP.
///
/// This is the reason `Event::DealDamage` carries a `DamageSource`: the hook
/// needs both who hit it and what they hit it with.
fn rough_skin_recoil(ctx: &HookCtx, trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let (target, amount, source) = match trigger {
		Trigger::AfterDamage { target, amount, source } => (target, amount, source),
		_ => return,
	};
	// Only when the holder is the one that got hit, and actually took something.
	if ctx.source.owner() != Some(*target) || *amount == 0 {
		return;
	}
	let source = match source {
		Some(source) => source,
		None => return, // weather or status chip, nobody to punish
	};
	if !ctx.registry.get_move(source.move_id).flags.contact {
		return;
	}
	let attacker = match ctx.battle_state.get_mon(source.attacker) {
		Some(attacker) => attacker,
		None => return,
	};
	if !attacker.is_alive() {
		return;
	}
	events.push_back(Event::DealDamage {
		amount: fraction_of_max(attacker.max_hp, ROUGH_SKIN_FRACTION),
		target: source.attacker,
		source: None,
	});
}

// ---------------------------------------------------------------------------
// Guts — a query that has to cooperate with another query.
// ---------------------------------------------------------------------------

static GUTS: &[HookDef] = &[HookDef::query(
	QueryKind::ModifyStat,
	// Deliberately after burn's MULTIPLIER halving, so it can undo it.
	order::MULTIPLIER + 10,
	guts_boosts_attack,
)];

/// Attack ×1.5 while afflicted with any non-volatile status.
///
/// The ×3 case is not a typo. Burn's own hook has already halved Attack by the
/// time this runs, so ×3 lands on the same ×1.5 as every other status — which is
/// how Guts behaves in the games, where it ignores the burn drop.
///
/// Worth noticing that this reads only its *own* holder's state via
/// `source.owner()`. It never inspects the other creature's ability, which is
/// the cross-checking the hook system exists to avoid.
fn guts_boosts_attack(ctx: &HookCtx, query: &mut Query) {
	if let Query::ModifyStat { pos, stat: Stat::Attack, value } = query {
		let owner = match ctx.source.owner() {
			Some(owner) if owner == *pos => owner,
			_ => return,
		};
		let mon = match ctx.battle_state.get_mon(owner) {
			Some(mon) => mon,
			None => return,
		};
		match mon.non_vol_status {
			NonVolatileStatus::NoStatus => {}
			NonVolatileStatus::Burn => *value = (*value * 3).max(1),
			_ => *value = (*value * 3 / 2).max(1),
		}
	}
}

// ---------------------------------------------------------------------------
// Sand Stream — a reactive hook on arrival.
// ---------------------------------------------------------------------------

static SAND_STREAM: &[HookDef] = &[HookDef::reactive(
	TriggerKind::SwitchIn,
	order::DEFAULT,
	sand_stream_summons,
)];

/// Sets a sandstorm when the holder switches in.
///
/// Only on switch-in: there is no battle-start trigger yet, so a Sand Stream
/// creature that begins the battle already on the field will not summon until it
/// is switched out and back in. Adding `TriggerKind::BattleStart` and
/// broadcasting it on the first `step` would close that gap.
fn sand_stream_summons(ctx: &HookCtx, trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let pos = match trigger {
		Trigger::SwitchIn { pos } => pos,
		_ => return,
	};
	if ctx.source.owner() != Some(*pos) {
		return;
	}
	events.push_back(Event::SetWeather {
		weather: Some(TimedWeather {
			weather: Weather::Sandstorm,
			turns_left: SAND_STREAM_TURNS,
		}),
	});
}

// ---------------------------------------------------------------------------
// Natural Cure — a reactive hook on departure.
// ---------------------------------------------------------------------------

static NATURAL_CURE: &[HookDef] = &[HookDef::reactive(
	TriggerKind::SwitchOut,
	order::DEFAULT,
	natural_cure_clears_status,
)];

/// Clears the holder's status as it leaves.
///
/// `SwitchOut` fires before the field pointer moves, so the holder is still at
/// `pos` and the cure lands on the right creature. Curing is expressed as
/// applying `NoStatus` — which is why the "already has a status" veto has to
/// make an exception for it.
fn natural_cure_clears_status(ctx: &HookCtx, trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let pos = match trigger {
		Trigger::SwitchOut { pos } => pos,
		_ => return,
	};
	if ctx.source.owner() != Some(*pos) {
		return;
	}
	let mon = match ctx.battle_state.get_mon(*pos) {
		Some(mon) => mon,
		None => return,
	};
	if mon.non_vol_status.is_afflicted() {
		events.push_back(Event::ApplyNonVolStatus {
			status: NonVolatileStatus::NoStatus,
			target: *pos,
		});
	}
}
