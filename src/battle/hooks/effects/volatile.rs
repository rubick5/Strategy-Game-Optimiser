//! Volatile conditions, expressed as hooks.
//!
//! Note where the work is split. Anything that *rolls* happens on
//! `TurnStart` (a reactive hook, which has RNG); anything that *decides* happens
//! in a query (which does not). Confusion is the clearest case: it rolls at turn
//! start and sets `Immobilised`, and the actual "you don't get to move" is a
//! deterministic read of that flag. Keeping the randomness on one side of that
//! line is what lets queries stay pure.
//!
//! Substitute is the one condition not implemented here — it intercepts damage
//! before it reaches HP, which no hook can do, so it lives in
//! `engine::execute_event` alongside the damage it absorbs.

use std::collections::VecDeque;

use rand::{Rng, RngCore};

use crate::battle::engine::calculate_damage::calculate_damage;
use crate::battle::event::Event;
use crate::battle::hooks::effects::fraction_of_max;
use crate::battle::hooks::handler::{HookCtx, HookDef};
use crate::battle::hooks::order;
use crate::battle::hooks::query::{Query, QueryKind};
use crate::battle::hooks::trigger::{Trigger, TriggerKind};
use crate::battle::state::field::PositionId;
use crate::battle::state::volatile::{Volatile, VolatileKind};
use crate::model::effect::Effect;
use crate::model::pmove::MoveType;
use crate::model::speciesdata::Stat;

/// Chance per turn that confusion makes a creature hit itself instead.
const CONFUSION_SELF_HIT_CHANCE: u8 = 33;
/// Power of the typeless self-hit.
const CONFUSION_SELF_HIT_POWER: u32 = 40;
/// Leech Seed moves this fraction of the seeded creature's max HP each turn.
const LEECH_SEED_FRACTION: u32 = 8;

/// The hooks a given volatile installs.
pub fn hooks(kind: VolatileKind) -> &'static [HookDef] {
	match kind {
		VolatileKind::Confusion => CONFUSION,
		VolatileKind::Flinch => FLINCH,
		VolatileKind::Taunt => TAUNT,
		VolatileKind::LeechSeed => LEECH_SEED,
		VolatileKind::Protect => PROTECT,
		VolatileKind::Immobilised => IMMOBILISED,
		VolatileKind::ProtectStreak => PROTECT_STREAK,
		// Substitute is handled in the engine's damage path, not here.
		VolatileKind::Substitute => &[],
		// Pure storage, read by the bad-poison residual hook.
		VolatileKind::ToxicCounter => &[],
	}
}

// ---------------------------------------------------------------------------
// Confusion — rolls at turn start, is read deterministically later.
// ---------------------------------------------------------------------------

static CONFUSION: &[HookDef] = &[HookDef::reactive(
	TriggerKind::TurnStart,
	order::DEFAULT,
	confusion_roll,
)];

fn confusion_roll(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, rng: &mut dyn RngCore) {
	let pos = match ctx.source.owner() {
		Some(pos) => pos,
		None => return,
	};
	let mon = match ctx.battle_state.get_mon(pos) {
		Some(mon) => mon,
		None => return,
	};
	if rng.random_range(1..=100) > CONFUSION_SELF_HIT_CHANCE {
		return;
	}

	// A typeless hit using the creature's own Attack against its own Defense.
	// No chart, no STAB, and no `DamageSource` — so it cannot trigger Rough Skin
	// or anything else that punishes an attacker.
	let attack = mon.get_stat(Stat::Attack, ctx.registry);
	let defense = mon.get_stat(Stat::Defense, ctx.registry);
	events.push_back(Event::DealDamage {
		amount: calculate_damage(attack, defense, CONFUSION_SELF_HIT_POWER),
		target: pos,
		source: None,
	});
	events.push_back(Event::ApplyVolatile {
		target: pos,
		volatile: Volatile::new(VolatileKind::Immobilised),
	});
}

// ---------------------------------------------------------------------------
// The three "you lose your turn" conditions — all the same deterministic veto.
// ---------------------------------------------------------------------------

static IMMOBILISED: &[HookDef] = &[HookDef::query(
	QueryKind::TryMove,
	order::IMMUNITY,
	block_own_move,
)];

static FLINCH: &[HookDef] = &[HookDef::query(
	QueryKind::TryMove,
	order::IMMUNITY,
	block_own_move,
)];

/// The holder does not get to act.
///
/// This is `TryMove`'s first real subscriber. Before volatiles existed the query
/// was machinery with nothing plugged into it.
fn block_own_move(ctx: &HookCtx, query: &mut Query) {
	if let Query::TryMove { user, allowed, .. } = query {
		if ctx.source.owner() == Some(*user) {
			*allowed = false;
		}
	}
}

// ---------------------------------------------------------------------------
// Taunt — blocks status moves only, and needs no randomness at all.
// ---------------------------------------------------------------------------

static TAUNT: &[HookDef] = &[HookDef::query(
	QueryKind::TryMove,
	order::IMMUNITY,
	taunt_blocks_status_moves,
)];

fn taunt_blocks_status_moves(ctx: &HookCtx, query: &mut Query) {
	if let Query::TryMove { user, move_id, allowed } = query {
		if ctx.source.owner() != Some(*user) {
			return;
		}
		if ctx.registry.get_move(*move_id).move_type == MoveType::Status {
			*allowed = false;
		}
	}
}

// ---------------------------------------------------------------------------
// Protect — blocks moves aimed at the holder, rather than the holder's own move.
// ---------------------------------------------------------------------------

static PROTECT: &[HookDef] = &[HookDef::query(
	QueryKind::TryHit,
	order::IMMUNITY,
	protect_blocks_incoming,
)];

fn protect_blocks_incoming(ctx: &HookCtx, query: &mut Query) {
	if let Query::TryHit { attacker, target, allowed, .. } = query {
		// Self-targeting moves are not "incoming", so Protect must not block the
		// holder from acting on itself.
		if ctx.source.owner() == Some(*target) && attacker != target {
			*allowed = false;
		}
	}
}

// ---------------------------------------------------------------------------
// Protect streak — the counter that makes repeated Protects fail.
// ---------------------------------------------------------------------------

static PROTECT_STREAK: &[HookDef] = &[HookDef::reactive(
	TriggerKind::AfterMove,
	order::DEFAULT,
	protect_streak_resets_on_any_other_move,
)];

/// Using anything other than Protect breaks the chain.
///
/// The counter itself is incremented where the roll happens, in
/// `effect_handler`. This hook only handles the reset, because that depends on
/// what the creature did *instead*, which is exactly what `AfterMove` reports.
fn protect_streak_resets_on_any_other_move(ctx: &HookCtx, trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let (user, move_id) = match trigger {
		Trigger::AfterMove { user, move_id } => (user, move_id),
		_ => return,
	};
	if ctx.source.owner() != Some(*user) {
		return;
	}
	let was_protect = ctx
		.registry
		.get_move(*move_id)
		.effects
		.iter()
		.any(|effect| matches!(effect, Effect::Protect));
	if !was_protect {
		events.push_back(Event::RemoveVolatile {
			target: *user,
			kind: VolatileKind::ProtectStreak,
		});
	}
}

// ---------------------------------------------------------------------------
// Leech Seed — drains the holder and feeds whoever planted it.
// ---------------------------------------------------------------------------

static LEECH_SEED: &[HookDef] = &[HookDef::reactive(
	TriggerKind::Residual,
	order::BINDING_MOVES,
	leech_seed_drain,
)];

/// The seeder's position is stored in the volatile's `value`, which is how the
/// hook knows where to send the stolen HP without searching for it.
fn leech_seed_drain(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let pos = match ctx.source.owner() {
		Some(pos) => pos,
		None => return,
	};
	let mon = match ctx.battle_state.get_mon(pos) {
		Some(mon) => mon,
		None => return,
	};
	let amount = fraction_of_max(mon.max_hp, LEECH_SEED_FRACTION);
	events.push_back(Event::DealDamage { amount, target: pos, source: None });

	let seeder = PositionId(mon.volatiles.value(VolatileKind::LeechSeed) as usize);
	// Only heal a seeder that is still out and still alive.
	if let Some(other) = ctx.battle_state.get_mon(seeder) {
		if other.is_alive() && seeder != pos {
			events.push_back(Event::Heal { amount, target: seeder });
		}
	}
}
