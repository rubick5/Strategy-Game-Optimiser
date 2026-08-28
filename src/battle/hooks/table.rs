//! The subscription index — the thing that makes hooks cheap.

use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::event::Event;
use crate::battle::hooks::handler::{HookCtx, HookKind, QueryFn, ReactiveFn};
use crate::battle::hooks::providers;
use crate::battle::hooks::query::{Query, QueryKind, QUERY_KIND_COUNT};
use crate::battle::hooks::source::HookSource;
use crate::battle::hooks::trigger::{Trigger, TriggerKind, TRIGGER_KIND_COUNT};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;

/// One live subscription: a handler plus the context needed to order and run it.
#[derive(Clone, Copy)]
pub struct Subscription<F: Copy + 'static> {
	pub source: HookSource,
	pub order: i16,
	/// The owner's Speed at the time the table was built, cached so that sorting
	/// does not have to reach back into the registry on every broadcast.
	pub speed: u32,
	pub run: F,
}

/// Which effects are currently listening to what.
///
/// The point of this type: dispatching `Residual` walks *only* the handlers that
/// asked for `Residual`. Nothing iterates the field asking "is this one burned?
/// does it have Levitate? is it holding Leftovers?". Adding a hundred abilities
/// adds zero work to any moment none of them subscribe to.
///
/// The table is derived state — it is rebuilt from the [`BattleState`], never
/// stored in it. That matters because handlers are `fn` pointers, which do not
/// serialise; `BattleState` stays exactly as serialisable as it was before.
pub struct HookTable {
	reactive: [Vec<Subscription<ReactiveFn>>; TRIGGER_KIND_COUNT],
	queries: [Vec<Subscription<QueryFn>>; QUERY_KIND_COUNT],
	dirty: bool,
}

impl Default for HookTable {
	fn default() -> Self {
		Self::new()
	}
}

impl HookTable {
	/// A new, empty table. Starts dirty, so the first `refresh` builds it.
	pub fn new() -> Self {
		HookTable {
			reactive: std::array::from_fn(|_| Vec::new()),
			queries: std::array::from_fn(|_| Vec::new()),
			dirty: true,
		}
	}

	/// Mark the table stale. Call this whenever something that *decides
	/// subscriptions* changes: a switch, a status landing, a faint, a weather
	/// change. Plain damage does not need it.
	#[inline]
	pub fn invalidate(&mut self) {
		self.dirty = true;
	}

	#[inline]
	pub fn is_dirty(&self) -> bool {
		self.dirty
	}

	/// Rebuild the index if it is stale, otherwise do nothing.
	///
	/// Rebuilding is O(creatures on the field), and only happens when the field
	/// actually changed — a few times a turn, not a few times an event.
	pub fn refresh(&mut self, battle_state: &BattleState, registry: &Registry) {
		if !self.dirty {
			return;
		}
		for slot in self.reactive.iter_mut() {
			slot.clear();
		}
		for slot in self.queries.iter_mut() {
			slot.clear();
		}

		for (source, defs, speed) in providers::collect(battle_state, registry) {
			for def in defs {
				match def.kind {
					HookKind::Reactive { trigger, run } => {
						self.reactive[trigger.index()].push(Subscription {
							source,
							order: def.order,
							speed,
							run,
						});
					}
					HookKind::Query { query, run } => {
						self.queries[query.index()].push(Subscription {
							source,
							order: def.order,
							speed,
							run,
						});
					}
				}
			}
		}

		// Declared order first, then Speed (faster acts first), matching how
		// simultaneous effects resolve in the games.
		for slot in self.reactive.iter_mut() {
			slot.sort_by(|a, b| a.order.cmp(&b.order).then(b.speed.cmp(&a.speed)));
		}
		for slot in self.queries.iter_mut() {
			slot.sort_by(|a, b| a.order.cmp(&b.order).then(b.speed.cmp(&a.speed)));
		}

		self.dirty = false;
	}

	/// How many handlers are listening to a moment. Diagnostics and tests.
	pub fn reactive_count(&self, kind: TriggerKind) -> usize {
		self.reactive[kind.index()].len()
	}

	/// How many handlers are listening to a query. Diagnostics and tests.
	pub fn query_count(&self, kind: QueryKind) -> usize {
		self.queries[kind.index()].len()
	}

	// -----------------------------------------------------------------------
	// Broadcasting
	// -----------------------------------------------------------------------

	/// Announce that something happened. Every subscriber gets a look and may
	/// append events to the queue.
	pub fn dispatch(
		&self,
		trigger: Trigger,
		battle_state: &BattleState,
		registry: &Registry,
		events: &mut VecDeque<Event>,
		rng: &mut dyn RngCore,
	) {
		debug_assert!(
			!self.dirty,
			"dispatched on a stale HookTable - call refresh() first"
		);
		for sub in self.reactive[trigger.kind().index()].iter() {
			let ctx = HookCtx {
				source: sub.source,
				battle_state,
				registry,
			};
			(sub.run)(&ctx, &trigger, events, rng);
		}
	}

	/// Fold a value through every subscriber, in order, in place.
	pub fn run_query(&self, query: &mut Query, battle_state: &BattleState, registry: &Registry) {
		debug_assert!(
			!self.dirty,
			"queried a stale HookTable - call refresh() first"
		);
		for sub in self.queries[query.kind().index()].iter() {
			let ctx = HookCtx {
				source: sub.source,
				battle_state,
				registry,
			};
			(sub.run)(&ctx, query);
		}
	}

	// -----------------------------------------------------------------------
	// Convenience wrappers, so the engine reads as prose
	// -----------------------------------------------------------------------

	/// The effective value of `stat` for this calculation, after modifiers.
	pub fn effective_stat(
		&self,
		battle_state: &BattleState,
		registry: &Registry,
		pos: PositionId,
		stat: Stat,
		base: u32,
	) -> u32 {
		let mut query = Query::ModifyStat {
			pos,
			stat,
			value: base,
		};
		self.run_query(&mut query, battle_state, registry);
		query.value().unwrap_or(base)
	}

	/// The damage that should actually be dealt, after modifiers.
	pub fn final_damage(
		&self,
		battle_state: &BattleState,
		registry: &Registry,
		attacker: PositionId,
		target: PositionId,
		move_id: MoveId,
		base: u32,
	) -> u32 {
		let mut query = Query::ModifyDamage {
			attacker,
			target,
			move_id,
			amount: base,
		};
		self.run_query(&mut query, battle_state, registry);
		query.value().unwrap_or(base)
	}

	/// Whether a status is allowed to land on `target`.
	pub fn allows_status(
		&self,
		battle_state: &BattleState,
		registry: &Registry,
		target: PositionId,
		status: NonVolatileStatus,
	) -> bool {
		let mut query = Query::TryApplyStatus {
			target,
			status,
			allowed: true,
		};
		self.run_query(&mut query, battle_state, registry);
		query.allowed()
	}

	/// Whether `user` is allowed to execute `move_id` this turn.
	pub fn allows_move(
		&self,
		battle_state: &BattleState,
		registry: &Registry,
		user: PositionId,
		move_id: MoveId,
	) -> bool {
		let mut query = Query::TryMove {
			user,
			move_id,
			allowed: true,
		};
		self.run_query(&mut query, battle_state, registry);
		query.allowed()
	}
}
