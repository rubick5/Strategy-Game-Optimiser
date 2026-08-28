//! Handler signatures and the static declaration format effects are written in.

use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::event::Event;
use crate::battle::hooks::query::{Query, QueryKind};
use crate::battle::hooks::source::HookSource;
use crate::battle::hooks::trigger::{Trigger, TriggerKind};
use crate::battle::state::battle_state::BattleState;
use crate::model::registry::Registry;

/// Everything a handler is allowed to see.
///
/// Note that `battle_state` is immutable: a hook cannot mutate the battle
/// directly. Reactive hooks express change by pushing [`Event`]s; query hooks
/// express change by editing the [`Query`]. This is what keeps every mutation
/// funnelled through `engine::execute_event`.
pub struct HookCtx<'a> {
	/// The entity this handler is running on behalf of. Use `source.owner()` to
	/// find out which creature it is attached to.
	pub source: HookSource,
	pub battle_state: &'a BattleState,
	pub registry: &'a Registry,
}

/// A reactive handler: reads the battle, queues events.
///
/// Plain `fn` pointer rather than `Box<dyn Fn>` on purpose — no allocation, no
/// lifetimes, `Copy`, and the whole subscription list stays in one contiguous
/// `Vec` that the CPU can walk cheaply.
pub type ReactiveFn = fn(&HookCtx, &Trigger, &mut VecDeque<Event>, &mut dyn RngCore);

/// A query handler: edits a value in flight.
///
/// Deliberately has no RNG. Queries are folded through several handlers and are
/// sometimes re-run, so they must be pure. Anything random belongs in a reactive
/// hook, which runs exactly once per broadcast.
pub type QueryFn = fn(&HookCtx, &mut Query);

/// Which list a hook subscribes to, and the function to call.
#[derive(Clone, Copy)]
pub enum HookKind {
	Reactive { trigger: TriggerKind, run: ReactiveFn },
	Query { query: QueryKind, run: QueryFn },
}

/// One subscription, as written in an effect's static hook table.
///
/// Effects declare these in `static` slices, so an effect's full behaviour is a
/// single readable list of "when, in what order, do what".
#[derive(Clone, Copy)]
pub struct HookDef {
	pub kind: HookKind,
	/// Lower runs first. See [`order`](super::order) for the shared scale.
	/// Ties are broken by the owner's Speed, faster first.
	pub order: i16,
}

impl HookDef {
	pub const fn reactive(trigger: TriggerKind, order: i16, run: ReactiveFn) -> Self {
		HookDef {
			kind: HookKind::Reactive { trigger, run },
			order,
		}
	}

	pub const fn query(query: QueryKind, order: i16, run: QueryFn) -> Self {
		HookDef {
			kind: HookKind::Query { query, run },
			order,
		}
	}
}
