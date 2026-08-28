//! # The hook system
//!
//! The problem this solves: every ability, item, status and field condition wants
//! to interfere at some *specific moment* of a turn. The naive way to support that
//! is to sprinkle `if attacker.ability == Levitate { ... }` checks through the
//! engine, which means every effect you add costs another branch in a hot path
//! that runs millions of times during training.
//!
//! Instead, effects *subscribe* to the moments they care about, and the engine
//! *broadcasts* those moments. If nothing on the field cares about `SwitchIn`,
//! the switch-in broadcast iterates an empty list and costs nothing.
//!
//! There are two kinds of hook, because effects want to do two different things:
//!
//! * **Reactive hooks** ([`Trigger`]) — "something happened, react to it". They
//!   receive an immutable view of the battle and push new [`Event`]s onto the
//!   queue. Rough Skin, Leftovers, poison damage. See [`trigger`].
//!
//! * **Query hooks** ([`Query`]) — "I am about to compute a value, does anyone
//!   want to change it?". They fold over a mutable value before the engine acts
//!   on it. Burn halving Attack, Levitate granting immunity, a status failing
//!   because the target already has one. See [`query`].
//!
//! Reactive hooks can never mutate the battle state directly — they can only
//! queue events. That keeps every mutation flowing through the one place
//! (`engine::execute_event`), so a hook can never silently desync the state.
//!
//! [`Event`]: crate::battle::event::Event

pub mod effects;
pub mod handler;
pub mod order;
pub mod providers;
pub mod query;
pub mod source;
pub mod table;
pub mod trigger;

pub use handler::{HookCtx, HookDef, HookKind, QueryFn, ReactiveFn};
pub use query::{Query, QueryKind, QUERY_KIND_COUNT};
pub use source::HookSource;
pub use table::HookTable;
pub use trigger::{Trigger, TriggerKind, TRIGGER_KIND_COUNT};
