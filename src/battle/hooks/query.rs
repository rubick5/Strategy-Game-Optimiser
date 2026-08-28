//! Query hooks: values the engine asks about before it commits to them.
//!
//! A [`Query`] is threaded by `&mut` through every subscribed handler in order,
//! so each one sees the previous one's result. The engine then reads the final
//! value out. This is the mechanism for anything that *changes a number* or
//! *vetoes an action*, as opposed to reacting after the fact.

use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::model::pmove::MoveId;
use crate::model::speciesdata::Stat;

/// The *identity* of a query, with no payload. Indexes the query dispatch array.
///
/// If you add a variant here you **must** bump [`QUERY_KIND_COUNT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryKind {
	/// A damage number, after the base formula, before it becomes an event.
	ModifyDamage = 0,
	/// A creature's effective stat *for this calculation only*.
	ModifyStat = 1,
	/// Whether a non-volatile status is allowed to land.
	TryApplyStatus = 2,
	/// Whether a creature is allowed to execute the move it picked.
	TryMove = 3,
}

/// Number of variants in [`QueryKind`]. Sizes the dispatch array.
pub const QUERY_KIND_COUNT: usize = 4;

impl QueryKind {
	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}
}

/// A value in flight, offered to every subscribed query hook for modification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Query {
	ModifyDamage {
		attacker: PositionId,
		target: PositionId,
		move_id: MoveId,
		/// Mutated in place by handlers.
		amount: u32,
	},
	ModifyStat {
		pos: PositionId,
		stat: Stat,
		/// Mutated in place by handlers.
		value: u32,
	},
	TryApplyStatus {
		target: PositionId,
		status: NonVolatileStatus,
		/// Handlers set this to `false` to make the status fail.
		allowed: bool,
	},
	TryMove {
		user: PositionId,
		move_id: MoveId,
		/// Handlers set this to `false` to make the move fail outright.
		allowed: bool,
	},
}

impl Query {
	/// Which subscription list this query should be folded through.
	pub fn kind(&self) -> QueryKind {
		match self {
			Query::ModifyDamage { .. } => QueryKind::ModifyDamage,
			Query::ModifyStat { .. } => QueryKind::ModifyStat,
			Query::TryApplyStatus { .. } => QueryKind::TryApplyStatus,
			Query::TryMove { .. } => QueryKind::TryMove,
		}
	}

	/// The numeric payload, for the two queries that carry one.
	pub fn value(&self) -> Option<u32> {
		match self {
			Query::ModifyDamage { amount, .. } => Some(*amount),
			Query::ModifyStat { value, .. } => Some(*value),
			_ => None,
		}
	}

	/// The boolean verdict, for the two queries that carry one.
	/// Defaults to `true` (allowed) for queries that are not vetoes.
	pub fn allowed(&self) -> bool {
		match self {
			Query::TryApplyStatus { allowed, .. } => *allowed,
			Query::TryMove { allowed, .. } => *allowed,
			_ => true,
		}
	}
}
