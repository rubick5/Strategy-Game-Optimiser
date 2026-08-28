//! Reactive hooks: the moments the engine broadcasts, and their payloads.

use crate::battle::event::DamageSource;
use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::model::pmove::MoveId;

/// The *identity* of a moment, with no payload attached.
///
/// This is what the [`HookTable`](super::HookTable) is indexed by. It is a plain
/// C-like enum so `kind as usize` is a direct array index — dispatching a trigger
/// is one array lookup, not a hash or a scan.
///
/// If you add a variant here you **must** bump [`TRIGGER_KIND_COUNT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerKind {
	/// A creature has just arrived on the field.
	SwitchIn = 0,
	/// A creature is about to leave the field (fires *before* the field changes,
	/// so the departing creature is still readable).
	SwitchOut = 1,
	/// A move has been selected and passed its can-I-move check.
	BeforeMove = 2,
	/// A move has finished queueing all of its damage and effects.
	AfterMove = 3,
	/// Damage was actually applied to a creature.
	AfterDamage = 4,
	/// A creature hit 0 HP.
	AfterFaint = 5,
	/// A non-volatile status was actually applied.
	StatusApplied = 6,
	/// The weather was set, changed or cleared.
	WeatherChanged = 7,
	/// End-of-turn "residual" phase: chip damage, healing, countdown effects.
	Residual = 8,
	/// The very end of the turn, after residuals have resolved.
	TurnEnd = 9,
}

/// Number of variants in [`TriggerKind`]. Sizes the dispatch array.
pub const TRIGGER_KIND_COUNT: usize = 10;

impl TriggerKind {
	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}
}

/// A moment *with* its payload, handed to every subscribed reactive hook.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trigger {
	SwitchIn {
		pos: PositionId,
	},
	SwitchOut {
		pos: PositionId,
	},
	BeforeMove {
		user: PositionId,
		move_id: MoveId,
	},
	AfterMove {
		user: PositionId,
		move_id: MoveId,
	},
	AfterDamage {
		target: PositionId,
		/// HP actually lost, which can be less than the damage rolled if the
		/// target did not have that much HP left.
		amount: u32,
		/// Attacker and move, when there was one. `None` for chip damage.
		source: Option<DamageSource>,
	},
	AfterFaint {
		pos: PositionId,
	},
	StatusApplied {
		target: PositionId,
		status: NonVolatileStatus,
	},
	WeatherChanged,
	Residual,
	TurnEnd,
}

impl Trigger {
	/// Which subscription list this trigger should be broadcast to.
	pub fn kind(&self) -> TriggerKind {
		match self {
			Trigger::SwitchIn { .. } => TriggerKind::SwitchIn,
			Trigger::SwitchOut { .. } => TriggerKind::SwitchOut,
			Trigger::BeforeMove { .. } => TriggerKind::BeforeMove,
			Trigger::AfterMove { .. } => TriggerKind::AfterMove,
			Trigger::AfterDamage { .. } => TriggerKind::AfterDamage,
			Trigger::AfterFaint { .. } => TriggerKind::AfterFaint,
			Trigger::StatusApplied { .. } => TriggerKind::StatusApplied,
			Trigger::WeatherChanged => TriggerKind::WeatherChanged,
			Trigger::Residual => TriggerKind::Residual,
			Trigger::TurnEnd => TriggerKind::TurnEnd,
		}
	}
}
