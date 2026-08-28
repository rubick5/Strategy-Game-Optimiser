//! Who owns a hook.
//!
//! Handlers are plain `fn` pointers with no captured state, so they need to be
//! told which entity they are running on behalf of. That is what the source is
//! for: a burn's residual handler asks its source "which position am I burning?"
//! rather than scanning the field for burned creatures.

use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::volatile::VolatileKind;

/// The entity a hook subscription belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HookSource {
	/// A non-volatile status on the creature at `pos`.
	Status {
		pos: PositionId,
		status: NonVolatileStatus,
	},
	/// The ability of the creature at `pos`.
	///
	/// Not produced yet — see the seam in [`providers`](super::providers).
	Ability { pos: PositionId },
	/// The held item of the creature at `pos`.
	///
	/// Not produced yet — see the seam in [`providers`](super::providers).
	Item { pos: PositionId },
	/// A volatile condition on the creature at `pos`.
	Volatile {
		pos: PositionId,
		kind: VolatileKind,
	},
	/// The current field weather. Not owned by any one creature.
	Weather,
}

impl HookSource {
	/// The creature this hook is attached to, if it is attached to one at all.
	///
	/// Handlers use this constantly: it is how "the burned creature" is
	/// identified without searching for it.
	pub fn owner(&self) -> Option<PositionId> {
		match self {
			HookSource::Status { pos, .. } => Some(*pos),
			HookSource::Ability { pos } => Some(*pos),
			HookSource::Item { pos } => Some(*pos),
			HookSource::Volatile { pos, .. } => Some(*pos),
			HookSource::Weather => None,
		}
	}
}
