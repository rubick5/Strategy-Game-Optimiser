use crate::battle::state::{field::PositionId, non_volatile_status::NonVolatileStatus, roster::RosterId};


pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
	},
	Switch {
		current: PositionId,
		new: RosterId,
	},
	ApplyNonVolStatus {
		status: NonVolatileStatus,
		target: PositionId,
	}
}