use crate::battle::state::{field::PositionId, roster::RosterId};


pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
	},
	Switch {
		current: PositionId,
		new: RosterId,
	}
}