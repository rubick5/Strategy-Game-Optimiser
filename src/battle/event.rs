use crate::battle::state::{PositionId, RosterId};

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