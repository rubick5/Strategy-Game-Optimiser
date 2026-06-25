use crate::battle::state::PositionId;

pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
	},
	Switch {
		pos: PositionId,
	}
}