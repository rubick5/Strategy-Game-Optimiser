use crate::model::pmove::MoveId;
use crate::battle::command::Command;
use crate::battle::state::PositionId;

pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
	},
	CommandEvent(Command)
}