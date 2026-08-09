use crate::{battle::state::{field::PositionId, roster::RosterId}, model::pmove::MoveId};

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
	MoveAction(MoveCommand),
	Switch {
		current: PositionId,
		new: RosterId,
	},
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoveCommand {
	pub move_id: MoveId,
	pub user: PositionId,
	pub targets: Vec<PositionId>,
}