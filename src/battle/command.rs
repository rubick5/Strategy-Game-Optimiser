use crate::battle::state::PositionId;
use crate::model::pmove::MoveId;

#[derive(Clone, PartialEq)]
pub enum Command {
	MoveAction(MoveCommand),
	Switch(PositionId),
}

#[derive(Clone, PartialEq)]
pub struct MoveCommand {
	pub move_id: MoveId,
	pub user: PositionId,
	pub targets: Vec<PositionId>,
}