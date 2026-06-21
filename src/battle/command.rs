use crate::battle::state::PositionId;
use crate::model::pmove::MoveId;

pub enum Command {
	MoveAction(MoveCommand),
	Switch,
}

pub struct MoveCommand {
	pub move_id: MoveId,
	pub user: PositionId,
	pub targets: Vec<PositionId>,
}