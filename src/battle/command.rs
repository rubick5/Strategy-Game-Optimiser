use crate::battle::state::PositionId;
use crate::model::pmove::MoveId;

pub enum Command {
	MoveAction(MoveAction),
	Switch,
}

pub struct MoveAction {
	pub move_id: MoveId,
	pub user: PositionId,
	pub targets: Vec<PositionId>,
}