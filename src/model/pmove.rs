use crate::{battle::state::{BattleState, PositionId}, model::effect::Effect};

#[derive(Copy, Clone, PartialEq)]
pub struct MoveId(pub u32);
pub struct PMove {
	pub name: String,
	pub move_id: MoveId,
	pub base_power: u32,
	pub effects: Vec<Effect>,
	pub calc_prio: fn(&BattleState, PositionId) -> i8,
}