use crate::model::effect::Effect;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MoveId(pub u32);
pub struct PMove {
	pub name: String,
	pub move_id: MoveId,
	pub base_power: u32,
	pub effects: Vec<Effect>,
	pub base_prio: i8
}