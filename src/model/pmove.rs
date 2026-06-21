use crate::model::effect::Effect;

pub struct MoveId(pub u32);
pub struct PMove {
	pub move_id: MoveId,
	pub base_power: u32,
	pub effects: Vec<Effect>,
}