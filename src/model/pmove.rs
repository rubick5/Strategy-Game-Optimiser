use crate::{battle::state::{battle_state::BattleState, field::PositionId}, model::effect::Effect};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MoveId(pub u32);

pub enum MoveType {
	Physical,
	Special,
	Status
}

pub enum MoveTargeting {
	Single,
	Foes,
	Surrounding,
	All,
	Oneself,
	Ally,
	Allies
}

impl MoveTargeting {
	pub fn calc_targets(&self, battle_state: &BattleState, position: PositionId) -> Vec<PositionId> {
		match self {
			MoveTargeting::Single => {
				let target = match position {
					PositionId(0) => PositionId(1),
					PositionId(1) => PositionId(0),
					_ => panic!("someone tried to use a move, but they don't exist!")
				};
				vec![target]
			},
			MoveTargeting::Foes => {
				let target_team = position.team().other();
				battle_state.field.team_positions(&target_team)
			},
			MoveTargeting::Surrounding => {
				battle_state.field.all_field_positions().iter().filter(|p| **p != position).map(|p| *p).collect()
			},
			MoveTargeting::All => {
				battle_state.field.all_field_positions()
			},
			MoveTargeting::Oneself => vec![position],
			MoveTargeting::Ally => battle_state.field.team_positions(&position.team()).iter().filter(|p| **p != position).map(|p| *p).collect(),
			MoveTargeting::Allies => battle_state.field.team_positions(&position.team()),
		}
	}
}

pub struct PMove {
	pub name: String,
	pub move_type: MoveType,
	pub move_targeting: MoveTargeting,
	pub move_id: MoveId,
	pub base_power: u32,
	pub effects: Vec<Effect>,
	pub base_prio: i8
}