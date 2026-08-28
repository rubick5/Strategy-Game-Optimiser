use serde::{Deserialize, Serialize};

use crate::{battle::state::{battle_state::BattleState, field::PositionId}, model::effect::Effect};
use crate::model::speciesdata::Stat;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoveId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveType {
	Physical,
	Special,
	Status
}

impl MoveType {
	/// Whether this kind of move rolls damage at all.
	///
	/// Status moves used to queue a `DealDamage { amount: 0 }` event, which fired
	/// `AfterDamage` hooks for a hit that never happened.
	pub fn is_damaging(&self) -> bool {
		!matches!(self, MoveType::Status)
	}

	/// The attacker's stat this move scales off.
	pub fn attacking_stat(&self) -> Option<Stat> {
		match self {
			MoveType::Physical => Some(Stat::Attack),
			MoveType::Special => Some(Stat::SpecialAttack),
			MoveType::Status => None,
		}
	}

	/// The defender's stat this move is reduced by.
	pub fn defending_stat(&self) -> Option<Stat> {
		match self {
			MoveType::Physical => Some(Stat::Defense),
			MoveType::Special => Some(Stat::SpecialDefense),
			MoveType::Status => None,
		}
	}
}

/// Properties hooks ask about.
///
/// There is no type chart in this model, so these two flags carry the weight a
/// type would: `contact` is what Rough Skin punishes, `ground` is what Levitate
/// is immune to. Add flags here rather than adding branches to the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MoveFlags {
	/// The user physically touches the target.
	pub contact: bool,
	/// A ground-based move. Levitate ignores these.
	pub ground: bool,
}

impl MoveFlags {
	pub const NONE: MoveFlags = MoveFlags { contact: false, ground: false };
	pub const CONTACT: MoveFlags = MoveFlags { contact: true, ground: false };
	pub const GROUND: MoveFlags = MoveFlags { contact: false, ground: true };
	pub const CONTACT_GROUND: MoveFlags = MoveFlags { contact: true, ground: true };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
	pub base_prio: i8,
	pub flags: MoveFlags,
}
