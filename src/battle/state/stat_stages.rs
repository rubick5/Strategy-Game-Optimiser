use serde::{Deserialize, Serialize};

use crate::model::speciesdata::Stat;

/// Stat stages run -6..=+6, as in the games.
pub const MAX_STAGE: i8 = 6;

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct StatStages {
	pub attack: i8,
	pub defense: i8,
	pub speed: i8,
	// `serde(default)` so battle states saved before the special split still load.
	#[serde(default)]
	pub special_attack: i8,
	#[serde(default)]
	pub special_defense: i8,
}

impl StatStages {
	pub fn new() -> Self {
		StatStages {
			attack: 0,
			defense: 0,
			speed: 0,
			special_attack: 0,
			special_defense: 0,
		}
	}

	pub fn get(&self, stat: Stat) -> i8 {
		match stat {
			Stat::Attack => self.attack,
			Stat::Defense => self.defense,
			Stat::SpecialAttack => self.special_attack,
			Stat::SpecialDefense => self.special_defense,
			Stat::Speed => self.speed,
		}
	}

	pub fn set(&mut self, stat: Stat, value: i8) {
		let clamped = value.clamp(-MAX_STAGE, MAX_STAGE);
		match stat {
			Stat::Attack => self.attack = clamped,
			Stat::Defense => self.defense = clamped,
			Stat::SpecialAttack => self.special_attack = clamped,
			Stat::SpecialDefense => self.special_defense = clamped,
			Stat::Speed => self.speed = clamped,
		}
	}

	pub fn stage_multiplier(stage: i8) -> f32 {
		match stage {
			n if n < 0 => 2.0 / (-n as f32 + 2.0),
			n => (2.0 + n as f32) / 2.0
		}
	}
}

impl Default for StatStages {
	fn default() -> Self {
		Self::new()
	}
}
