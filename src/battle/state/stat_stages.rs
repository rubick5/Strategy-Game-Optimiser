use serde::{Deserialize, Serialize};


#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct StatStages {
	pub attack: i8,
	pub defense: i8,
	pub speed: i8,
}

impl StatStages {
	pub fn new() -> Self {
		StatStages {
			attack: 0,
			defense: 0,
			speed: 0,
		}
	}
	pub fn stage_multiplier(stage: i8) -> f32 {
		match stage {
			n if n < 0 => 2.0 / (-n as f32 + 2.0),
			n => (2.0 + n as f32) / 2.0
		}
	}
}