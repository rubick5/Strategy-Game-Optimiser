use serde::{Deserialize, Serialize};

use crate::model::speciesdata::Stat;

/// Stat stages run -6..=+6, as in the games.
pub const MAX_STAGE: i8 = 6;

macro_rules! stat_stages {
	(
		$($field:ident => $stat:ident ),*
		$(,)?
	) => {
			#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize, Copy)]
			pub struct StatStages {
				$(
					pub $field: i8,
				)*
			}

			impl StatStages {
				pub fn stage_multiplier(stage: i8) -> f32 {
					match stage {
						n if n < 0 => 2.0 / (-n as f32 + 2.0),
						n => (2.0 + n as f32) / 2.0,
					}
				}
				pub fn new() -> Self {
					Self {
						$(
							$field: 0,
						)*
					}
				}

				pub fn get(&self, stat: Stat) -> i8 {
					match stat {
						$(
							Stat::$stat => self.$field,
						)*
					}
				}

				pub fn combine_in_place(&mut self, other: &Self) {
					$(
						self.$field = (self.$field + other.$field).clamp(-6, 6);
					)*
				}

				pub fn combine(&self, other: &Self) -> Self {
					Self {
						$(
							$field: (self.$field + other.$field).clamp(-6, 6),
						)*
					}
				}

				fn set(&mut self, stat: Stat, stage: i8) {
					match stat {
						$(
							Stat::$stat => self.$field = stage.clamp(-6, 6),
						)*
					}
				}

				pub fn from_pairs(pairs: Vec<(Stat, i8)>) -> Self {
					let mut s = Self::new();
					for (stat, stage) in pairs {
						s.set(stat, stage);
					}
					s
				}

			}
	};
}

stat_stages!(
	attack => Attack,
	defense => Defense,
	special_defense => SpecialDefense,
	special_attack => SpecialAttack,
	speed => Speed,
);