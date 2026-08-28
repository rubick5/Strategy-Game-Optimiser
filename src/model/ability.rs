//! Abilities.
//!
//! An ability is just an id here. Everything it *does* lives in
//! [`battle::hooks::effects::ability`](crate::battle::hooks::effects::ability)
//! as a static hook table, so adding one never touches the engine.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbilityId {
	/// Immune to damage from moves flagged `ground`.
	Levitate = 0,
	/// Attackers making contact take 1/8 of their max HP.
	RoughSkin = 1,
	/// Attack is boosted while afflicted with a non-volatile status, and the
	/// burn Attack drop is cancelled rather than compounded.
	Guts = 2,
	/// Summons a sandstorm on switch-in.
	SandStream = 3,
	/// Non-volatile status is cured on switch-out.
	NaturalCure = 4,
}

pub const ABILITY_COUNT: usize = 5;

impl AbilityId {
	pub const ALL: [AbilityId; ABILITY_COUNT] = [
		AbilityId::Levitate,
		AbilityId::RoughSkin,
		AbilityId::Guts,
		AbilityId::SandStream,
		AbilityId::NaturalCure,
	];

	/// Stable index, used for the RL one-hot encoding and nothing else.
	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}

	pub fn name(self) -> &'static str {
		match self {
			AbilityId::Levitate => "Levitate",
			AbilityId::RoughSkin => "Rough Skin",
			AbilityId::Guts => "Guts",
			AbilityId::SandStream => "Sand Stream",
			AbilityId::NaturalCure => "Natural Cure",
		}
	}
}

impl std::fmt::Display for AbilityId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name())
	}
}
