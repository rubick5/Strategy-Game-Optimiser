#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, Serialize, Deserialize)]
pub enum NonVolatileStatus {
	NoStatus = 0,
	Poison = 1,
	BadPoison = 2,
	Burn = 3,
	Paralysis = 4,
}
use NonVolatileStatus::*;
use serde::{Deserialize, Serialize};

/// Number of *afflicted* states, i.e. excluding `NoStatus`. Sizes the RL one-hot.
pub const STATUS_COUNT: usize = 4;

impl NonVolatileStatus {
	/// The afflicted states, in one-hot order. `NoStatus` is deliberately absent:
	/// an all-zero block already means "healthy".
	pub const AFFLICTIONS: [NonVolatileStatus; STATUS_COUNT] = [Poison, BadPoison, Burn, Paralysis];

	pub fn is_afflicted(&self) -> bool {
		!matches!(self, NoStatus)
	}
}

impl std::fmt::Display for NonVolatileStatus {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", match self {
			NoStatus => "None",
			Poison => "Poison",
			BadPoison => "Bad Poison",
			Burn => "Burn",
			Paralysis => "Paralysis",
		})
	}
}
