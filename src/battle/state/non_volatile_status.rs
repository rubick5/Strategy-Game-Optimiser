#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum NonVolatileStatus {
	NoStatus,
	Poison,
	BadPoison,
	Burn,
}
use NonVolatileStatus::*;
use serde::{Deserialize, Serialize};

impl std::fmt::Display for NonVolatileStatus {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", match self {
			NoStatus => "None",
			Poison => "Poison",
			BadPoison => "Bad Poison",
			Burn => "Burn",
		})
	}
}