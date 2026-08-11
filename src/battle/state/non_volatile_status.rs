#[derive(Debug, PartialEq, Copy, Clone)]
pub enum NonVolatileStatus {
	NoStatus,
	Poison,
	BadPoison,
	Burn,
}
use NonVolatileStatus::*;

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