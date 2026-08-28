/// Secondary effects a move rolls for when it is used.
///
/// These fire at move time. Anything that needs to *react* to something later
/// belongs in a hook (see `battle::hooks::effects`), not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
	PoisonChance { chance: u8 },
	BadPoisonChance { chance: u8 },
	BurnChance { chance: u8 },
	ParalysisChance { chance: u8 },
}

impl Effect {
	/// Used by the RL encoder to tell "this move has a rider" from "it doesn't".
	pub fn chance(&self) -> u8 {
		match self {
			Effect::PoisonChance { chance }
			| Effect::BadPoisonChance { chance }
			| Effect::BurnChance { chance }
			| Effect::ParalysisChance { chance } => *chance,
		}
	}
}
