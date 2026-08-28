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
	/// Volatile riders. `chance` is rolled the same way as a status rider.
	ConfusionChance { chance: u8 },
	FlinchChance { chance: u8 },
	/// Always-on volatile effects, used by dedicated status moves.
	Taunt,
	LeechSeed,
	/// Self-targeting. Reach these with `MoveTargeting::Oneself`, which makes the
	/// move's only target the user, so no separate self-effect plumbing is needed.
	Substitute,
	Protect,
}

impl Effect {
	/// Short human label, for the battle UI.
	pub fn label(&self) -> &'static str {
		match self {
			Effect::PoisonChance { .. } => "poison",
			Effect::BadPoisonChance { .. } => "bad poison",
			Effect::BurnChance { .. } => "burn",
			Effect::ParalysisChance { .. } => "paralyse",
			Effect::ConfusionChance { .. } => "confuse",
			Effect::FlinchChance { .. } => "flinch",
			Effect::Taunt => "taunt",
			Effect::LeechSeed => "leech seed",
			Effect::Substitute => "substitute",
			Effect::Protect => "protect",
		}
	}

	/// Used by the RL encoder to tell "this move has a rider" from "it doesn't".
	pub fn chance(&self) -> u8 {
		match self {
			Effect::PoisonChance { chance }
			| Effect::BadPoisonChance { chance }
			| Effect::BurnChance { chance }
			| Effect::ParalysisChance { chance }
			| Effect::ConfusionChance { chance }
			| Effect::FlinchChance { chance } => *chance,
			Effect::Taunt | Effect::LeechSeed | Effect::Substitute | Effect::Protect => 100,
		}
	}
}
