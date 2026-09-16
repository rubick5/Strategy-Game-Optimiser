pub mod field;
pub mod roster;
pub mod creature_state;
pub mod battle_state;
pub mod stat_stages;
pub mod non_volatile_status;
pub mod volatile;
pub mod weather;

pub const TEAM_SIZE: usize = 6;

/// `Copy` so a perspective can be passed around cheaply — the encoder threads one
/// through every call now.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum Team {
	Zero,
	One
}

impl Team {
	pub fn other(&self) -> Team {
		match self {
			Team::Zero => Team::One,
			Team::One => Team::Zero,
		}
	}

	/// Where this team's creatures sit in the interleaved roster: team index `i`
	/// is roster slot `i * 2 + offset`.
	pub fn roster_offset(&self) -> usize {
		match self {
			Team::Zero => 0,
			Team::One => 1,
		}
	}

	/// Every roster slot belonging to this team, bench included, in team order.
	pub fn roster_ids(&self) -> Vec<usize> {
		let offset = self.roster_offset();
		(0..TEAM_SIZE).map(|i| i * 2 + offset).collect()
	}
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Outcome {
	Win { team: Team },
	Draw,
}