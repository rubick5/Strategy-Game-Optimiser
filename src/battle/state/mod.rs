pub mod field;
pub mod roster;
pub mod creature_state;
pub mod battle_state;
pub mod stat_stages;
pub mod non_volatile_status;

pub const TEAM_SIZE: usize = 6;

#[derive(PartialEq, Debug)]
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
}

#[derive(Debug, PartialEq)]
pub enum Outcome {
	Win { team: Team },
	Draw,
}