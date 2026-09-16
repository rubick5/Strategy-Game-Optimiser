use serde::{Deserialize, Serialize};

use crate::battle::state::{Team, roster::RosterId};
use std::fmt::Display;



#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct PositionId(pub usize);

impl PositionId {
	pub fn team(&self) -> Team {
		if self.0 % 2 == 0 {
			Team::Zero
		} else {
			Team::One
		}
	}
	pub fn is_ally(&self, other: Self) -> bool {
		self.team() == other.team()
	}
	pub fn is_opponent(&self, other: Self) -> bool {
		!self.is_ally(other)
	}
}

impl Display for PositionId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		self.0.fmt(f)
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
	mons: Vec<RosterId>,
}

impl Field {
	pub fn from(mons: Vec<usize>) -> Self {
		Field{
			mons: mons.iter().map(|x| RosterId(*x)).collect()
		}
	}
	pub fn all_field_mons(&self) -> &Vec<RosterId> {
		&self.mons
	}

	pub fn team(&self, team: &Team) -> Vec<&RosterId> {
		match team {
			Team::Zero => self.mons.iter().step_by(2).collect(),

			Team::One => self.mons.iter().skip(1).step_by(2).collect()

		}
	}

	pub fn team_positions(&self, team: &Team) -> Vec<PositionId> {
		match team {
			Team::Zero => (0..self.mons.len()).step_by(2).map(|n| PositionId(n)).collect(),
			Team::One => (1..self.mons.len()).step_by(2).map(|n| PositionId(n)).collect(),
		}
	}

	pub fn all_field_positions(&self) -> Vec<PositionId> {
		(0..self.mons.len()).map(|n| PositionId(n)).collect()
	}
}

impl std::ops::Index<PositionId> for Field {
	type Output = RosterId;
	fn index(&self, pos: PositionId) -> &Self::Output {
		&self.mons[pos.0 as usize]
	}
}
impl std::ops::IndexMut<PositionId> for Field {
	fn index_mut(&mut self, pos: PositionId) -> &mut Self::Output {
		&mut self.mons[pos.0 as usize]
	}
}
