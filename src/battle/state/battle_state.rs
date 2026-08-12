use crate::battle::state::{Outcome, Team, creature_state::CreatureState, field::{Field, PositionId}, roster::{Roster, RosterId}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleState {
	pub roster: Roster,
	pub field: Field,
}

impl BattleState {
	pub fn from(team0: Vec<CreatureState>, team1: Vec<CreatureState>, field: Vec<usize>) -> Self {
		BattleState {
			field: Field::from(field),
			roster: Roster::from(team0, team1),
		}
	}

	pub fn outcome(&self) -> Option<Outcome> {
		let mut side1_alive = false;
		let mut side0_alive = false;
		for creature in self.roster.team(&Team::Zero).into_iter().flatten() {
			if creature.current_hp > 0 {
				side0_alive = true;
			}
		}
		for creature in self.roster.team(&Team::One).into_iter().flatten() {
			if creature.current_hp > 0 {
				side1_alive = true;
			}
		}
		match (side0_alive, side1_alive) {
			(true, true) => None,
			(true, false) => Some(Outcome::Win { team: Team::Zero }),
			(false, true) => Some(Outcome::Win { team: Team::One }),
			(false, false) => Some(Outcome::Draw)

		}
	}

	pub fn get_mon_from_team(&self, team: &Team, team_index: usize) -> Option<&CreatureState> {
		//self.roster.team(team).get(team_index).as_ref()?
		let roster_id = match team {
			Team::One => RosterId(1 + team_index * 2),
			Team::Zero => RosterId(team_index * 2),
		};
		self.roster.get_mon(roster_id)
	}

	pub fn get_mon(&self, pos: PositionId) -> Option<&CreatureState> {
		let index = self.field[pos];
		self.roster.get_mon(index)
	}

	pub fn get_mut_mon(&mut self, pos: PositionId) -> Option<&mut CreatureState> {
		let index = self.field[pos];
		self.roster.get_mut_mon(index)
	}
}