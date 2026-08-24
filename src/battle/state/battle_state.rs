use std::{error::Error, fs::File, io::Write as _};

use crate::{battle::state::{Outcome, Team, creature_state::CreatureState, field::{Field, PositionId}, roster::{Roster, RosterId}, weather::TimedWeather}, model::registry::Registry};
use serde::{Deserialize, Serialize};
use crate::model::speciesdata::Stat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleState {
	pub roster: Roster,
	pub field: Field,
	pub weather: Option<TimedWeather>,
	pub trick_room: bool,
}

impl BattleState {
	pub fn from_file(file_name: &str) -> Result<Self, Box<dyn Error>> {
		let bytes = std::fs::read(file_name)?;
		let saved = serde_json::from_slice(&bytes)?;
		Ok(saved)
	}

	pub fn to_file(&self, target: &str) -> Result<(), Box<dyn Error>> {
		let bytes = serde_json::to_string(&self)?;
		let mut file = File::create(target)?;
		Ok(file.write_all(bytes.as_bytes())?)
	}
	
	pub fn from(team0: Vec<CreatureState>, team1: Vec<CreatureState>, field: Vec<usize>) -> Self {
		BattleState {
			field: Field::from(field),
			roster: Roster::from(team0, team1),
			weather: None,
			trick_room: false,
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

	pub fn all_field_mons_ordered(&self, registry: &Registry) -> Vec<RosterId> {
		let rids = self.field.all_field_mons();
		let mut states: Vec<(&RosterId, &CreatureState)> = rids.into_iter().zip(rids.iter().map(|rid| self.roster.get_mon(*rid)).flatten()).collect();
		states.sort_by(|(_, a), (_, b)| a.get_stat(Stat::Speed, registry).cmp(&b.get_stat(Stat::Speed, registry)));

		if self.trick_room {
			states.reverse() // technically they dont calculate it this way in real game who cares it never makes a difference...
		}
		states.iter().map(|(rid, _)| **rid).collect()
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