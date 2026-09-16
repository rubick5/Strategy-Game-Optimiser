use serde::{Deserialize, Serialize};

use crate::battle::state::{TEAM_SIZE, Team, creature_state::CreatureState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RosterId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Roster {
	mons: Vec<Option<CreatureState>>,   // Option so an empty/fainted slot still exists
}

impl Roster {

	pub fn get_mon(&self, rid: RosterId) -> Option<&CreatureState> {
		self.mons[rid.0].as_ref()
	}
	pub fn get_mut_mon(&mut self, rid: RosterId) -> Option<&mut CreatureState> {
		self.mons[rid.0].as_mut()
	}

	pub fn from(team0: Vec<CreatureState>, team1: Vec<CreatureState>) -> Self {
		let mut mons: Vec<Option<CreatureState>> = vec![None; 2 * TEAM_SIZE];
		for i in 0..team0.len() {
			let monstate = match team0.get(i) {
				Some(creature) => Some(creature.clone()),
				None => None
			};
			mons[i*2] = monstate;
		}
		for i in 0..team1.len() {
			let monstate = match team1.get(i) {
				Some(creature) => Some(creature.clone()),
				None => None
			};
			mons[i*2 + 1] = monstate;
		}
		Roster {
			mons
		}
	}

	pub fn team(&self, team: &Team) -> Vec<&Option<CreatureState>> {
		match team {
			Team::Zero => self.mons.iter().step_by(2).collect(),
			Team::One => self.mons.iter().skip(1).step_by(2).collect()
		}
	}
	
	pub fn all_mons(&self) -> impl Iterator<Item = Option<&CreatureState>> + '_ {
		self.mons.iter().map(|slot| slot.as_ref())
	}
}