use crate::model::pmove::MoveId;
use crate::model::speciesdata::{SpeciesData, SpeciesId};
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;
use std::fmt::Display;

pub const TEAM_SIZE: usize = 6;

#[derive(PartialEq)]
pub enum Team {
	Zero,
	One
}

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

pub enum Outcome {
	Side0Wins,
	Side1Wins,
	Draw,
}

#[derive(Debug, Clone)]
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

	pub fn team(&self, team: Team) -> Vec<&RosterId> {
		match team {
			Team::Zero => self.mons.iter().step_by(2).collect(),

			Team::One => self.mons.iter().skip(1).step_by(2).collect()

		}
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RosterId(pub usize);

#[derive(Debug, Clone)]
pub struct Roster {
	mons: Vec<Option<PokemonState>>,   // Option so an empty/fainted slot still exists
}

impl Roster {

	pub fn get_mon(&self, rid: RosterId) -> &Option<PokemonState> {
		&self.mons[rid.0]
	}
	pub fn get_mut_mon(&mut self, rid: RosterId) -> Option<&mut PokemonState> {
		self.mons[rid.0].as_mut()
	}

	pub fn from(team0: Vec<PokemonState>, team1: Vec<PokemonState>) -> Self {
		let mut mons: Vec<Option<PokemonState>> = vec![None; 2 * TEAM_SIZE];
		for i in 0..team0.len() {
			let monstate = match team0.get(i) {
				Some(mon) => Some(mon.clone()),
				None => None
			};
			mons[i*2] = monstate;
		}
		for i in 0..team1.len() {
			let monstate = match team1.get(i) {
				Some(mon) => Some(mon.clone()),
				None => None
			};
			mons[i*2 + 1] = monstate;
		}
		Roster {
			mons
		}
	}

	pub fn team(&self, team: Team) -> Vec<&Option<PokemonState>> {
		match team {
			Team::Zero => self.mons.iter().step_by(2).collect(),
			Team::One => self.mons.iter().skip(1).step_by(2).collect()
		}
	}
	/*
	pub fn get(&self, pos: PositionId) -> Option<&PokemonState> {
		self.mons[pos.0 as usize].as_ref()
	}

	pub fn get_mut(&mut self, pos: PositionId) -> Option<&mut PokemonState> {
		self.mons.get_mut(pos.0 as usize)?.as_mut()
	}

	pub fn iter(&self) -> impl Iterator<Item = (PositionId, &PokemonState)> + '_ {
		self.mons.iter().enumerate().filter_map(|(i, slot)| {
			slot.as_ref().map(|mon| (PositionId(i as u32), mon))
		})
	} */
	pub fn all_mons(&self) -> impl Iterator<Item = Option<&PokemonState>> + '_ {
		self.mons.iter().map(|slot| slot.as_ref())
	}
}

#[derive(Debug, Clone)]
pub struct BattleState {
	pub roster: Roster,
	pub field: Field,
	pub switch_needed: Vec<PositionId>,
}

impl BattleState {
	pub fn from(team0: Vec<PokemonState>, team1: Vec<PokemonState>, field: Vec<usize>) -> Self {
		BattleState {
			field: Field::from(field),
			roster: Roster::from(team0, team1),
			switch_needed: vec![],
		}
	}

	pub fn outcome(&self) -> Option<Outcome> {
		let mut side1_alive = false;
		let mut side0_alive = false;
		for mon in self.roster.team(Team::Zero).into_iter().flatten() {
			if mon.current_hp > 0 {
				side0_alive = true;
			}
		}
		for mon in self.roster.team(Team::One).into_iter().flatten() {
			if mon.current_hp > 0 {
				side1_alive = true;
			}
		}
		match (side0_alive, side1_alive) {
			(true, true) => None,
			(true, false) => Some(Outcome::Side0Wins),
			(false, true) => Some(Outcome::Side1Wins),
			(false, false) => Some(Outcome::Draw)

		}
	}

	pub fn get_mon_from_team(&self, team: &Team, team_index: usize) -> Option<&PokemonState> {
		//self.roster.team(team).get(team_index).as_ref()?
		let roster_id = match team {
			Team::One => RosterId(1 + team_index * 2),
			Team::Zero => RosterId(team_index * 2),
		};
		self.roster.get_mon(roster_id).as_ref()
	}

	pub fn get_mon(&self, pos: PositionId) -> Option<&PokemonState> {
		let index = self.field[pos];
		self.roster.get_mon(index).as_ref()
	}

	pub fn get_mut_mon(&mut self, pos: PositionId) -> Option<&mut PokemonState> {
		let index = self.field[pos];
		self.roster.get_mut_mon(index)
	}
}

#[derive(Debug, Clone)]
pub struct PokemonState {
	pub species_id: SpeciesId,
	pub stat_changes: StatStages,
	pub current_hp: u32,
	pub moves: Vec<MoveId>,
}

impl PokemonState {
	pub fn from_species_data(species_data: SpeciesData, moves: Vec<MoveId>) -> Self {
		PokemonState {
			species_id: species_data.species_id,
			stat_changes: StatStages::new(),
			current_hp: species_data.base_hp as u32,
			moves,
		}
	}
	pub fn from_species(registry: &Registry, species_id: SpeciesId, moves: Vec<MoveId>) -> Self {
		let pokemon = registry.get_pokemon(species_id);
		PokemonState {
			species_id,
			stat_changes: StatStages::new(),
			current_hp: pokemon.base_hp as u32,
			moves,
		}
	}
	pub fn get_stat(&self, stat: Stat, registry: &Registry) -> u32 {
		let species_data = registry.get_pokemon(self.species_id);
		match stat {
			Stat::Attack => {
				
				let stage_multiplier = StatStages::stage_multiplier(self.stat_changes.attack);
				(species_data.attack as f32 * stage_multiplier) as u32
			}
			Stat::Defense => {
				let stage_multiplier = StatStages::stage_multiplier(self.stat_changes.defense);
				(species_data.defense as f32 * stage_multiplier) as u32
			},
			Stat::Speed => {
				let stage_multiplier = StatStages::stage_multiplier(self.stat_changes.speed);
				(species_data.speed as f32 * stage_multiplier) as u32
			}
		}
	}
}

#[derive(Debug, PartialEq, Clone)]
pub struct StatStages {
	pub attack: i8,
	pub defense: i8,
	pub speed: i8,
}

impl StatStages {
	pub fn new() -> Self {
		StatStages {
			attack: 0,
			defense: 0,
			speed: 0,
		}
	}
	pub fn stage_multiplier(stage: i8) -> f32 {
		match stage {
			n if n < 0 => 2.0 / (-n as f32 + 2.0),
			n => (2.0 + n as f32) / 2.0
		}
	}
}