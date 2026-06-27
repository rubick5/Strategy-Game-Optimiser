use std::collections::HashMap;

use crate::model::speciesdata::{SpeciesData, SpeciesId};
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct PositionId(pub u32);

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

#[derive(Debug)]
pub struct BattleState {
	pub mons: HashMap<PositionId, PokemonState>,
}

impl BattleState {
	pub fn outcome(&self) -> Option<Outcome> {
		None
	}
}

#[derive(Debug)]
pub struct PokemonState {
	pub species_id: SpeciesId,
	pub stat_changes: StatStages,
	pub current_hp: u32,
}

impl PokemonState {
	pub fn from_species_data(species_data: SpeciesData) -> Self {
		PokemonState {
			species_id: species_data.species_id,
			stat_changes: StatStages::new(),
			current_hp: species_data.base_hp as u32,
		}
	}
	pub fn from_species(registry: &Registry, species_id: SpeciesId) -> Self {
		let pokemon = registry.get_pokemon(species_id);
		PokemonState {
			species_id,
			stat_changes: StatStages::new(),
			current_hp: pokemon.base_hp as u32,
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

#[derive(Debug, PartialEq)]
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