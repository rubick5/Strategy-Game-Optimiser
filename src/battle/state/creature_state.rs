use serde::{Deserialize, Serialize};

use crate::{battle::state::{non_volatile_status::NonVolatileStatus, stat_stages::StatStages}, model::{pmove::MoveId, registry::Registry, speciesdata::{SpeciesDatum, SpeciesId, Stat}}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatureState {
	pub species_id: SpeciesId,
	pub stat_changes: StatStages,
	pub current_hp: u32,
	pub max_hp: u32,
	pub moves: Vec<MoveId>,
	pub non_vol_status: NonVolatileStatus,
}

impl CreatureState {
	pub fn from_species_data(species_data: &SpeciesDatum, moves: Vec<MoveId>) -> Self {
		CreatureState {
			species_id: species_data.species_id,
			stat_changes: StatStages::new(),
			current_hp: species_data.base_hp as u32,
			max_hp: species_data.base_hp as u32,
			moves,
			non_vol_status: NonVolatileStatus::NoStatus,
		}
	}
	pub fn from_species(registry: &Registry, species_id: SpeciesId, moves: Vec<MoveId>) -> Self {
		let species_datum = registry.get_species_data(species_id);
		Self::from_species_data(species_datum, moves)
	}
	pub fn get_stat(&self, stat: Stat, registry: &Registry) -> u32 {
		let species_data = registry.get_species_data(self.species_id);
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