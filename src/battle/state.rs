use std::collections::HashMap;

use crate::model::speciesdata::SpeciesId;

pub type PositionId = u32;

pub struct BattleState {
	pub mons: HashMap<PositionId, PokemonState>,
}

pub struct PokemonState {
	pub species_id: SpeciesId,
	pub stat_changes: StatStages,
	pub current_hp: u32,
}

pub struct StatStages {
	pub attack: i8,
	pub defense: i8,
	pub speed: i8,
}