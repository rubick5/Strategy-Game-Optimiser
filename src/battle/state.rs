use std::collections::HashMap;

use crate::model::speciesdata::SpeciesId;

pub type PositionId = u32;

pub struct BattleState {
	pub mons: HashMap<PositionId, PokemonState>,
}

pub struct PokemonState {
	pub species_id: SpeciesId,

}

pub struct StatChanges {
}