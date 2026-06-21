use crate::model::pokemon::SpeciesId;

pub struct BattleState {
	pub team1: Vec<PokemonState>,
	pub team2: Vec<PokemonState>,
}

pub struct PokemonState {
	pub species_id: SpeciesId,

}

pub struct StatChanges {
}