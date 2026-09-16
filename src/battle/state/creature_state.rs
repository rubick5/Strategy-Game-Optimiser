use serde::{Deserialize, Serialize};

use crate::{battle::state::{non_volatile_status::NonVolatileStatus, stat_stages::StatStages, volatile::Volatiles}, model::{ability::AbilityId, pmove::MoveId, registry::Registry, speciesdata::{SpeciesDatum, SpeciesId, Stat}, typing::Typing}};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CreatureState {
	pub species_id: SpeciesId,
	pub stat_changes: StatStages,
	pub current_hp: u32,
	pub max_hp: u32,
	pub moves: Vec<MoveId>,
	pub non_vol_status: NonVolatileStatus,
	/// `serde(default)` so battle states saved before abilities existed still load
	/// (they come back with no ability, which is a legal state).
	#[serde(default)]
	pub ability: Option<AbilityId>,
	/// Temporary conditions, wiped whenever this creature leaves the field.
	/// `serde(default)` so battle states saved before volatiles existed load.
	#[serde(default)]
	pub volatiles: Volatiles,
}

impl CreatureState {

	pub fn get_typing(&self, registry: &Registry) -> Typing {
		registry.get_species_data(self.species_id).typing
	}
	pub fn from_species_data(species_data: &SpeciesDatum, moves: Vec<MoveId>) -> Self {
		CreatureState {
			species_id: species_data.species_id,
			stat_changes: StatStages::new(),
			current_hp: species_data.base_hp as u32,
			max_hp: species_data.base_hp as u32,
			moves,
			non_vol_status: NonVolatileStatus::NoStatus,
			ability: species_data.ability,
			volatiles: Volatiles::new(),
		}
	}

	pub fn from_species(registry: &Registry, species_id: SpeciesId, moves: Vec<MoveId>) -> Self {
		let species_datum = registry.get_species_data(species_id);
		Self::from_species_data(species_datum, moves)
	}

	/// The creature's own stat, stat stages included.
	///
	/// Note what is *not* here: combat modifiers like a burn halving Attack. Those
	/// live in the hook layer, because the RL encoder reads this function and
	/// wants the creature's own numbers, not numbers bent by the current battle.
	pub fn get_stat(&self, stat: Stat, registry: &Registry) -> u32 {
		let species_data = registry.get_species_data(self.species_id);
		let base = match stat {
			Stat::Attack => species_data.attack,
			Stat::Defense => species_data.defense,
			Stat::SpecialAttack => species_data.special_attack,
			Stat::SpecialDefense => species_data.special_defense,
			Stat::Speed => species_data.speed,
		};
		let stage_multiplier = StatStages::stage_multiplier(self.stat_changes.get(stat));
		(base as f32 * stage_multiplier) as u32
	}

	pub fn is_alive(&self) -> bool {
		self.current_hp > 0
	}
}
