use serde::{Deserialize, Serialize};

#[derive(Copy, Clone)]

#[derive(Debug, Serialize, Deserialize)]
pub struct SpeciesId(pub u32);

#[derive(Clone)]
pub struct SpeciesDatum {
	pub name: String,
	pub species_id: SpeciesId,
	pub base_hp: u32,
	pub attack: u32,
	pub defense: u32,
	pub speed: u32,
}

/// `Copy + PartialEq` so a `Stat` can travel inside a `Query::ModifyStat` and be
/// matched on by hook handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
	Attack,
	Defense,
	Speed,
}