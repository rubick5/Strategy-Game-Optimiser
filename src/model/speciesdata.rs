use serde::{Deserialize, Serialize};

use crate::model::ability::AbilityId;

#[derive(Copy, Clone)]

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SpeciesId(pub u32);

#[derive(Clone)]
pub struct SpeciesDatum {
	pub name: String,
	pub species_id: SpeciesId,
	pub base_hp: u32,
	pub attack: u32,
	pub defense: u32,
	pub special_attack: u32,
	pub special_defense: u32,
	pub speed: u32,
	/// The ability every member of this species has. `None` is legal and is a
	/// useful control case for the learner.
	pub ability: Option<AbilityId>,
}

/// `Copy + PartialEq` so a `Stat` can travel inside a `Query::ModifyStat` and be
/// matched on by hook handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stat {
	Attack = 0,
	Defense = 1,
	SpecialAttack = 2,
	SpecialDefense = 3,
	Speed = 4,
}

pub const STAT_COUNT: usize = 5;

impl Stat {
	pub const ALL: [Stat; STAT_COUNT] = [
		Stat::Attack,
		Stat::Defense,
		Stat::SpecialAttack,
		Stat::SpecialDefense,
		Stat::Speed,
	];

	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}
}
