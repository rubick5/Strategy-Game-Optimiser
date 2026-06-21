#[derive(Copy, Clone)]

#[derive(Debug)]
pub struct SpeciesId(pub u32);

pub struct SpeciesData {
	pub name: String,
	pub species_id: SpeciesId,
	pub base_hp: u8,
	pub attack: u8,
	pub defense: u8,
	pub speed: u8,
}

pub enum Stat {
	Attack,
	Defense,
	Speed,
}