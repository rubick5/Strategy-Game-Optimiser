pub type SpeciesId = u32;

pub struct Pokemon {
	pub name: String,
	pub species_id: SpeciesId,
	pub attack: u8,
	pub defense: u8,
	pub speed: u8,
}