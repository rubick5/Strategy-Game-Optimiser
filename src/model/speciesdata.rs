pub struct SpeciesId(pub u32);

pub struct SpeciesData {
	pub name: String,
	pub species_id: SpeciesId,
	pub attack: u8,
	pub defense: u8,
	pub speed: u8,
}