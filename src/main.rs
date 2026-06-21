use poke_sim::model::pokemon::Pokemon;
fn main() {
	let frail_attacker = Pokemon {
		name: String::from("frail_attacker"),
		species_id: 0,
		attack: 100,
		defense: 20,
		speed: 90,
	};

	let fat_defender = Pokemon {
		name: String::from("fat_defender"),
		species_id: 1,
		attack: 80,
		defense: 80,
		speed: 30,
	};
}