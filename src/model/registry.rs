use crate::model::pokemon::Pokemon;
use crate::model::pokemon::SpeciesId;
use crate::model::pmove::PMove;
use crate::model::pmove::MoveId;

pub struct Registry {
	pub pokemon: Vec<Pokemon>,
	pub moves: Vec<PMove>,
}

impl Registry {
	pub fn load() -> Self {
		let frail_attacker = Pokemon {
			name: String::from("frail_attacker"),
			species_id: SpeciesId(0),
			attack: 100,
			defense: 20,
			speed: 90,
		};
	
		let fat_defender = Pokemon {
			name: String::from("fat_defender"),
			species_id: SpeciesId(1),
			attack: 80,
			defense: 80,
			speed: 30,
		};

		let tackle = PMove {
			move_id: MoveId(0),
			base_power: 40,
		};
		Registry {
			pokemon: vec![frail_attacker, fat_defender],
			moves: vec![tackle],
		}
	}
}