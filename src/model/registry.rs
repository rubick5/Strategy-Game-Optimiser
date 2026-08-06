use crate::model::pmove::MoveTargeting;
use crate::model::pmove::MoveType;
use crate::model::speciesdata::SpeciesDatum;
use crate::model::speciesdata::SpeciesId;
use crate::model::pmove::PMove;
use crate::model::pmove::MoveId;

pub struct Registry {
	pub species_data: Vec<SpeciesDatum>,
	pub moves: Vec<PMove>,
}

impl Registry {
	pub fn get_move(&self, move_id: MoveId) -> &PMove {
		self.moves.get(move_id.0 as usize).unwrap()
	}

	pub fn get_species_data(&self, species_id: SpeciesId) -> &SpeciesDatum {
		self.species_data.get(species_id.0 as usize).unwrap()
	}

	pub fn load() -> Self {
		let frail_attacker = SpeciesDatum {
			name: String::from("frail_attacker"),
			base_hp: 85,
			species_id: SpeciesId(0),
			attack: 100,
			defense: 100,
			speed: 90,
		};
	
		let fat_defender = SpeciesDatum {
			name: String::from("fat_defender"),
			base_hp: 120,
			species_id: SpeciesId(1),
			attack: 80,
			defense: 80,
			speed: 30,
		};

		let tackle = PMove {
			name: String::from("tackle"),
			move_id: MoveId(0),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 40,
			effects: vec![],
			base_prio: 0,
		};

		let quick_attack = PMove {
			name: String::from("quick-attack"),
			move_id: MoveId(1),
			move_targeting: MoveTargeting::Single,
			move_type: MoveType::Physical,
			base_power: 25,
			effects: vec![],
			base_prio: 1,
		};

		Registry {
			species_data: vec![frail_attacker, fat_defender],
			moves: vec![tackle, quick_attack],
		}
	}
}
