use crate::battle::state::BattleState;
use crate::model::speciesdata::SpeciesData;
use crate::model::speciesdata::SpeciesId;
use crate::model::pmove::PMove;
use crate::model::pmove::MoveId;
use crate::battle::state::PositionId;

pub struct Registry {
	pub pokemon: Vec<SpeciesData>,
	moves: Vec<PMove>,
}

impl Registry {
	pub fn get_move(self: &Self, move_id: MoveId) -> &PMove {
		self.moves.get(move_id.0 as usize).unwrap()
	}

	pub fn get_pokemon(self: &Self, species_id: SpeciesId) -> &SpeciesData {
		self.pokemon.get(species_id.0 as usize).unwrap()
	}

	pub fn load() -> Self {
		let frail_attacker = SpeciesData {
			name: String::from("frail_attacker"),
			base_hp: 80,
			species_id: SpeciesId(0),
			attack: 100,
			defense: 100,
			speed: 90,
		};
	
		let fat_defender = SpeciesData {
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
			base_power: 40,
			effects: vec![],
			calc_prio: normal_priority,
		};

		let quick_attack = PMove {
			name: String::from("quick-attack"),
			move_id: MoveId(1),
			base_power: 40,
			effects: vec![],
			calc_prio: plus_one_prio,
		};
		Registry {
			pokemon: vec![frail_attacker, fat_defender],
			moves: vec![tackle, quick_attack],
		}
	}
}

fn normal_priority(_: &BattleState, _: PositionId) -> i8 {
	0
}

fn plus_one_prio(_: &BattleState, _: PositionId) -> i8 {
	1
}