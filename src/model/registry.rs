use crate::model::ability::AbilityId;
use crate::model::effect::Effect;
use crate::model::pmove::MoveFlags;
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

	/// IDs 0-1 (species) and 0-2 (moves) are the originals, kept at their old
	/// numbers and stats so `example_battles/normal_battle.json` still loads.
	/// Everything new is appended.
	///
	/// A note on movesets: `Mask` indexes a `[bool; MOVESLOT_COUNT]`, so **no
	/// creature may be given more than 4 moves**.
	pub fn load() -> Self {
		Registry {
			species_data: Self::species(),
			moves: Self::moves(),
		}
	}

	fn species() -> Vec<SpeciesDatum> {
		vec![
			// --- originals, untouched apart from the new special stats -------
			SpeciesDatum {
				name: String::from("frail_attacker"),
				species_id: SpeciesId(0),
				base_hp: 85,
				attack: 100,
				defense: 100,
				special_attack: 100,
				special_defense: 100,
				speed: 90,
				ability: None,
			},
			SpeciesDatum {
				name: String::from("fat_defender"),
				species_id: SpeciesId(1),
				base_hp: 1000,
				attack: 80,
				defense: 1000,
				special_attack: 80,
				special_defense: 1000,
				speed: 30,
				ability: None,
			},

			// --- the new roster ----------------------------------------------
			// Fast physical attacker that *wants* to be statused.
			SpeciesDatum {
				name: String::from("cinderfox"),
				species_id: SpeciesId(2),
				base_hp: 100,
				attack: 115,
				defense: 70,
				special_attack: 95,
				special_defense: 70,
				speed: 105,
				ability: Some(AbilityId::Guts),
			},
			// Slow, bulky, turns the field hostile the moment it arrives.
			SpeciesDatum {
				name: String::from("stonewarden"),
				species_id: SpeciesId(3),
				base_hp: 150,
				attack: 95,
				defense: 125,
				special_attack: 60,
				special_defense: 95,
				speed: 45,
				ability: Some(AbilityId::SandStream),
			},
			// Special wall whose pivot is genuinely free.
			SpeciesDatum {
				name: String::from("mireling"),
				species_id: SpeciesId(4),
				base_hp: 130,
				attack: 70,
				defense: 100,
				special_attack: 110,
				special_defense: 105,
				speed: 60,
				ability: Some(AbilityId::NaturalCure),
			},
			// Punishes anything that touches it.
			SpeciesDatum {
				name: String::from("thornbeast"),
				species_id: SpeciesId(5),
				base_hp: 140,
				attack: 110,
				defense: 105,
				special_attack: 60,
				special_defense: 75,
				speed: 70,
				ability: Some(AbilityId::RoughSkin),
			},
			// Fast special attacker with a hard immunity to read around.
			SpeciesDatum {
				name: String::from("gustling"),
				species_id: SpeciesId(6),
				base_hp: 95,
				attack: 65,
				defense: 60,
				special_attack: 120,
				special_defense: 85,
				speed: 125,
				ability: Some(AbilityId::Levitate),
			},
			// Deliberately ability-less: the control case, so the learner has
			// something to contrast the others against.
			SpeciesDatum {
				name: String::from("brackenox"),
				species_id: SpeciesId(7),
				base_hp: 160,
				attack: 100,
				defense: 95,
				special_attack: 75,
				special_defense: 110,
				speed: 50,
				ability: None,
			},
		]
	}

	fn moves() -> Vec<PMove> {
		vec![
			// --- originals, ids unchanged ------------------------------------
			PMove {
				name: String::from("tackle"),
				move_id: MoveId(0),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 40,
				effects: vec![],
				base_prio: 0,
				flags: MoveFlags::CONTACT,
			},
			PMove {
				name: String::from("quick-attack"),
				move_id: MoveId(1),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 25,
				effects: vec![],
				base_prio: 1,
				flags: MoveFlags::CONTACT,
			},
			PMove {
				name: String::from("poison attack"),
				move_id: MoveId(2),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![Effect::PoisonChance { chance: 100 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},

			// --- new moves ---------------------------------------------------
			PMove {
				name: String::from("flame lash"),
				move_id: MoveId(3),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 65,
				effects: vec![Effect::BurnChance { chance: 20 }],
				base_prio: 0,
				flags: MoveFlags::CONTACT,
			},
			PMove {
				name: String::from("cinder blast"),
				move_id: MoveId(4),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Special,
				base_power: 80,
				effects: vec![Effect::BurnChance { chance: 10 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("stone edge"),
				move_id: MoveId(5),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 75,
				effects: vec![],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			// Ground moves are what Levitate reads.
			PMove {
				name: String::from("earth spike"),
				move_id: MoveId(6),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 70,
				effects: vec![],
				base_prio: 0,
				flags: MoveFlags::CONTACT_GROUND,
			},
			PMove {
				name: String::from("mud wave"),
				move_id: MoveId(7),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Special,
				base_power: 60,
				effects: vec![],
				base_prio: 0,
				flags: MoveFlags::GROUND,
			},
			PMove {
				name: String::from("venom fang"),
				move_id: MoveId(8),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 55,
				effects: vec![Effect::PoisonChance { chance: 30 }],
				base_prio: 0,
				flags: MoveFlags::CONTACT,
			},
			PMove {
				name: String::from("toxic mist"),
				move_id: MoveId(9),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![Effect::BadPoisonChance { chance: 90 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("static jolt"),
				move_id: MoveId(10),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Special,
				base_power: 65,
				effects: vec![Effect::ParalysisChance { chance: 30 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("numbing gaze"),
				move_id: MoveId(11),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Status,
				base_power: 0,
				effects: vec![Effect::ParalysisChance { chance: 100 }],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("aqua pulse"),
				move_id: MoveId(12),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Special,
				base_power: 70,
				effects: vec![],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("shadow dart"),
				move_id: MoveId(13),
				move_targeting: MoveTargeting::Single,
				move_type: MoveType::Physical,
				base_power: 45,
				effects: vec![],
				base_prio: 1,
				flags: MoveFlags::CONTACT,
			},
		]
	}

	/// The intended moveset for each new species — four moves each, which is the
	/// hard cap `Mask` imposes.
	///
	/// Kept here rather than in the battle files so a team can be rebuilt in code
	/// without hand-copying move ids.
	pub fn default_moveset(species_id: SpeciesId) -> Vec<MoveId> {
		match species_id.0 {
			// cinderfox: strong physical + self-status synergy with Guts
			2 => vec![MoveId(3), MoveId(13), MoveId(8), MoveId(5)],
			// stonewarden: sand setter with rock/ground coverage
			3 => vec![MoveId(5), MoveId(6), MoveId(0), MoveId(11)],
			// mireling: special wall that spreads status and pivots it away
			4 => vec![MoveId(12), MoveId(7), MoveId(9), MoveId(10)],
			// thornbeast: contact punisher that also punishes contact back
			5 => vec![MoveId(6), MoveId(0), MoveId(8), MoveId(5)],
			// gustling: fast specials, immune to the ground moves it fears
			6 => vec![MoveId(4), MoveId(10), MoveId(12), MoveId(1)],
			// brackenox: no ability, plain coverage
			7 => vec![MoveId(5), MoveId(0), MoveId(12), MoveId(11)],
			// originals
			_ => vec![MoveId(0), MoveId(1), MoveId(2)],
		}
	}
}
