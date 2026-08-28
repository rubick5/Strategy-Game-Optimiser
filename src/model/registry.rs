use crate::battle::state::stat_stages::StatStages;
use crate::model::ability::AbilityId;
use crate::model::effect::Effect;
use crate::model::pmove::MoveFlags;
use crate::model::pmove::MoveTargeting;
use crate::model::pmove::MoveType;
use crate::model::speciesdata::SpeciesDatum;
use crate::model::speciesdata::SpeciesId;
use crate::model::speciesdata::Stat;
use crate::model::typing::{Type, Typing};
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
	/// numbers so `example_battles/normal_battle.json` still loads.
	///
	/// A note on movesets: `Mask` indexes a `[bool; MOVESLOT_COUNT]`, so **no
	/// creature may be given more than 4 moves**.
	pub fn load() -> Self {
		Registry {
			species_data: Self::species(),
			moves: Self::moves(),
		}
	}

	/// The roster is deliberately a rock-paper-scissors web rather than a
	/// straight power ladder, because the point of the type chart is to make
	/// *which* creature is out matter more than which has the bigger numbers:
	///
	/// * stonewarden (Rock/Ground) walls gustling's Electric but folds 4x to
	///   Water and Grass — mireling and thornbeast punish it hard.
	/// * gustling (Electric) is weak to Ground, and Levitate is exactly what
	///   cancels that, so its ability is worth reading before you click a move.
	/// * cinderfox (Fire) eats thornbeast (Grass) but drowns to mireling (Water).
	/// * brackenox (Steel/Ground) resists a great deal and is immune to Electric
	///   and Poison, but takes double from Fire and Water.
	fn species() -> Vec<SpeciesDatum> {
		vec![
			// --- originals, kept for the old battle files ---------------------
			SpeciesDatum {
				name: String::from("frail_attacker"),
				species_id: SpeciesId(0),
				base_hp: 85,
				attack: 100, defense: 100, special_attack: 100, special_defense: 100, speed: 90,
				typing: Typing::mono(Type::Normal),
				ability: None,
			},
			SpeciesDatum {
				name: String::from("fat_defender"),
				species_id: SpeciesId(1),
				base_hp: 1000,
				attack: 80, defense: 1000, special_attack: 80, special_defense: 1000, speed: 30,
				typing: Typing::mono(Type::Normal),
				ability: None,
			},

			// --- the roster ---------------------------------------------------
			SpeciesDatum {
				name: String::from("cinderfox"),
				species_id: SpeciesId(2),
				base_hp: 120,
				attack: 115, defense: 80, special_attack: 95, special_defense: 80, speed: 105,
				typing: Typing::mono(Type::Fire),
				ability: Some(AbilityId::Guts),
			},
			SpeciesDatum {
				name: String::from("stonewarden"),
				species_id: SpeciesId(3),
				base_hp: 150,
				attack: 105, defense: 125, special_attack: 60, special_defense: 95, speed: 45,
				typing: Typing::dual(Type::Rock, Type::Ground),
				ability: Some(AbilityId::SandStream),
			},
			SpeciesDatum {
				name: String::from("mireling"),
				species_id: SpeciesId(4),
				base_hp: 135,
				attack: 70, defense: 100, special_attack: 110, special_defense: 105, speed: 65,
				typing: Typing::dual(Type::Water, Type::Poison),
				ability: Some(AbilityId::NaturalCure),
			},
			SpeciesDatum {
				name: String::from("thornbeast"),
				species_id: SpeciesId(5),
				base_hp: 140,
				attack: 110, defense: 100, special_attack: 65, special_defense: 80, speed: 70,
				typing: Typing::mono(Type::Grass),
				ability: Some(AbilityId::RoughSkin),
			},
			SpeciesDatum {
				name: String::from("gustling"),
				species_id: SpeciesId(6),
				base_hp: 120,
				attack: 65, defense: 75, special_attack: 120, special_defense: 90, speed: 120,
				typing: Typing::mono(Type::Electric),
				ability: Some(AbilityId::Levitate),
			},
			// Deliberately ability-less: the control case, so the learner has
			// something to contrast the others against.
			SpeciesDatum {
				name: String::from("brackenox"),
				species_id: SpeciesId(7),
				base_hp: 145,
				attack: 100, defense: 105, special_attack: 75, special_defense: 100, speed: 55,
				typing: Typing::dual(Type::Steel, Type::Ground),
				ability: None,
			},
		]
	}

	fn moves() -> Vec<PMove> {
		let attack = |id: u32, name: &str, element: Type, cat: MoveType, power: u32, prio: i8, flags: MoveFlags, effects: Vec<Effect>| PMove {
			name: String::from(name),
			move_id: MoveId(id),
			move_targeting: MoveTargeting::Single,
			move_type: cat,
			element,
			base_power: power,
			effects,
			base_prio: prio,
			flags,
		};

		use MoveType::{Physical, Special, Status};
		vec![
			// --- originals, ids unchanged ------------------------------------
			attack(0, "tackle", Type::Normal, Physical, 40, 0, MoveFlags::CONTACT, vec![]),
			attack(1, "quick-attack", Type::Normal, Physical, 25, 1, MoveFlags::CONTACT, vec![]),
			attack(2, "poison attack", Type::Poison, Status, 0, 0, MoveFlags::NONE,
				vec![Effect::PoisonChance { chance: 100 }]),

			// --- attacking moves, one per relevant type ----------------------
			attack(3, "flame lash", Type::Fire, Physical, 65, 0, MoveFlags::CONTACT,
				vec![Effect::BurnChance { chance: 20 }]),
			attack(4, "cinder blast", Type::Fire, Special, 80, 0, MoveFlags::NONE,
				vec![Effect::BurnChance { chance: 10 }]),
			attack(5, "stone edge", Type::Rock, Physical, 75, 0, MoveFlags::NONE, vec![]),
			attack(6, "earth spike", Type::Ground, Physical, 70, 0, MoveFlags::CONTACT, vec![]),
			attack(7, "mud wave", Type::Ground, Special, 60, 0, MoveFlags::NONE, vec![]),
			attack(8, "venom fang", Type::Poison, Physical, 55, 0, MoveFlags::CONTACT,
				vec![Effect::PoisonChance { chance: 30 }]),
			attack(9, "toxic mist", Type::Poison, Status, 0, 0, MoveFlags::NONE,
				vec![Effect::BadPoisonChance { chance: 90 }]),
			attack(10, "static jolt", Type::Electric, Special, 65, 0, MoveFlags::NONE,
				vec![Effect::ParalysisChance { chance: 30 }]),
			attack(11, "numbing gaze", Type::Normal, Status, 0, 0, MoveFlags::NONE,
				vec![Effect::ParalysisChance { chance: 100 }]),
			attack(12, "aqua pulse", Type::Water, Special, 70, 0, MoveFlags::NONE, vec![]),
			attack(13, "shadow dart", Type::Ghost, Physical, 45, 1, MoveFlags::CONTACT, vec![]),
			attack(14, "thorn whip", Type::Grass, Physical, 65, 0, MoveFlags::CONTACT, vec![]),
			attack(15, "frost bolt", Type::Ice, Special, 70, 0, MoveFlags::NONE, vec![]),
			attack(16, "iron press", Type::Steel, Physical, 70, 0, MoveFlags::CONTACT, vec![]),
			attack(17, "mind shatter", Type::Psychic, Special, 75, 0, MoveFlags::NONE, vec![]),
			attack(18, "wing slash", Type::Flying, Physical, 60, 0, MoveFlags::CONTACT, vec![]),

			// --- volatile-status moves --------------------------------------
			// Damaging moves with volatile riders.
			attack(19, "dizzy ray", Type::Psychic, Special, 55, 0, MoveFlags::NONE,
				vec![Effect::ConfusionChance { chance: 40 }]),
			attack(20, "rock smash", Type::Rock, Physical, 60, 0, MoveFlags::CONTACT,
				vec![Effect::FlinchChance { chance: 30 }]),
			// Pure status moves.
			attack(21, "jeer", Type::Dark, Status, 0, 0, MoveFlags::NONE, vec![Effect::Taunt]),
			attack(22, "sap seed", Type::Grass, Status, 0, 0, MoveFlags::NONE, vec![Effect::LeechSeed]),
			attack(23, "baffle", Type::Ghost, Status, 0, 0, MoveFlags::NONE,
				vec![Effect::ConfusionChance { chance: 100 }]),
			// Self-targeting: `Oneself` makes the move's only target the user, so
			// these need no separate self-effect plumbing.
			PMove {
				name: String::from("decoy"),
				move_id: MoveId(24),
				move_targeting: MoveTargeting::Oneself,
				move_type: Status,
				element: Type::Normal,
				base_power: 0,
				effects: vec![Effect::Substitute],
				base_prio: 0,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("guard"),
				move_id: MoveId(25),
				move_targeting: MoveTargeting::Oneself,
				move_type: Status,
				element: Type::Normal,
				base_power: 0,
				effects: vec![Effect::Protect],
				// +4 so it resolves before the attack it is meant to block.
				base_prio: 4,
				flags: MoveFlags::NONE,
			},
			PMove {
				name: String::from("blade dance"),
				move_id: MoveId(26),
				move_targeting: MoveTargeting::Oneself,
				move_type: Status,
				element: Type::Normal,
				base_power: 0,
				effects: vec![
					Effect::SelfStatChanges {
						stat_changes: StatStages::from_pairs(vec![(Stat::Attack, 2)]),
						chance: 100,
					}
				],
				base_prio: 0,
				flags: MoveFlags::NONE,

			}
		]
	}

	/// Four moves each — the cap `Mask` imposes.
	///
	/// Every set is STAB plus coverage, so the right click depends on what is
	/// across from you. gustling in particular carries aqua pulse specifically
	/// because its STAB does nothing at all to stonewarden.
	pub fn default_moveset(species_id: SpeciesId) -> Vec<MoveId> {
		let ids: Vec<u32> = match species_id.0 {
			// cinderfox: Fire STAB, Rock coverage, priority, and a Substitute to
			// set up behind — Guts wants to survive the status it profits from.
			2 => vec![3, 5, 13, 24],
			// stonewarden: Rock + Ground STAB, Steel coverage, and Protect to stall
			// sand chip while the opponent takes it. brackenox's Taunt shuts that
			// Protect off, which is the interaction worth having on the board.
			3 => vec![5, 6, 16, 25],
			// mireling: Water STAB, Ice coverage, bad poison, and Leech Seed —
			// Natural Cure lets it pivot the poison away and come back clean.
			4 => vec![12, 15, 9, 22],
			// thornbeast: Grass STAB, Ground coverage, flinch pressure, Leech Seed.
			5 => vec![14, 6, 20, 22],
			// gustling: Electric STAB, Water for the Ground types it can't touch,
			// and confusion off its high Special Attack.
			6 => vec![10, 12, 19, 4],
			// brackenox: Steel + Ground STAB, Rock coverage, Taunt to shut down
			// the status-move users it walls.
			7 => vec![16, 6, 5, 26],
			// originals
			_ => vec![0, 1, 2],
		};
		ids.into_iter().map(MoveId).collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::rl::moveslot::MOVESLOT_COUNT;

	/// `Mask` indexes a fixed [bool; MOVESLOT_COUNT], so an over-long moveset is
	/// an out-of-bounds panic a long way from the registry entry that caused it.
	#[test]
	fn no_moveset_exceeds_the_slot_cap() {
		let registry = Registry::load();
		for species in &registry.species_data {
			let moves = Registry::default_moveset(species.species_id);
			assert!(
				moves.len() <= MOVESLOT_COUNT,
				"{} has {} moves, cap is {}",
				species.name, moves.len(), MOVESLOT_COUNT
			);
		}
	}

	/// Every id referenced by a moveset must exist, and ids must match position.
	#[test]
	fn registry_ids_are_consistent() {
		let registry = Registry::load();
		for (i, mv) in registry.moves.iter().enumerate() {
			assert_eq!(mv.move_id.0 as usize, i, "{} is filed under the wrong id", mv.name);
		}
		for (i, s) in registry.species_data.iter().enumerate() {
			assert_eq!(s.species_id.0 as usize, i, "{} is filed under the wrong id", s.name);
		}
		for species in &registry.species_data {
			for m in Registry::default_moveset(species.species_id) {
				assert!(
					(m.0 as usize) < registry.moves.len(),
					"{} references move {} which does not exist",
					species.name, m.0
				);
			}
		}
	}

	/// Each roster member should get STAB off at least one of its moves,
	/// otherwise its typing is decoration.
	#[test]
	fn every_roster_member_has_stab_coverage() {
		let registry = Registry::load();
		for species in registry.species_data.iter().filter(|s| s.species_id.0 >= 2) {
			let has_stab = Registry::default_moveset(species.species_id)
				.iter()
				.any(|m| {
					let mv = registry.get_move(*m);
					mv.move_type.is_damaging() && species.typing.contains(mv.element)
				});
			assert!(has_stab, "{} has no STAB attack", species.name);
		}
	}

	/// Every roster member needs an answer to something its STAB cannot touch,
	/// or there is no move-choice decision to learn.
	#[test]
	fn every_roster_member_carries_off_type_coverage() {
		let registry = Registry::load();
		for species in registry.species_data.iter().filter(|s| s.species_id.0 >= 2) {
			let off_type = Registry::default_moveset(species.species_id)
				.iter()
				.filter(|m| {
					let mv = registry.get_move(**m);
					mv.move_type.is_damaging() && !species.typing.contains(mv.element)
				})
				.count();
			assert!(off_type >= 1, "{} has no coverage move", species.name);
		}
	}
}
