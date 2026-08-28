//! Elemental types and the full 18x18 effectiveness chart.
//!
//! This is the mechanic the sim was missing. Without it there is no reason to
//! ever switch — you give up a turn and gain nothing — so "use your strongest
//! move every turn" was very close to the optimal policy, and an agent had
//! almost nothing to learn. With a chart, a switch buys resistance or an
//! immunity, coverage moves matter, and move choice depends on what is in front
//! of you rather than on base power alone.
//!
//! Multipliers are kept as exact rationals rather than floats so damage stays
//! integer-deterministic: the chart stores each matchup in half-units
//! (`0` = immune, `1` = 0.5x, `2` = 1x, `4` = 2x) and [`Effectiveness`] carries a
//! numerator/denominator pair through the calculation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
	Normal = 0,
	Fire = 1,
	Water = 2,
	Electric = 3,
	Grass = 4,
	Ice = 5,
	Fighting = 6,
	Poison = 7,
	Ground = 8,
	Flying = 9,
	Psychic = 10,
	Bug = 11,
	Rock = 12,
	Ghost = 13,
	Dragon = 14,
	Dark = 15,
	Steel = 16,
	Fairy = 17,
}

pub const TYPE_COUNT: usize = 18;

impl Type {
	pub const ALL: [Type; TYPE_COUNT] = [
		Type::Normal, Type::Fire, Type::Water, Type::Electric, Type::Grass, Type::Ice,
		Type::Fighting, Type::Poison, Type::Ground, Type::Flying, Type::Psychic, Type::Bug,
		Type::Rock, Type::Ghost, Type::Dragon, Type::Dark, Type::Steel, Type::Fairy,
	];

	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}

	pub fn name(self) -> &'static str {
		match self {
			Type::Normal => "Normal",
			Type::Fire => "Fire",
			Type::Water => "Water",
			Type::Electric => "Electric",
			Type::Grass => "Grass",
			Type::Ice => "Ice",
			Type::Fighting => "Fighting",
			Type::Poison => "Poison",
			Type::Ground => "Ground",
			Type::Flying => "Flying",
			Type::Psychic => "Psychic",
			Type::Bug => "Bug",
			Type::Rock => "Rock",
			Type::Ghost => "Ghost",
			Type::Dragon => "Dragon",
			Type::Dark => "Dark",
			Type::Steel => "Steel",
			Type::Fairy => "Fairy",
		}
	}
}

impl std::fmt::Display for Type {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name())
	}
}

/// A creature's type or types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Typing {
	pub primary: Type,
	pub secondary: Option<Type>,
}

impl Typing {
	pub const fn mono(primary: Type) -> Self {
		Typing { primary, secondary: None }
	}

	pub const fn dual(primary: Type, secondary: Type) -> Self {
		Typing { primary, secondary: Some(secondary) }
	}

	pub fn contains(&self, t: Type) -> bool {
		self.primary == t || self.secondary == Some(t)
	}

	pub fn iter(&self) -> impl Iterator<Item = Type> + '_ {
		std::iter::once(self.primary).chain(self.secondary.into_iter())
	}
}

/// An exact multiplier, kept as a rational so damage never touches a float.
///
/// Reachable values for a single hit are 0, 1/4, 1/2, 1, 2 and 4 from the chart,
/// each optionally multiplied by 3/2 for STAB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Effectiveness {
	pub numerator: u32,
	pub denominator: u32,
}

impl Effectiveness {
	pub const NEUTRAL: Effectiveness = Effectiveness { numerator: 1, denominator: 1 };
	pub const IMMUNE: Effectiveness = Effectiveness { numerator: 0, denominator: 1 };

	pub fn is_immune(&self) -> bool {
		self.numerator == 0
	}

	pub fn is_super_effective(&self) -> bool {
		self.numerator > self.denominator
	}

	pub fn is_resisted(&self) -> bool {
		self.numerator > 0 && self.numerator < self.denominator
	}

	/// Fold another rational in.
	pub fn times(self, numerator: u32, denominator: u32) -> Self {
		Effectiveness {
			numerator: self.numerator * numerator,
			denominator: self.denominator * denominator,
		}
	}

	/// Same-type attack bonus: x1.5.
	pub fn with_stab(self) -> Self {
		self.times(3, 2)
	}

	/// Apply to a damage figure. Rounds down, but never to zero unless the hit
	/// is a true immunity — a heavily resisted hit still does 1.
	pub fn apply(&self, damage: u32) -> u32 {
		if self.is_immune() {
			return 0;
		}
		if self.denominator == 0 {
			return damage;
		}
		(damage * self.numerator / self.denominator).max(1)
	}

	pub fn as_f32(&self) -> f32 {
		if self.denominator == 0 { 0.0 } else { self.numerator as f32 / self.denominator as f32 }
	}
}

/// Attacking type (row) against defending type (column), in half-units:
/// `0` = immune, `1` = 0.5x, `2` = 1x, `4` = 2x.
///
/// Row/column order is the discriminant order of [`Type`].
#[rustfmt::skip]
const CHART: [[u8; TYPE_COUNT]; TYPE_COUNT] = [
	//        Nor Fir Wat Ele Gra Ice Fig Poi Gro Fly Psy Bug Roc Gho Dra Dar Ste Fai
	/* Nor */ [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  2,  2,  1,  2],
	/* Fir */ [ 2,  1,  1,  2,  4,  4,  2,  2,  2,  2,  2,  4,  1,  2,  1,  2,  4,  2],
	/* Wat */ [ 2,  4,  1,  2,  1,  2,  2,  2,  4,  2,  2,  2,  4,  2,  1,  2,  2,  2],
	/* Ele */ [ 2,  2,  4,  1,  1,  2,  2,  2,  0,  4,  2,  2,  2,  2,  1,  2,  2,  2],
	/* Gra */ [ 2,  1,  4,  2,  1,  2,  2,  1,  4,  1,  2,  1,  4,  2,  1,  2,  1,  2],
	/* Ice */ [ 2,  1,  1,  2,  4,  1,  2,  2,  4,  4,  2,  2,  2,  2,  4,  2,  1,  2],
	/* Fig */ [ 4,  2,  2,  2,  2,  4,  2,  1,  2,  1,  1,  1,  4,  0,  2,  4,  4,  1],
	/* Poi */ [ 2,  2,  2,  2,  4,  2,  2,  1,  1,  2,  2,  2,  1,  1,  2,  2,  0,  4],
	/* Gro */ [ 2,  4,  2,  4,  1,  2,  2,  4,  2,  0,  2,  1,  4,  2,  2,  2,  4,  2],
	/* Fly */ [ 2,  2,  2,  1,  4,  2,  4,  2,  2,  2,  2,  4,  1,  2,  2,  2,  1,  2],
	/* Psy */ [ 2,  2,  2,  2,  2,  2,  4,  4,  2,  2,  1,  2,  2,  2,  2,  0,  1,  2],
	/* Bug */ [ 2,  1,  2,  2,  4,  2,  1,  1,  2,  1,  4,  2,  2,  1,  2,  4,  1,  1],
	/* Roc */ [ 2,  4,  2,  2,  2,  4,  1,  2,  1,  4,  2,  4,  2,  2,  2,  2,  1,  2],
	/* Gho */ [ 0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  4,  2,  2,  4,  2,  1,  2,  2],
	/* Dra */ [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  4,  2,  1,  0],
	/* Dar */ [ 2,  2,  2,  2,  2,  2,  1,  2,  2,  2,  4,  2,  2,  4,  2,  1,  2,  1],
	/* Ste */ [ 2,  1,  1,  1,  2,  4,  2,  2,  2,  2,  2,  2,  4,  2,  2,  2,  1,  4],
	/* Fai */ [ 2,  1,  2,  2,  2,  2,  4,  1,  2,  2,  2,  2,  2,  2,  4,  4,  1,  2],
];

/// One attacking type against one defending type.
pub fn single(attacking: Type, defending: Type) -> Effectiveness {
	let half_units = CHART[attacking.index()][defending.index()] as u32;
	Effectiveness { numerator: half_units, denominator: 2 }
}

/// One attacking type against a full typing, both types folded in.
///
/// A dual type can reach 4x and 1/4x, and a single immunity anywhere makes the
/// whole thing an immunity.
pub fn effectiveness(attacking: Type, defending: &Typing) -> Effectiveness {
	let mut result = Effectiveness::NEUTRAL;
	for def in defending.iter() {
		let half_units = CHART[attacking.index()][def.index()] as u32;
		result = result.times(half_units, 2);
	}
	result
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Nothing outside the four legal multipliers, and the chart is square.
	#[test]
	fn chart_is_well_formed() {
		for (i, row) in CHART.iter().enumerate() {
			assert_eq!(row.len(), TYPE_COUNT, "row {} is the wrong length", i);
			for (j, cell) in row.iter().enumerate() {
				assert!(
					matches!(cell, 0 | 1 | 2 | 4),
					"chart[{}][{}] = {} is not a legal multiplier",
					i, j, cell
				);
			}
		}
		assert_eq!(CHART.len(), TYPE_COUNT);
		assert_eq!(Type::ALL.len(), TYPE_COUNT);
	}

	/// Discriminants and the ALL array must stay in step, since the chart is
	/// indexed by discriminant and the encoder one-hots by ALL order.
	#[test]
	fn type_indices_match_their_position() {
		for (i, t) in Type::ALL.iter().enumerate() {
			assert_eq!(t.index(), i, "{:?} is out of order", t);
		}
	}

	/// Spot-checks against the real chart, including all three immunities that
	/// come from an attacking type doing nothing.
	#[test]
	fn known_matchups_are_right() {
		use Type::*;
		let cases = [
			(Fire, Grass, 2.0), (Grass, Fire, 0.5), (Water, Fire, 2.0),
			(Electric, Ground, 0.0), (Ground, Flying, 0.0), (Normal, Ghost, 0.0),
			(Ghost, Normal, 0.0), (Fighting, Ghost, 0.0), (Poison, Steel, 0.0),
			(Psychic, Dark, 0.0), (Dragon, Fairy, 0.0),
			(Fighting, Normal, 2.0), (Fairy, Dragon, 2.0), (Steel, Fairy, 2.0),
			(Bug, Dark, 2.0), (Dark, Ghost, 2.0), (Ice, Dragon, 2.0),
			(Rock, Flying, 2.0), (Flying, Fighting, 2.0), (Steel, Steel, 0.5),
		];
		for (atk, def, expected) in cases {
			let got = single(atk, def).as_f32();
			assert!(
				(got - expected).abs() < 1e-6,
				"{:?} -> {:?} was {} but should be {}",
				atk, def, got, expected
			);
		}
	}

	/// Dual types multiply, including to 4x and 1/4x.
	#[test]
	fn dual_types_stack() {
		use Type::*;
		// Water hits Rock 2x and Ground 2x -> 4x on Rock/Ground
		let rock_ground = Typing::dual(Rock, Ground);
		assert_eq!(effectiveness(Water, &rock_ground).as_f32(), 4.0);
		assert_eq!(effectiveness(Grass, &rock_ground).as_f32(), 4.0);
		// Electric is immune on the Ground half, so the whole thing is immune
		assert!(effectiveness(Electric, &rock_ground).is_immune());

		// Bug resisted by Fire (0.5) and Steel (0.5) -> 0.25x
		let fire_steel = Typing::dual(Fire, Steel);
		assert_eq!(effectiveness(Bug, &fire_steel).as_f32(), 0.25);
	}

	/// An immunity survives being combined with a super-effective half.
	#[test]
	fn immunity_beats_super_effective() {
		use Type::*;
		// Ground is 2x on Electric but 0x on Flying
		let electric_flying = Typing::dual(Electric, Flying);
		assert!(effectiveness(Ground, &electric_flying).is_immune());
		assert_eq!(effectiveness(Ground, &electric_flying).apply(500), 0);
	}

	/// Applying a multiplier is exact and never silently zeroes a resisted hit.
	#[test]
	fn applying_multipliers_is_exact() {
		let neutral = Effectiveness::NEUTRAL;
		assert_eq!(neutral.apply(100), 100);
		assert_eq!(neutral.with_stab().apply(100), 150);

		let quarter = Effectiveness::NEUTRAL.times(1, 4);
		assert_eq!(quarter.apply(100), 25);
		// a tiny hit stays at 1 rather than rounding away
		assert_eq!(quarter.apply(2), 1);

		assert_eq!(Effectiveness::IMMUNE.apply(9999), 0);
	}

	/// STAB and effectiveness compose in either order.
	#[test]
	fn stab_composes_with_effectiveness() {
		use Type::*;
		let e = effectiveness(Fire, &Typing::mono(Grass)).with_stab();
		assert_eq!(e.as_f32(), 3.0);
		assert_eq!(e.apply(40), 120);
	}
}
