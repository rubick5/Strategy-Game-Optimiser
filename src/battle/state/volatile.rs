//! Volatile status: the temporary conditions that live on a creature only while
//! it is on the field.
//!
//! The distinction from [`NonVolatileStatus`](super::non_volatile_status) is the
//! one the games make: a burn survives switching out, confusion does not. So the
//! single most important rule here is that switching wipes the lot, which
//! `engine::execute_event` does in the `Switch` arm.
//!
//! Everything is stored in one flat list rather than as named fields, so adding a
//! new condition is a `VolatileKind` variant plus a hook table — no change to
//! `CreatureState`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VolatileKind {
	/// May hurt itself instead of acting. Lasts 1-4 turns.
	Confusion = 0,
	/// Cannot act this turn. Applied by a move that already resolved.
	Flinch = 1,
	/// Cannot use status moves for a few turns.
	Taunt = 2,
	/// A decoy that soaks attack damage until its own HP runs out.
	Substitute = 3,
	/// Drains HP each turn and gives it to whoever planted it.
	LeechSeed = 4,
	/// Blocks moves aimed at the holder for the rest of this turn.
	Protect = 5,
	/// "This creature loses its turn." Set at turn start by the things that roll
	/// for it — confusion and full paralysis — so that the actual veto stays a
	/// pure, deterministic query.
	Immobilised = 6,
}

pub const VOLATILE_COUNT: usize = 7;

impl VolatileKind {
	pub const ALL: [VolatileKind; VOLATILE_COUNT] = [
		VolatileKind::Confusion,
		VolatileKind::Flinch,
		VolatileKind::Taunt,
		VolatileKind::Substitute,
		VolatileKind::LeechSeed,
		VolatileKind::Protect,
		VolatileKind::Immobilised,
	];

	#[inline]
	pub fn index(self) -> usize {
		self as usize
	}

	/// Conditions that last exactly the turn they were applied on, and are swept
	/// at the start of the next one.
	pub fn is_turn_scoped(self) -> bool {
		matches!(
			self,
			VolatileKind::Flinch | VolatileKind::Protect | VolatileKind::Immobilised
		)
	}

	pub fn name(self) -> &'static str {
		match self {
			VolatileKind::Confusion => "Confused",
			VolatileKind::Flinch => "Flinched",
			VolatileKind::Taunt => "Taunted",
			VolatileKind::Substitute => "Substitute",
			VolatileKind::LeechSeed => "Leech Seed",
			VolatileKind::Protect => "Protected",
			VolatileKind::Immobilised => "Can't move",
		}
	}
}

impl std::fmt::Display for VolatileKind {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.name())
	}
}

/// One active condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Volatile {
	pub kind: VolatileKind,
	/// Turns still to run. `None` means "until this creature leaves the field".
	pub turns_left: Option<u8>,
	/// Whatever the condition needs to remember: Substitute stores its remaining
	/// HP here, Leech Seed stores the position that planted it.
	pub value: u32,
}

impl Volatile {
	pub fn new(kind: VolatileKind) -> Self {
		Volatile { kind, turns_left: None, value: 0 }
	}

	pub fn lasting(kind: VolatileKind, turns: u8) -> Self {
		Volatile { kind, turns_left: Some(turns), value: 0 }
	}

	pub fn with_value(mut self, value: u32) -> Self {
		self.value = value;
		self
	}
}

/// Every volatile currently on one creature.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Volatiles {
	entries: Vec<Volatile>,
}

impl Volatiles {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn get(&self, kind: VolatileKind) -> Option<&Volatile> {
		self.entries.iter().find(|v| v.kind == kind)
	}

	pub fn has(&self, kind: VolatileKind) -> bool {
		self.get(kind).is_some()
	}

	/// The condition's stored counter, or 0 if it is not present.
	pub fn value(&self, kind: VolatileKind) -> u32 {
		self.get(kind).map(|v| v.value).unwrap_or(0)
	}

	pub fn set_value(&mut self, kind: VolatileKind, value: u32) {
		if let Some(v) = self.entries.iter_mut().find(|v| v.kind == kind) {
			v.value = value;
		}
	}

	/// Add a condition, replacing any existing one of the same kind.
	///
	/// Re-applying rather than stacking is what the games do — a second Taunt
	/// refreshes the timer, it does not run two Taunts at once.
	pub fn add(&mut self, volatile: Volatile) {
		self.remove(volatile.kind);
		self.entries.push(volatile);
	}

	pub fn remove(&mut self, kind: VolatileKind) {
		self.entries.retain(|v| v.kind != kind);
	}

	/// Wipe everything. Called when a creature leaves the field — this is the
	/// whole point of the volatile/non-volatile split.
	pub fn clear(&mut self) {
		self.entries.clear();
	}

	/// Drop the conditions that only ever last the turn they were applied on.
	pub fn clear_turn_scoped(&mut self) {
		self.entries.retain(|v| !v.kind.is_turn_scoped());
	}

	/// Count down every timed condition and drop the ones that ran out.
	pub fn tick(&mut self) {
		for v in self.entries.iter_mut() {
			if let Some(turns) = v.turns_left {
				v.turns_left = Some(turns.saturating_sub(1));
			}
		}
		self.entries.retain(|v| v.turns_left != Some(0));
	}

	pub fn iter(&self) -> impl Iterator<Item = &Volatile> + '_ {
		self.entries.iter()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn adding_the_same_kind_replaces_rather_than_stacks() {
		let mut v = Volatiles::new();
		v.add(Volatile::lasting(VolatileKind::Taunt, 3));
		v.add(Volatile::lasting(VolatileKind::Taunt, 5));
		assert_eq!(v.iter().count(), 1);
		assert_eq!(v.get(VolatileKind::Taunt).unwrap().turns_left, Some(5));
	}

	#[test]
	fn ticking_expires_timed_conditions_and_leaves_open_ended_ones() {
		let mut v = Volatiles::new();
		v.add(Volatile::lasting(VolatileKind::Confusion, 2));
		v.add(Volatile::new(VolatileKind::LeechSeed));

		v.tick();
		assert!(v.has(VolatileKind::Confusion), "two turns should survive one tick");
		v.tick();
		assert!(!v.has(VolatileKind::Confusion), "it should have run out");
		assert!(v.has(VolatileKind::LeechSeed), "open-ended conditions never tick away");
	}

	#[test]
	fn turn_scoped_conditions_are_swept_separately() {
		let mut v = Volatiles::new();
		v.add(Volatile::new(VolatileKind::Flinch));
		v.add(Volatile::new(VolatileKind::Immobilised));
		v.add(Volatile::lasting(VolatileKind::Taunt, 3));

		v.clear_turn_scoped();
		assert!(!v.has(VolatileKind::Flinch));
		assert!(!v.has(VolatileKind::Immobilised));
		assert!(v.has(VolatileKind::Taunt), "a multi-turn condition must survive");
	}

	#[test]
	fn values_round_trip_and_default_to_zero() {
		let mut v = Volatiles::new();
		assert_eq!(v.value(VolatileKind::Substitute), 0);
		v.add(Volatile::new(VolatileKind::Substitute).with_value(37));
		assert_eq!(v.value(VolatileKind::Substitute), 37);
		v.set_value(VolatileKind::Substitute, 12);
		assert_eq!(v.value(VolatileKind::Substitute), 12);
	}

	#[test]
	fn volatile_indices_match_their_position() {
		for (i, kind) in VolatileKind::ALL.iter().enumerate() {
			assert_eq!(kind.index(), i, "{:?} is out of order", kind);
		}
	}
}
