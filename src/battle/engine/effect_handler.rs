use rand::{Rng, RngCore};

use crate::{battle::{event::Event, state::{field::PositionId, non_volatile_status::NonVolatileStatus::{Burn, Poison}}}, model::effect::Effect};

/// Roll an N-percent chance.
///
/// Was `rng.random_range(1..=100) < chance`, which is off by one: a
/// `chance: 100` effect failed on a roll of exactly 100, and every effect was
/// one percentage point less likely than it claimed.
fn rolls_under(chance: u8, rng: &mut dyn RngCore) -> bool {
	rng.random_range(1..=100) <= chance
}

// this function no longer has to grow forever: an effect that needs to *react*
// to something rather than fire at move time belongs in a hook
// (see battle::hooks::effects), not in this match.
pub(in crate::battle::engine) fn effect_to_event(effect: &Effect, target: PositionId, rng: &mut dyn RngCore) -> Option<Event> {
	match effect {
		Effect::PoisonChance { chance } => {
			if rolls_under(*chance, rng) {
				Some(Event::ApplyNonVolStatus { status: Poison, target })
			} else {
				None
			}
		}
		Effect::BurnChance { chance } => {
			if rolls_under(*chance, rng) {
				Some(Event::ApplyNonVolStatus { status: Burn, target })
			} else {
				None
			}
		},
	}
}
