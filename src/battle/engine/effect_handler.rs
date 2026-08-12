use rand::{Rng, RngCore};

use crate::{battle::{event::Event, state::{field::PositionId, non_volatile_status::NonVolatileStatus::{Burn, Poison}}}, model::effect::Effect};


// this function will get insanely fat as we add in more effects like protecting,
// pivoting, stat lowering, so many things...
pub(in crate::battle::engine) fn effect_to_event(effect: &Effect, target: PositionId, rng: &mut dyn RngCore) -> Option<Event> {
	match effect {
		Effect::PoisonChance { chance } => {
			if rng.random_range(1..=100) < *chance {
				Some(Event::ApplyNonVolStatus { status: Poison, target })
			} else {
				None
			}
		}
		Effect::BurnChance { chance } => {
			if rng.random_range(1..=100) < *chance {
				Some(Event::ApplyNonVolStatus { status: Burn, target })
			} else {
				None
			}
		},
	}
}