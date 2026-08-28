use rand::{Rng, RngCore};

use crate::{battle::{event::Event, state::{field::PositionId, non_volatile_status::NonVolatileStatus, volatile::{Volatile, VolatileKind}}}, model::effect::Effect};
use crate::battle::state::battle_state::BattleState;

/// Confusion runs for 1-4 turns.
const CONFUSION_MIN_TURNS: u8 = 1;
const CONFUSION_MAX_TURNS: u8 = 4;
/// Taunt runs for 3 turns.
const TAUNT_TURNS: u8 = 3;
/// A Substitute costs, and is worth, this fraction of the user's max HP.
const SUBSTITUTE_FRACTION: u32 = 4;

/// Roll an N-percent chance.
///
/// Was `rng.random_range(1..=100) < chance`, which is off by one: a
/// `chance: 100` effect failed on a roll of exactly 100, and every effect was
/// one percentage point less likely than it claimed.
fn rolls_under(chance: u8, rng: &mut dyn RngCore) -> bool {
	rng.random_range(1..=100) <= chance
}

/// Turn one of a move's riders into the events it should queue.
///
/// Returns a list rather than a single event because some effects are two things
/// at once — a Substitute is HP spent *and* a decoy created.
///
/// `user` matters now: Leech Seed has to record who planted it, and self-targeting
/// effects need to know whether the target really is the user.
pub(in crate::battle::engine) fn effect_to_events(
	effect: &Effect,
	user: PositionId,
	target: PositionId,
	battle_state: &BattleState,
	rng: &mut dyn RngCore,
) -> Vec<Event> {
	let status = |status: NonVolatileStatus| vec![Event::ApplyNonVolStatus { status, target }];
	let volatile = |volatile: Volatile| vec![Event::ApplyVolatile { target, volatile }];

	match effect {
		Effect::PoisonChance { chance } => {
			if rolls_under(*chance, rng) { status(NonVolatileStatus::Poison) } else { vec![] }
		}
		Effect::BadPoisonChance { chance } => {
			if rolls_under(*chance, rng) { status(NonVolatileStatus::BadPoison) } else { vec![] }
		}
		Effect::BurnChance { chance } => {
			if rolls_under(*chance, rng) { status(NonVolatileStatus::Burn) } else { vec![] }
		}
		Effect::ParalysisChance { chance } => {
			if rolls_under(*chance, rng) { status(NonVolatileStatus::Paralysis) } else { vec![] }
		}

		Effect::ConfusionChance { chance } => {
			if !rolls_under(*chance, rng) {
				return vec![];
			}
			let turns = rng.random_range(CONFUSION_MIN_TURNS..=CONFUSION_MAX_TURNS);
			volatile(Volatile::lasting(VolatileKind::Confusion, turns))
		}

		Effect::FlinchChance { chance } => {
			// A flinch only means anything if the target has not moved yet this
			// turn. Commands resolve one at a time, so simply applying it works:
			// if the target already acted, the volatile is swept at the next turn
			// start having done nothing.
			if rolls_under(*chance, rng) {
				volatile(Volatile::new(VolatileKind::Flinch))
			} else {
				vec![]
			}
		}

		Effect::Taunt => volatile(Volatile::lasting(VolatileKind::Taunt, TAUNT_TURNS)),

		Effect::LeechSeed => {
			// Re-seeding an already-seeded target does nothing.
			if creature_has(battle_state, target, VolatileKind::LeechSeed) {
				return vec![];
			}
			// The planter's position is stored on the volatile, so the drain hook
			// knows where to send the HP without searching for it.
			volatile(Volatile::new(VolatileKind::LeechSeed).with_value(user.0 as u32))
		}

		Effect::Substitute => {
			let mon = match battle_state.get_mon(target) {
				Some(mon) => mon,
				None => return vec![],
			};
			let cost = (mon.max_hp / SUBSTITUTE_FRACTION).max(1);
			// Fails outright if one is already up, or if paying for it would be
			// fatal — you cannot faint yourself making a Substitute.
			if mon.volatiles.has(VolatileKind::Substitute) || mon.current_hp <= cost {
				return vec![];
			}
			vec![
				Event::DealDamage { amount: cost, target, source: None },
				Event::ApplyVolatile {
					target,
					volatile: Volatile::new(VolatileKind::Substitute).with_value(cost),
				},
			]
		}

		Effect::Protect => volatile(Volatile::new(VolatileKind::Protect)),
	}
}

fn creature_has(battle_state: &BattleState, pos: PositionId, kind: VolatileKind) -> bool {
	battle_state.get_mon(pos).map_or(false, |mon| mon.volatiles.has(kind))
}
