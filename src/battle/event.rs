use crate::battle::state::{
	battle_state::BattleState, field::PositionId, non_volatile_status::NonVolatileStatus,
	roster::RosterId, weather::TimedWeather,
};
use crate::battle::state::volatile::{Volatile, VolatileKind};
use crate::model::pmove::MoveId;

/// Who caused a hit, and with what.
///
/// Carried on damage so `Trigger::AfterDamage` can name both. Rough Skin needs
/// the attacker to hit back and the move to check whether it made contact —
/// without this it would have to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DamageSource {
	pub attacker: PositionId,
	pub move_id: MoveId,
}

/// The only way anything in the battle changes.
///
/// Hooks cannot touch the battle state; they queue these. That means every
/// mutation goes through `engine::execute_event`, which is also the one place
/// that broadcasts what just happened — so a new effect can never change the
/// state without the rest of the system hearing about it.
///
/// `Copy` so the engine can inspect an event (to gate it through a query, say)
/// and still queue it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
		/// `None` for chip damage from weather or status.
		source: Option<DamageSource>,
	},
	Heal {
		amount: u32,
		target: PositionId,
	},
	Switch {
		current: PositionId,
		new: RosterId,
	},
	/// Applying `NoStatus` is how an effect *cures* a creature.
	ApplyNonVolStatus {
		status: NonVolatileStatus,
		target: PositionId,
	},
	/// Set, change or (with `None`) clear the field weather.
	SetWeather {
		weather: Option<TimedWeather>,
	},
	/// A creature reached 0 HP. Queued by the engine rather than by effects, so
	/// that `AfterFaint` broadcasts exactly once per faint.
	Faint {
		target: PositionId,
	},
	/// Add a volatile, replacing any existing one of the same kind.
	ApplyVolatile {
		target: PositionId,
		volatile: Volatile,
	},
	RemoveVolatile {
		target: PositionId,
		kind: VolatileKind,
	},
}

impl Event {
	/// Damage equal to `max_hp / denominator`, floored at 1.
	///
	/// Replaces the old `deal_percent_damage`, which was based on *current* HP —
	/// chip damage in the games is a fraction of maximum HP, so a creature at
	/// 1 HP took a fraction of 1 rather than its usual tick.
	pub fn max_hp_fraction(
		battle_state: &BattleState,
		denominator: u32,
		target: PositionId,
	) -> Result<Self, Box<dyn std::error::Error>> {
		let mon = battle_state
			.get_mon(target)
			.ok_or("position doesn't exist in the field")?;
		if denominator == 0 {
			return Err("denominator must be non-zero".into());
		}
		Ok(Event::DealDamage {
			amount: (mon.max_hp / denominator).max(1),
			target,
			source: None,
		})
	}
}
