use crate::battle::state::{
	battle_state::BattleState, field::PositionId, non_volatile_status::NonVolatileStatus,
	roster::RosterId, weather::TimedWeather,
};

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
		/// Who is responsible, when anyone is. `None` for chip damage from
		/// weather or status.
		///
		/// Carried on the event so that `Trigger::AfterDamage` can name the
		/// attacker — which is what a Rough Skin or Rocky Helmet hook needs in
		/// order to hit back.
		source: Option<PositionId>,
	},
	Heal {
		amount: u32,
		target: PositionId,
	},
	Switch {
		current: PositionId,
		new: RosterId,
	},
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
