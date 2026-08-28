//! Where subscriptions come from.
//!
//! This is the single place that answers "given this battle state, who is
//! listening?". Everything else in the hook system is generic machinery; this
//! file is the only part that knows what kinds of thing can own a hook.
//!
//! ## Adding a new kind of hook owner
//!
//! Held items are the obvious next one, and the seam is marked below. It is
//! three lines here plus an `effects/item.rs` shaped exactly like
//! `effects/ability.rs`. The engine does not change — it is already broadcasting
//! every moment an item could want.

use crate::battle::hooks::effects::{ability, status, volatile, weather};
use crate::battle::hooks::handler::HookDef;
use crate::battle::hooks::source::HookSource;
use crate::battle::state::battle_state::BattleState;
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;

/// One entry per hook-owning entity currently on the field: who it is, what it
/// subscribes to, and how fast its owner is (for tie-breaking).
pub type Subscriber = (HookSource, &'static [HookDef], u32);

/// Walk the battle state and gather every live subscription.
///
/// Fainted creatures are skipped: a creature at 0 HP should not be taking poison
/// damage or modifying anything. (The one exception is handled in the engine —
/// `AfterFaint` is broadcast *before* the table is invalidated, so a creature's
/// own on-faint hooks still get to run.)
pub fn collect(battle_state: &BattleState, registry: &Registry) -> Vec<Subscriber> {
	let mut out: Vec<Subscriber> = Vec::new();

	for pos in battle_state.field.all_field_positions() {
		let mon = match battle_state.get_mon(pos) {
			Some(mon) => mon,
			None => continue,
		};
		if !mon.is_alive() {
			continue;
		}
		let speed = mon.get_stat(Stat::Speed, registry);

		let status_defs = status::hooks(mon.non_vol_status);
		if !status_defs.is_empty() {
			out.push((
				HookSource::Status { pos, status: mon.non_vol_status },
				status_defs,
				speed,
			));
		}

		for active in mon.volatiles.iter() {
			let defs = volatile::hooks(active.kind);
			if !defs.is_empty() {
				out.push((HookSource::Volatile { pos, kind: active.kind }, defs, speed));
			}
		}

		if let Some(ability_id) = mon.ability {
			let ability_defs = ability::hooks(ability_id);
			if !ability_defs.is_empty() {
				out.push((HookSource::Ability { pos }, ability_defs, speed));
			}
		}

		// SEAM: held items go here.
	}

	if let Some(timed) = battle_state.weather {
		let weather_defs = weather::hooks(timed.weather);
		if !weather_defs.is_empty() {
			out.push((HookSource::Weather, weather_defs, 0));
		}
	}

	out
}
