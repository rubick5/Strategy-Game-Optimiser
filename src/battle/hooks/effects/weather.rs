//! Weather, expressed as hooks.
//!
//! This replaces the half-written `end_turn_resolution::subside_weather` /
//! `deal_weather_damage` pair, including the `todo!()` that would have panicked
//! the moment anything called it.

use std::any::Any;
use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::event::Event;
use crate::battle::hooks::effects::fraction_of_max;
use crate::battle::hooks::handler::{HookCtx, HookDef};
use crate::battle::hooks::order;
use crate::battle::hooks::trigger::{Trigger, TriggerKind};
use crate::battle::state::weather::{TimedWeather, Weather};
use crate::model::typing::Type;

/// Sandstorm chips 1/16 max HP a turn.
const SANDSTORM_FRACTION: u32 = 16;

/// The hooks a given weather installs.
pub fn hooks(weather: Weather) -> &'static [HookDef] {
	match weather {
		Weather::Sandstorm => SANDSTORM,
		Weather::HarshSun | Weather::Rain | Weather::Snow => INERT_WEATHER,
	}
}

static SANDSTORM: &[HookDef] = &[
	HookDef::reactive(TriggerKind::Residual, order::WEATHER_DAMAGE, sandstorm_damage),
	HookDef::reactive(TriggerKind::TurnEnd, order::WEATHER_COUNTDOWN, countdown),
];

/// Weathers that currently do nothing but still have to expire.
static INERT_WEATHER: &[HookDef] = &[HookDef::reactive(
	TriggerKind::TurnEnd,
	order::WEATHER_COUNTDOWN,
	countdown,
)];

/// Chip every creature standing in the sandstorm.
fn sandstorm_damage(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	for pos in ctx.battle_state.field.all_field_positions() {
		let mon = match ctx.battle_state.get_mon(pos) {
			Some(mon) => mon,
			None => continue,
		};
		if !mon.is_alive() || mon.get_typing(ctx.registry).contains(Type::Rock) || mon.get_typing(ctx.registry).contains(Type::Ground) || mon.get_typing(ctx.registry).contains(Type::Steel) {
			continue;
		}
		events.push_back(Event::DealDamage {
			amount: fraction_of_max(mon.max_hp, SANDSTORM_FRACTION),
			target: pos,
			source: None,
		});
	}
}

/// Tick the weather down, clearing it when it runs out.
///
/// Runs on `TurnEnd` rather than `Residual` so that the weather which was up
/// during the turn is the weather that dealt damage — the old
/// `END_OF_TURN_ORDER` had `WeatherSubsides` before `WeatherDamage`, which would
/// have let a sandstorm expire and then still chip on its way out.
fn countdown(ctx: &HookCtx, _trigger: &Trigger, events: &mut VecDeque<Event>, _rng: &mut dyn RngCore) {
	let next = match ctx.battle_state.weather {
		None => return,
		Some(TimedWeather { turns_left: 0, .. }) | Some(TimedWeather { turns_left: 1, .. }) => None,
		Some(TimedWeather { weather, turns_left }) => Some(TimedWeather {
			weather,
			turns_left: turns_left - 1,
		}),
	};
	events.push_back(Event::SetWeather { weather: next });
}
