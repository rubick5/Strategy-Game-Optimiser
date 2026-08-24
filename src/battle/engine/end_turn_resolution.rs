use std::error::Error;

use eframe::wgpu::wgc::registry;

use crate::battle::engine::engine;
use crate::battle::event::Event;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::weather::{TimedWeather, Weather};
use crate::model::registry::Registry;

pub enum EndTurnOrder {
	WeatherSubsides,
	WeatherDamage,
	FutureSight,
	Wish,
	Poison,
	Burn,
	BindingMoves,
}

use EndTurnOrder::*;

const END_OF_TURN_ORDER: [EndTurnOrder; 7] = [
	WeatherSubsides,
	WeatherDamage,
	FutureSight,
	Wish,
	Poison,
	Burn,
	BindingMoves,
];

// we maybe should do this by having it push events to an event queue so the battle
// state handles it more naturally...

pub fn resolve_end_of_turn(battle_state: &mut BattleState, registry: &Registry) -> Result<(), Box<dyn Error>> {
	for turn in END_OF_TURN_ORDER {
		match turn {
			WeatherSubsides => subside_weather(battle_state),
			WeatherDamage => todo!(), //deal_weather_damage(battle_state, registry).iter().for_each(|event| engine::execute_event(battle_state, event, fainted)),
			FutureSight => {},
			Wish => {},
			Poison => {},
			Burn => {},
			BindingMoves => {},
		}
	}
	Ok(())
}

fn subside_weather(battle_state: &mut BattleState) {
	match &mut battle_state.weather {
		None => {},
		Some(TimedWeather { weather: _, turns_left: 1 }) => battle_state.weather = None,
		Some(TimedWeather { weather: _, turns_left }) => *turns_left -= 1,
	}
}

/** Returns vector of events that deal weather damage to all the creatures on the field
 * 
 */
fn deal_weather_damage(battle_state: &mut BattleState, registry: &Registry) -> Vec<Event> {
	battle_state.all_field_mons_ordered(registry).iter().map(|pid| {
		weather_damage_event(battle_state, *pid)
	}).flatten().collect()
}

fn weather_damage_event(battle_state: &BattleState, target: PositionId) -> Option<Event> {
	match battle_state.weather {
		Some(TimedWeather { weather: Weather::Sandstorm, turns_left: _ }) => {
			Some(Event::deal_percent_damage(battle_state, 12.5, target).expect("must deal weather damage to existing mon"))
		}
		_ => None,
	}
}

/** Returns the damage event associated with the non volatile status given
 * 
 *  Panics if the positionid doesn't point to any mon
 */
fn status_damage_event(battle_state: &BattleState, status: NonVolatileStatus, target: PositionId) -> Option<Event> {
	let monstate = battle_state.get_mon(target).expect("mon needed");
	match (monstate.non_vol_status == status, status) {
		(false, _) => None,
		(true, NonVolatileStatus::Poison) => Some(Event::deal_percent_damage(battle_state, 12.5, target).expect("must deal status damage to existig mon")),
		(true, NonVolatileStatus::NoStatus) => None,
		(true, NonVolatileStatus::BadPoison) => None, // TODO
		(true, NonVolatileStatus::Burn) => Some(Event::deal_percent_damage(battle_state, 6.25, target).expect("must deal burn damage to existing mon")),
	}
}