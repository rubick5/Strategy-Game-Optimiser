use std::error::Error;

use crate::battle::event::Event;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::non_volatile_status::NonVolatileStatus;
use crate::battle::state::weather::{TimedWeather, Weather};

pub enum EndTurnOrder {
	WeatherSubsides,
	WeatherDamage,
	FutureSight,
	Wish,
	Poison,
	Burn,
	BindingMoves,
}

// we maybe should do this by having it push events to an event queue so the battle
// state handles it more naturally...

pub fn resolve_end_of_turn(battle_state: &mut BattleState) -> Result<(), Box<dyn Error>> {
	Ok(())
}

fn subside_weather(battle_state: &mut BattleState) {
	match &mut battle_state.weather {
		None => {},
		Some(TimedWeather { weather: _, turns_left: 1 }) => battle_state.weather = None,
		Some(TimedWeather { weather: _, turns_left }) => *turns_left -= 1,
	}
}

fn deal_weather_damage(battle_state: &mut BattleState) {
	if let Some(TimedWeather { weather, turns_left: _ }) = battle_state.weather {
		// we dont need the match but if we ever want to add hail this makes it easier...
	}
}

fn weather_damage_event(battle_state: &BattleState, weather: Weather, target: PositionId) -> Option<Event> {
	match weather {
		Weather::Sandstorm => {
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