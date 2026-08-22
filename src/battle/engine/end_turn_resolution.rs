use std::error::Error;

use crate::battle::state::battle_state::BattleState;

pub enum EndTurnOrder {
	WeatherSubsides,
	WeatherDamage,
	FutureSight,
	Wish,
	Poison,
	Burn,
	BindingMoves,
}

pub fn resolve_end_of_turn(battle_state: &mut BattleState) -> Result<(), Box<dyn Error>> {
	Ok(())
}