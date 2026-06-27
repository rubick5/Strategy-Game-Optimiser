use crate::{battle::state::{BattleState, PokemonState}, model::{registry::Registry, speciesdata::Stat}};

const ATTACK_SCALAR: f32 = 1000.0;
const SPEED_SCALAR: f32 = 1000.0;
const DEFENSE_SCALAR: f32 = 1000.0;
const HP_SCALAR: f32 = 1000.0;

fn encode(battle_state: BattleState, registry: &Registry) -> Vec<f32> {
	battle_state.mons.into_iter().flat_map(|(_, mon)| encode_mon(mon, registry)).collect()
}

fn encode_mon(mon: PokemonState, registry: &Registry) -> Vec<f32> {
	let attack = mon.get_stat(Stat::Attack, registry) as f32;
	let defense = mon.get_stat(Stat::Defense, registry) as f32;
	let speed = mon.get_stat(Stat::Speed, registry) as f32;
	let current_hp = mon.current_hp as f32;
	let attack_stage = mon.stat_changes.attack as f32;
	let defense_stage = mon.stat_changes.defense as f32;
	let speed_stage = mon.stat_changes.speed as f32;
	
	vec![ 
		attack / ATTACK_SCALAR,
		defense / DEFENSE_SCALAR,
		speed / SPEED_SCALAR,
		current_hp / HP_SCALAR,
		attack_stage,
		defense_stage,
		speed_stage,
	]
}