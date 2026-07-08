use crate::{battle::state::{BattleState, PokemonState, TEAM_SIZE}, model::{registry::Registry, speciesdata::Stat}};

const ATTACK_SCALAR: f32 = 200.0;
const SPEED_SCALAR: f32 = 200.0;
const DEFENSE_SCALAR: f32 = 200.0;
const HP_SCALAR: f32 = 200.0;

const MON_COUNT: usize = TEAM_SIZE + TEAM_SIZE;

const MON_ENCODING_LEN: usize = 7;

pub const TOTAL_ENCODING_LEN: usize = MON_COUNT * MON_ENCODING_LEN;

pub fn encode(battle_state: &BattleState, registry: &Registry) -> Vec<f32> {
	let v: Vec<f32> = vec![];
	let y: Vec<f32> = battle_state.roster.all_mons().flat_map(|mon| encode_mon(mon, registry)).collect();
	assert!(y.len() == TOTAL_ENCODING_LEN);
	return y;
}

fn encode_mon(op_mon: Option<&PokemonState>, registry: &Registry) -> Vec<f32> {
	match op_mon {
		Some(mon) => {
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
		},
		None => {
			vec![0.0; MON_ENCODING_LEN]
		},
	}
	
}