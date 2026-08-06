use crate::{battle::state::{BattleState, CreatureState, RosterId, TEAM_SIZE}, model::{registry::Registry, speciesdata::Stat}};

const ATTACK_SCALAR: f32 = 200.0;
const SPEED_SCALAR: f32 = 200.0;
const DEFENSE_SCALAR: f32 = 200.0;
const HP_SCALAR: f32 = 200.0;

const MON_COUNT: usize = TEAM_SIZE + TEAM_SIZE;

const MON_ENCODING_LEN: usize = 7;

const FIELD_ENCODING_LEN: usize = 1;

pub const TOTAL_ENCODING_LEN: usize = MON_COUNT * MON_ENCODING_LEN + FIELD_ENCODING_LEN;

pub fn encode(battle_state: &BattleState, registry: &Registry, replacement: bool) -> Vec<f32> {
	let field_mons = battle_state.field.all_field_mons();
	let front: Vec<usize> = field_mons.iter().map(|x| x.0).collect();
	// all this does is put all the field mons first in the weights before
	// the mons in the back it just looks super complicated bcz rust
	let ordered = front.iter().copied()
		.chain((0..MON_COUNT).filter(|x| !front.contains(x)));

	let mut y: Vec<f32> = ordered
		.flat_map(|rid| {
			let creature = battle_state.roster.get_mon(RosterId(rid));
			encode_mon(creature, registry)
		})
		.collect();
	y.push(replacement as u32 as f32);
	assert!(y.len() == TOTAL_ENCODING_LEN);
	y
}

fn encode_mon(op_mon: Option<&CreatureState>, registry: &Registry) -> Vec<f32> {
	match op_mon {
		Some(creature) => {
			let attack = creature.get_stat(Stat::Attack, registry) as f32;
			let defense = creature.get_stat(Stat::Defense, registry) as f32;
			let speed = creature.get_stat(Stat::Speed, registry) as f32;
			let current_hp = creature.current_hp as f32;
			let attack_stage = creature.stat_changes.attack as f32;
			let defense_stage = creature.stat_changes.defense as f32;
			let speed_stage = creature.stat_changes.speed as f32;

			let v = vec![
				attack / ATTACK_SCALAR,
				defense / DEFENSE_SCALAR,
				speed / SPEED_SCALAR,
				current_hp / HP_SCALAR,
				attack_stage,
				defense_stage,
				speed_stage,
			];
			assert!(v.len() == MON_ENCODING_LEN);
			v
		},
		None => vec![0.0; MON_ENCODING_LEN]
	}
	
}