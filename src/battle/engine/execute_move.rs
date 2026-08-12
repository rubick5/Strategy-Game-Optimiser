use rand::{Rng, RngCore};

use crate::battle::engine::calculate_damage::calculate_damage;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::battle::command::MoveCommand;
use crate::battle::event::Event;
use crate::model::effect::Effect::PoisonChance;
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;
use crate::battle::state::non_volatile_status::NonVolatileStatus::{Poison};

pub(in crate::battle::engine) fn execute_move(move_command: MoveCommand, registry: &Registry, battle_state: &BattleState, events: &mut Vec<Event>, rng: &mut dyn RngCore) {
	let mv = registry.get_move(move_command.move_id);
	let user: &CreatureState = battle_state.get_mon(move_command.user).unwrap();

	let user_attack = user.get_stat(Stat::Attack, registry);

	for target_pos in move_command.targets {
		let target: &CreatureState = battle_state.get_mon(target_pos).unwrap();
		let target_defense =  target.get_stat(Stat::Defense, registry);

		//log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

		events.push(Event::DealDamage {
			amount: calculate_damage(user_attack, target_defense, mv.base_power),
			target: target_pos
		});

		for effect in mv.effects.iter() {
			match effect {
				PoisonChance { chance: n } => {
					if rng.random_range(1..=100) <= *n {
						events.push(Event::ApplyNonVolStatus { status: Poison, target: target_pos })
					}
				},
				_ => {}
			}
		}
		
	}
}

