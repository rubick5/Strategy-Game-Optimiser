use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::command::MoveCommand;
use crate::battle::engine::calculate_damage::calculate_damage;
use crate::battle::engine::effect_handler::effect_to_event;
use crate::battle::event::Event;
use crate::battle::hooks::{HookTable, Trigger};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::model::registry::Registry;
use crate::model::speciesdata::Stat;

/// Turn a chosen move into queued events.
///
/// Every decision this function makes now passes through the hook table first,
/// which is what lets an ability or item interfere without this file knowing it
/// exists:
///
/// 1. `allows_move` — can the user act at all? (paralysis, Truant, Taunt)
/// 2. `BeforeMove` — anything that wants to react to the attempt
/// 3. `effective_stat` — Attack and Defense as they apply *to this hit*
/// 4. `final_damage` — the rolled number, before it becomes an event
/// 5. `allows_status` — can a secondary status actually land?
/// 6. `AfterMove` — anything that wants to react to the attempt finishing
pub(in crate::battle::engine) fn execute_move(
	move_command: MoveCommand,
	registry: &Registry,
	battle_state: &BattleState,
	events: &mut VecDeque<Event>,
	rng: &mut dyn RngCore,
	hooks: &HookTable,
) {
	let mv = registry.get_move(move_command.move_id);
	let user: &CreatureState = match battle_state.get_mon(move_command.user) {
		Some(user) => user,
		None => return,
	};

	// (1) Anything can veto the move outright.
	if !hooks.allows_move(battle_state, registry, move_command.user, move_command.move_id) {
		return;
	}

	// (2)
	hooks.dispatch(
		Trigger::BeforeMove {
			user: move_command.user,
			move_id: move_command.move_id,
		},
		battle_state,
		registry,
		events,
		rng,
	);

	// (3) The user's Attack as it applies to this move. A burn halves it here,
	// leaving `CreatureState::get_stat` (which the RL encoder reads) untouched.
	let user_attack = hooks.effective_stat(
		battle_state,
		registry,
		move_command.user,
		Stat::Attack,
		user.get_stat(Stat::Attack, registry),
	);

	for target_pos in move_command.targets {
		let target: &CreatureState = match battle_state.get_mon(target_pos) {
			Some(target) => target,
			None => continue,
		};
		let target_defense = hooks.effective_stat(
			battle_state,
			registry,
			target_pos,
			Stat::Defense,
			target.get_stat(Stat::Defense, registry),
		);

		//log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

		// (4)
		let amount = hooks.final_damage(
			battle_state,
			registry,
			move_command.user,
			target_pos,
			move_command.move_id,
			calculate_damage(user_attack, target_defense, mv.base_power),
		);

		events.push_back(Event::DealDamage {
			amount,
			target: target_pos,
			source: Some(move_command.user),
		});

		for effect in mv.effects.iter() {
			let event = match effect_to_event(effect, target_pos, rng) {
				Some(event) => event,
				None => continue,
			};

			// (5) A status only lands if nothing objects. The "already has a
			// status" rule lives on the existing status, not here.
			if let Event::ApplyNonVolStatus { status, target } = event {
				if !hooks.allows_status(battle_state, registry, target, status) {
					continue;
				}
			}

			events.push_back(event);
		}
	}

	// (6)
	hooks.dispatch(
		Trigger::AfterMove {
			user: move_command.user,
			move_id: move_command.move_id,
		},
		battle_state,
		registry,
		events,
		rng,
	);
}
