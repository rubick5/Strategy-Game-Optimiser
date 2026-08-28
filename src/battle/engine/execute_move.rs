use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::command::MoveCommand;
use crate::battle::engine::calculate_damage::calculate_damage;
use crate::battle::engine::effect_handler::effect_to_event;
use crate::battle::event::{DamageSource, Event};
use crate::battle::hooks::{HookTable, Trigger};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::model::registry::Registry;

/// Turn a chosen move into queued events.
///
/// Every decision this function makes now passes through the hook table first,
/// which is what lets an ability or item interfere without this file knowing it
/// exists:
///
/// 1. `allows_move` — can the user act at all? (Taunt, Truant, full paralysis)
/// 2. `BeforeMove` — anything that wants to react to the attempt
/// 3. `effective_stat` — the attacking and defending stats *for this hit*
/// 4. `final_damage` — the rolled number, before it becomes an event
/// 5. `allows_status` — can a secondary status actually land?
/// 6. `AfterMove` — anything that wants to react to the attempt finishing
///
/// Which stats step 3 asks about now depends on the move: physical moves use
/// Attack vs Defense, special moves Special Attack vs Special Defense, and
/// status moves skip damage entirely rather than queueing a zero-damage hit.
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

	// (3) The user's offensive stat as it applies to this move. A burn halves
	// Attack here, leaving `CreatureState::get_stat` (which the RL encoder reads)
	// untouched. `None` for a status move, which has no offensive stat at all.
	let user_offense = mv.move_type.attacking_stat().map(|stat| {
		hooks.effective_stat(
			battle_state,
			registry,
			move_command.user,
			stat,
			user.get_stat(stat, registry),
		)
	});

	for target_pos in move_command.targets {
		let target: &CreatureState = match battle_state.get_mon(target_pos) {
			Some(target) => target,
			None => continue,
		};

		//log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

		// Status moves apply their effects and nothing else. Previously they
		// queued `DealDamage { amount: 0 }`, which woke every AfterDamage hook
		// for a hit that never landed.
		if let (true, Some(attack), Some(defense_stat)) = (
			mv.move_type.is_damaging(),
			user_offense,
			mv.move_type.defending_stat(),
		) {
			let defense = hooks.effective_stat(
				battle_state,
				registry,
				target_pos,
				defense_stat,
				target.get_stat(defense_stat, registry),
			);

			// (4)
			let amount = hooks.final_damage(
				battle_state,
				registry,
				move_command.user,
				target_pos,
				move_command.move_id,
				calculate_damage(attack, defense, mv.base_power),
			);

			// A hook may have zeroed it (Levitate). Don't queue a no-op hit.
			if amount > 0 {
				events.push_back(Event::DealDamage {
					amount,
					target: target_pos,
					source: Some(DamageSource {
						attacker: move_command.user,
						move_id: move_command.move_id,
					}),
				});
			}
		}

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
