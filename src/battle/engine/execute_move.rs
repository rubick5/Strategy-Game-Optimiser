use std::collections::VecDeque;

use rand::RngCore;

use crate::battle::command::MoveCommand;
use crate::battle::engine::calculate_damage::calculate_damage;
use crate::battle::engine::effect_handler::effect_to_events;
use crate::battle::event::{DamageSource, Event};
use crate::battle::hooks::{HookTable, Trigger};
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::model::registry::Registry;
use crate::model::typing::{self, Effectiveness};

/// Turn a chosen move into queued events.
///
/// Every decision this function makes now passes through the hook table first,
/// which is what lets an ability or item interfere without this file knowing it
/// exists:
///
/// 1. `allows_move` — can the user act at all? (Taunt, Truant, full paralysis)
/// 2. `BeforeMove` — anything that wants to react to the attempt
/// 3. `effective_stat` — the attacking and defending stats *for this hit*
/// 3a. `allows_hit` — may this move touch this target at all? (Protect)
/// 3b. the type chart, then `final_effectiveness` so abilities can alter it
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

	// STAB: a move matching one of the user's own types hits for x1.5.
	let user_typing = registry.get_species_data(user.species_id).typing;
	let stab = user_typing.contains(mv.element);

	for target_pos in move_command.targets {
		let target: &CreatureState = match battle_state.get_mon(target_pos) {
			Some(target) => target,
			None => continue,
		};

		//log_move_usage(&battle_state, registry, move_command.user, target_pos, mv.move_id);

		// (3a) Protect and friends. Checked before the chart and before any
		// effect rolls, so a blocked move does nothing whatsoever to this target.
		if !hooks.allows_hit(battle_state, registry, move_command.user, target_pos, mv.move_id) {
			continue;
		}

		// (3b) The type chart, then abilities that alter it.
		//
		// Only damaging moves are typed against the target; a status move lands
		// regardless of the chart, which is how the games work (Thunder Wave's
		// Ground immunity is a special case, not the general rule).
		let chart = if mv.move_type.is_damaging() {
			let target_typing = registry.get_species_data(target.species_id).typing;
			let base = typing::effectiveness(mv.element, &target_typing);
			hooks.final_effectiveness(
				battle_state,
				registry,
				move_command.user,
				target_pos,
				move_command.move_id,
				base,
			)
		} else {
			Effectiveness::NEUTRAL
		};

		// An immunity makes the whole move fail against this target — no damage
		// AND no secondary effects. Doing this before the effect rolls is the
		// reason effectiveness is its own query rather than a damage multiplier:
		// a Ground move should not be able to burn a Levitate holder for zero.
		if chart.is_immune() {
			continue;
		}

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

			// (4) Base formula, then the chart and STAB, then the damage query so
			// abilities and items scale the final figure.
			let multiplier = if stab { chart.with_stab() } else { chart };
			let typed = multiplier.apply(calculate_damage(attack, defense, mv.base_power));

			let amount = hooks.final_damage(
				battle_state,
				registry,
				move_command.user,
				target_pos,
				move_command.move_id,
				typed,
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
			// One rider can produce more than one event — a Substitute is HP
			// spent and a decoy raised.
			for event in effect_to_events(effect, move_command.user, target_pos, battle_state, rng) {
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
