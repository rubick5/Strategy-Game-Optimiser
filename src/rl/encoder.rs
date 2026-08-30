//! Turning a battle state into a flat vector for the network.
//!
//! # Perspective
//!
//! Every encoding is taken *from a team's point of view*, passed in as `team`.
//! The layout is always:
//!
//! ```text
//! [ my actives | their actives | my bench | their bench | active movesets | field ]
//! ```
//!
//! so slot 0 is always "me" and the network never has to learn which side of the
//! board it happens to be sitting on. Before this, position 0 was encoded first
//! for both players, which meant an opponent agent read the board through the
//! learner's eyes — self-play was training the opponent to evaluate the *wrong*
//! side, and a trained net could only ever play as Team Zero.
//!
//! The cost is that a turn needs two encodings rather than one. That is real but
//! small next to two network forward passes, and [`encode_both`] halves it: the
//! per-creature blocks are perspective-independent, so it builds each one once
//! and assembles the two orderings from the same pieces.
//!
//! # Retraining
//!
//! `TOTAL_ENCODING_LEN` is unchanged by the perspective work, but the *meaning*
//! of every slot has moved. A net trained on the old layout will load and produce
//! confident nonsense. Retrain.

use crate::{
	battle::hooks::HookTable,
	battle::state::{
		TEAM_SIZE, Team, battle_state::BattleState, creature_state::CreatureState,
		non_volatile_status::{NonVolatileStatus, STATUS_COUNT},
		field::PositionId,
		roster::RosterId,
		volatile::{MAX_TOXIC_COUNTER, VOLATILE_COUNT, VolatileKind},
		weather::Weather,
	},
	model::{
		ability::{ABILITY_COUNT, AbilityId},
		pmove::{MoveType, PMove},
		registry::Registry,
		speciesdata::{STAT_COUNT, Stat},
		typing::{self, TYPE_COUNT, Type, Typing},
	},
	rl::moveslot::MOVESLOT_COUNT,
};

/// Stats are divided by this to land roughly in 0..1.
const STAT_SCALAR: f32 = 200.0;
/// Stat stages run -6..=+6.
const STAGE_SCALAR: f32 = 6.0;
const POWER_SCALAR: f32 = 100.0;
const PRIORITY_SCALAR: f32 = 3.0;
const WEATHER_TURNS_SCALAR: f32 = 8.0;

const MON_COUNT: usize = TEAM_SIZE + TEAM_SIZE;
/// Creatures on the field at once, across both sides.
const ACTIVE_COUNT: usize = 2;

const WEATHER_COUNT: usize = 4;

/// Per creature: 5 stats, HP fraction, 5 stat stages, status one-hot,
/// ability one-hot, type multi-hot, volatile multi-hot, and three counters
/// (substitute HP, toxic build-up, Protect streak).
///
/// The types and volatiles are multi-hots rather than one-hots because a
/// creature can have several at once and their order carries no meaning.
///
/// Volatiles have to be here or they are hidden dynamics: an agent that gets
/// Taunted would otherwise see its status move silently fail with no idea why.
pub const MON_ENCODING_LEN: usize =
	STAT_COUNT + 1 + STAT_COUNT + STATUS_COUNT + ABILITY_COUNT + TYPE_COUNT + VOLATILE_COUNT + 3;

/// Per move slot: power, is-special, is-status, priority, rider chance, and the
/// chart multiplier against the creature currently opposite.
///
/// That last one is the difference between the agent having to *infer* the type
/// chart from outcomes and being able to read it. The raw types are in the
/// encoding too, so it can still learn to generalise across match-ups it has not
/// seen; this just means it does not have to rediscover 18x18 matchups from
/// win/loss signal alone.
const MOVE_ENCODING_LEN: usize = 6;
/// The active creatures' movesets, mine first. Without this the agent picks
/// "slot 2" with no idea what slot 2 does.
const ACTIVE_MOVES_LEN: usize = ACTIVE_COUNT * MOVESLOT_COUNT * MOVE_ENCODING_LEN;

/// Weather one-hot, weather turns left, trick room, replacement flag.
const FIELD_ENCODING_LEN: usize = WEATHER_COUNT + 1 + 1 + 1;

pub const TOTAL_ENCODING_LEN: usize =
	MON_COUNT * MON_ENCODING_LEN + ACTIVE_MOVES_LEN + FIELD_ENCODING_LEN;

/// The roster slots in the order `team` should see them:
/// my actives, their actives, my bench, their bench.
///
/// Always returns exactly `MON_COUNT` entries, because every roster slot belongs
/// to exactly one team and is either active or benched.
fn perspective_order(battle_state: &BattleState, team: &Team) -> Vec<usize> {
	let actives_of = |t: &Team| -> Vec<usize> {
		battle_state
			.field
			.team_positions(t)
			.iter()
			.map(|pos| battle_state.field[*pos].0)
			.collect()
	};

	let foe = team.other();
	let mine_active = actives_of(team);
	let theirs_active = actives_of(&foe);

	let mut order: Vec<usize> = Vec::with_capacity(MON_COUNT);
	order.extend(mine_active.iter().copied());
	order.extend(theirs_active.iter().copied());
	order.extend(team.roster_ids().into_iter().filter(|r| !mine_active.contains(r)));
	order.extend(foe.roster_ids().into_iter().filter(|r| !theirs_active.contains(r)));

	debug_assert_eq!(order.len(), MON_COUNT, "perspective order must cover every roster slot");
	order
}

/// Encode the battle as `team` sees it.
pub fn encode(battle_state: &BattleState, registry: &Registry, replacement: bool, team: &Team) -> Vec<f32> {
	let order = perspective_order(battle_state, team);
	let mut hooks = HookTable::new();
	hooks.refresh(battle_state, registry);
	assemble(
		battle_state, registry, replacement, &order, &hooks, active_positions(battle_state, team),
		&|rid| encode_mon(battle_state.roster.get_mon(RosterId(rid)), registry),
	)
}

/// The viewer's active position and their opponent's.
fn active_positions(battle_state: &BattleState, team: &Team) -> (Option<PositionId>, Option<PositionId>) {
	(
		battle_state.field.team_positions(team).first().copied(),
		battle_state.field.team_positions(&team.other()).first().copied(),
	)
}

/// Both perspectives at once, sharing the per-creature work.
///
/// The per-creature blocks do not depend on who is looking, so they are built
/// once and the two orderings assembled from the same pieces. Returns
/// `(team_zero_view, team_one_view)`.
pub fn encode_both(
	battle_state: &BattleState,
	registry: &Registry,
	replacement: bool,
) -> (Vec<f32>, Vec<f32>) {
	let blocks: Vec<Vec<f32>> = (0..MON_COUNT)
		.map(|rid| encode_mon(battle_state.roster.get_mon(RosterId(rid)), registry))
		.collect();

	let lookup = |rid: usize| blocks[rid].clone();

	// One table serves both perspectives - it describes the board, not a viewer.
	let mut hooks = HookTable::new();
	hooks.refresh(battle_state, registry);

	let zero = assemble(
		battle_state, registry, replacement,
		&perspective_order(battle_state, &Team::Zero),
		&hooks, active_positions(battle_state, &Team::Zero), &lookup,
	);
	let one = assemble(
		battle_state, registry, replacement,
		&perspective_order(battle_state, &Team::One),
		&hooks, active_positions(battle_state, &Team::One), &lookup,
	);
	(zero, one)
}

/// Lay out one perspective, given the roster order it should see.
fn assemble(
	battle_state: &BattleState,
	registry: &Registry,
	replacement: bool,
	order: &[usize],
	hooks: &HookTable,
	actives: (Option<PositionId>, Option<PositionId>),
	mon_block: &dyn Fn(usize) -> Vec<f32>,
) -> Vec<f32> {
	let mut y: Vec<f32> = Vec::with_capacity(TOTAL_ENCODING_LEN);

	for rid in order {
		y.extend(mon_block(*rid));
	}

	// The first ACTIVE_COUNT entries are the creatures on the field, mine first.
	// Each one's moves are scored against the creature opposite it.
	let (mine, theirs) = actives;
	for slot in 0..ACTIVE_COUNT {
		let creature = order
			.get(slot)
			.and_then(|rid| battle_state.roster.get_mon(RosterId(*rid)));
		// Slot 0 is my active attacking theirs; slot 1 is the reverse.
		let (attacker, defender) = if slot == 0 { (mine, theirs) } else { (theirs, mine) };
		y.extend(encode_moveset(creature, registry, battle_state, hooks, attacker, defender));
	}

	y.extend(encode_field(battle_state, replacement));

	assert!(
		y.len() == TOTAL_ENCODING_LEN,
		"encoding length {} != TOTAL_ENCODING_LEN {}",
		y.len(),
		TOTAL_ENCODING_LEN
	);
	y
}

fn encode_mon(op_mon: Option<&CreatureState>, registry: &Registry) -> Vec<f32> {
	let creature = match op_mon {
		Some(creature) => creature,
		// An empty roster slot encodes as all zeroes, same as before.
		None => return vec![0.0; MON_ENCODING_LEN],
	};

	let mut v: Vec<f32> = Vec::with_capacity(MON_ENCODING_LEN);

	// Raw stats. These come from `get_stat`, which is the creature's own value —
	// combat modifiers like a burn's Attack drop live in the hook layer and are
	// deliberately not visible here.
	for stat in Stat::ALL {
		v.push(creature.get_stat(stat, registry) as f32 / STAT_SCALAR);
	}

	// HP as a fraction, so a 1000 HP wall and an 85 HP sweeper are comparable.
	v.push(if creature.max_hp == 0 {
		0.0
	} else {
		creature.current_hp as f32 / creature.max_hp as f32
	});

	for stat in Stat::ALL {
		v.push(creature.stat_changes.get(stat) as f32 / STAGE_SCALAR);
	}

	// Status one-hot. All zeroes means healthy, so NoStatus needs no slot.
	for status in NonVolatileStatus::AFFLICTIONS {
		v.push((creature.non_vol_status == status) as u32 as f32);
	}

	// Ability one-hot. All zeroes means no ability, which is a real state.
	for ability in AbilityId::ALL {
		v.push((creature.ability == Some(ability)) as u32 as f32);
	}

	// Type multi-hot: one or two slots set. Needed for switching decisions —
	// "which of my bench resists what is in front of me" is unanswerable without
	// it.
	let typing = registry.get_species_data(creature.species_id).typing;
	for t in Type::ALL {
		v.push(typing.contains(t) as u32 as f32);
	}

	// Volatile multi-hot. Several can be active at once, so this is not a one-hot.
	for kind in VolatileKind::ALL {
		v.push(creature.volatiles.has(kind) as u32 as f32);
	}

	// The multi-hot above only says a condition is present. These three say how
	// far along it is, which for a ramping effect is the whole decision.
	//
	// Without the toxic counter the agent cannot tell 1/16 poison from 8/16, so
	// it cannot learn when a switch is worth the tempo. Without the Protect
	// streak it cannot tell a Protect that will work from one that almost
	// certainly will not.
	let substitute_hp = creature.volatiles.value(VolatileKind::Substitute) as f32;
	v.push(if creature.max_hp == 0 { 0.0 } else { substitute_hp / creature.max_hp as f32 });
	v.push(creature.volatiles.value(VolatileKind::ToxicCounter) as f32 / MAX_TOXIC_COUNTER as f32);
	v.push((creature.volatiles.value(VolatileKind::ProtectStreak) as f32 / 4.0).min(1.0));

	debug_assert_eq!(v.len(), MON_ENCODING_LEN);
	v
}

/// The four move slots of one active creature, scored against `opposing`.
fn encode_moveset(
	op_mon: Option<&CreatureState>,
	registry: &Registry,
	battle_state: &BattleState,
	hooks: &HookTable,
	attacker: Option<PositionId>,
	defender: Option<PositionId>,
) -> Vec<f32> {
	let attacker_typing = op_mon.map(|c| registry.get_species_data(c.species_id).typing);
	let mut v: Vec<f32> = Vec::with_capacity(MOVESLOT_COUNT * MOVE_ENCODING_LEN);
	for slot in 0..MOVESLOT_COUNT {
		let mv = op_mon
			.and_then(|creature| creature.moves.get(slot))
			.map(|move_id| registry.get_move(*move_id));
		v.extend(encode_move(mv, attacker_typing, battle_state, registry, hooks, attacker, defender));
	}
	v
}

/// What a move would ACTUALLY land for: chart, abilities and STAB.
///
/// This used to report the raw chart and leave abilities to be inferred from
/// the ability one-hot. That was a mistake, and a measurable one. Against a
/// Levitate holder a Ground move reads 2x on the chart and does exactly nothing
/// in practice, so the agent was being handed a number that was not merely
/// incomplete but *inverted* — the feature pointed hardest at the one move that
/// could not work.
///
/// Measured with a greedy policy on `start_battle.json`: reading the raw chart
/// won 5% of games as Team Zero; asking the hook table instead won 100%. Same
/// policy, same rosters, only this call changed.
///
/// It also made the defect asymmetric, because only Team One fields a Levitate
/// creature — so only Team Zero was ever lied to.
fn move_multiplier(
	mv: &PMove,
	attacker_typing: Option<Typing>,
	battle_state: &BattleState,
	registry: &Registry,
	hooks: &HookTable,
	attacker: Option<PositionId>,
	defender: Option<PositionId>,
) -> f32 {
	if !mv.move_type.is_damaging() {
		return 0.0;
	}
	let (attacker_pos, defender_pos) = match (attacker, defender) {
		(Some(a), Some(d)) => (a, d),
		_ => return 0.0,
	};
	let defender_typing = match battle_state.get_mon(defender_pos) {
		Some(c) => registry.get_species_data(c.species_id).typing,
		None => return 0.0,
	};

	let base = typing::effectiveness(mv.element, &defender_typing);
	let mut eff = hooks.final_effectiveness(battle_state, registry, attacker_pos, defender_pos, mv.move_id, base);
	if attacker_typing.map_or(false, |a| a.contains(mv.element)) {
		eff = eff.with_stab();
	}
	// 0, 0.25, 0.5, 1, 2, 4 (x1.5 with STAB) -> scaled into roughly 0..1.
	eff.as_f32() / 4.0
}

fn encode_move(
	op_move: Option<&PMove>,
	attacker_typing: Option<Typing>,
	battle_state: &BattleState,
	registry: &Registry,
	hooks: &HookTable,
	attacker: Option<PositionId>,
	defender: Option<PositionId>,
) -> Vec<f32> {
	let mv = match op_move {
		Some(mv) => mv,
		// Empty slot. Also what the mask disallows, so the agent gets a
		// consistent "nothing here" signal from both sides.
		None => return vec![0.0; MOVE_ENCODING_LEN],
	};
	// Strongest rider on the move, so "this one can burn you" is visible.
	let best_chance = mv
		.effects
		.iter()
		.map(|effect| effect.chance())
		.max()
		.unwrap_or(0);

	vec![
		mv.base_power as f32 / POWER_SCALAR,
		(mv.move_type == MoveType::Special) as u32 as f32,
		(mv.move_type == MoveType::Status) as u32 as f32,
		mv.base_prio as f32 / PRIORITY_SCALAR,
		best_chance as f32 / 100.0,
		move_multiplier(mv, attacker_typing, battle_state, registry, hooks, attacker, defender),
	]
}

fn encode_field(battle_state: &BattleState, replacement: bool) -> Vec<f32> {
	let mut v: Vec<f32> = Vec::with_capacity(FIELD_ENCODING_LEN);

	let current = battle_state.weather.map(|timed| timed.weather);
	for weather in [Weather::Sandstorm, Weather::HarshSun, Weather::Rain, Weather::Snow] {
		v.push((current == Some(weather)) as u32 as f32);
	}
	v.push(
		battle_state
			.weather
			.map(|timed| timed.turns_left as f32 / WEATHER_TURNS_SCALAR)
			.unwrap_or(0.0),
	);
	v.push(battle_state.trick_room as u32 as f32);
	v.push(replacement as u32 as f32);

	debug_assert_eq!(v.len(), FIELD_ENCODING_LEN);
	v
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::creature_state::CreatureState;
	use crate::battle::state::field::PositionId;
	use crate::battle::state::roster::RosterId;
	use crate::battle::state::weather::{TimedWeather, Weather};
	use crate::model::speciesdata::SpeciesId;

	fn mon(registry: &Registry, id: u32) -> CreatureState {
		CreatureState::from_species(registry, SpeciesId(id), Registry::default_moveset(SpeciesId(id)))
	}

	/// cinderfox + stonewarden vs gustling + thornbeast.
	fn asymmetric_battle(registry: &Registry) -> BattleState {
		BattleState::from(
			vec![mon(registry, 2), mon(registry, 3)],
			vec![mon(registry, 6), mon(registry, 5)],
			vec![0, 1],
		)
	}

	/// The length assert in `assemble` is the thing that would bite silently, so
	/// pin it from both perspectives.
	#[test]
	fn encoding_is_the_declared_length_from_both_sides() {
		let registry = Registry::load();
		let battle_state = asymmetric_battle(&registry);
		for team in [Team::Zero, Team::One] {
			assert_eq!(encode(&battle_state, &registry, false, &team).len(), TOTAL_ENCODING_LEN);
			assert_eq!(encode(&battle_state, &registry, true, &team).len(), TOTAL_ENCODING_LEN);
		}
	}

	/// Every roster slot appears exactly once, from either side. If this ever
	/// broke, some creature would be silently invisible to the agent.
	#[test]
	fn perspective_order_is_a_permutation() {
		let registry = Registry::load();
		let battle_state = asymmetric_battle(&registry);
		for team in [Team::Zero, Team::One] {
			let mut order = perspective_order(&battle_state, &team);
			assert_eq!(order.len(), MON_COUNT);
			order.sort();
			assert_eq!(order, (0..MON_COUNT).collect::<Vec<_>>());
		}
	}

	/// The whole point: slot 0 is always the viewer's own active creature.
	#[test]
	fn each_side_sees_itself_first() {
		let registry = Registry::load();
		let battle_state = asymmetric_battle(&registry);

		let zero = perspective_order(&battle_state, &Team::Zero);
		let one = perspective_order(&battle_state, &Team::One);

		// team zero's active lives at roster 0, team one's at roster 1
		assert_eq!(zero[0], 0);
		assert_eq!(zero[1], 1);
		assert_eq!(one[0], 1);
		assert_eq!(one[1], 0);

		// and the encoded blocks agree: my-active block from one side equals the
		// their-active block from the other.
		let zero_view = encode(&battle_state, &registry, false, &Team::Zero);
		let one_view = encode(&battle_state, &registry, false, &Team::One);
		assert_eq!(
			zero_view[0..MON_ENCODING_LEN],
			one_view[MON_ENCODING_LEN..2 * MON_ENCODING_LEN]
		);
	}

	/// Two genuinely different sides must produce genuinely different vectors —
	/// this is the bug the perspective work fixes.
	#[test]
	fn the_two_perspectives_actually_differ() {
		let registry = Registry::load();
		let battle_state = asymmetric_battle(&registry);
		assert_ne!(
			encode(&battle_state, &registry, false, &Team::Zero),
			encode(&battle_state, &registry, false, &Team::One)
		);
	}

	/// A mirrored board must look identical to both players. If it doesn't, some
	/// side-specific information is leaking into the encoding.
	#[test]
	fn a_mirrored_board_looks_the_same_to_both_sides() {
		let registry = Registry::load();
		let mirrored = BattleState::from(
			vec![mon(&registry, 2), mon(&registry, 3)],
			vec![mon(&registry, 2), mon(&registry, 3)],
			vec![0, 1],
		);
		assert_eq!(
			encode(&mirrored, &registry, false, &Team::Zero),
			encode(&mirrored, &registry, false, &Team::One)
		);
	}

	/// `encode_both` must be exactly the two `encode` calls it replaces.
	#[test]
	fn encode_both_matches_encoding_separately() {
		let registry = Registry::load();
		let mut battle_state = asymmetric_battle(&registry);
		// perturb it so the two sides are definitely not symmetric
		battle_state.get_mut_mon(PositionId(0)).unwrap().current_hp = 37;
		battle_state.get_mut_mon(PositionId(1)).unwrap().non_vol_status = NonVolatileStatus::Burn;
		battle_state.weather = Some(TimedWeather { weather: Weather::Sandstorm, turns_left: 3 });

		let (zero, one) = encode_both(&battle_state, &registry, false);
		assert_eq!(zero, encode(&battle_state, &registry, false, &Team::Zero));
		assert_eq!(one, encode(&battle_state, &registry, false, &Team::One));
	}

	/// Switching a creature in must change what its own side sees at slot 0.
	#[test]
	fn perspective_follows_the_active_creature() {
		let registry = Registry::load();
		let mut battle_state = asymmetric_battle(&registry);

		let before = perspective_order(&battle_state, &Team::Zero);
		battle_state.field[PositionId(0)] = RosterId(2); // bring the bench mon out
		let after = perspective_order(&battle_state, &Team::Zero);

		assert_eq!(before[0], 0);
		assert_eq!(after[0], 2);
		// the creature that left is still present, now on the bench
		assert!(after[2..].contains(&0));
	}

	/// Two creatures that differ only by ability must encode differently, or the
	/// agent has no way to learn ability-dependent play.
	#[test]
	fn abilities_are_visible_to_the_agent() {
		let registry = Registry::load();
		let with_guts = CreatureState::from_species(&registry, SpeciesId(2), vec![]);
		let no_ability = CreatureState::from_species(&registry, SpeciesId(7), vec![]);

		let a = encode_mon(Some(&with_guts), &registry);
		let b = encode_mon(Some(&no_ability), &registry);
		assert_ne!(a, b);

		let ability_start = STAT_COUNT + 1 + STAT_COUNT + STATUS_COUNT;
		assert_ne!(a[ability_start..], b[ability_start..]);
	}

	/// Weather is part of the observation now.
	#[test]
	fn weather_is_visible_to_the_agent() {
		let registry = Registry::load();
		let clear = asymmetric_battle(&registry);
		let mut sandy = asymmetric_battle(&registry);
		sandy.weather = Some(TimedWeather { weather: Weather::Sandstorm, turns_left: 5 });

		assert_ne!(
			encode(&clear, &registry, false, &Team::Zero),
			encode(&sandy, &registry, false, &Team::Zero)
		);
	}

	/// Statuses beyond poison are distinguishable from each other.
	#[test]
	fn every_status_encodes_distinctly() {
		let registry = Registry::load();
		let mut seen: Vec<Vec<f32>> = Vec::new();
		for status in [
			NonVolatileStatus::NoStatus,
			NonVolatileStatus::Poison,
			NonVolatileStatus::BadPoison,
			NonVolatileStatus::Burn,
			NonVolatileStatus::Paralysis,
		] {
			let mut creature = CreatureState::from_species(&registry, SpeciesId(2), vec![]);
			creature.non_vol_status = status;
			let encoded = encode_mon(Some(&creature), &registry);
			assert!(!seen.contains(&encoded), "{:?} collided with another status", status);
			seen.push(encoded);
		}
	}

	/// The agent can tell its own move slots apart.
	#[test]
	fn active_movesets_are_encoded() {
		let registry = Registry::load();
		let creature = CreatureState::from_species(
			&registry,
			SpeciesId(4),
			Registry::default_moveset(SpeciesId(4)),
		);
		let battle_state = asymmetric_battle(&registry);
		let mut hooks = HookTable::new();
		hooks.refresh(&battle_state, &registry);
		let encoded = encode_moveset(
			Some(&creature), &registry, &battle_state, &hooks,
			Some(PositionId(0)), Some(PositionId(1)),
		);
		assert_eq!(encoded.len(), MOVESLOT_COUNT * MOVE_ENCODING_LEN);
		// mireling's slots are aqua pulse / frost bolt / static jolt / toxic mist —
		// a status move in slot 3 means that slot's is-status flag is set.
		assert_eq!(encoded[3 * MOVE_ENCODING_LEN + 2], 1.0);
		assert_eq!(encoded[0 * MOVE_ENCODING_LEN + 2], 0.0);
	}

	/// The move-effectiveness feature must report what a move would ACTUALLY do,
	/// abilities included.
	///
	/// This is the regression test for a measured defect: reporting the raw chart
	/// told the agent a Ground move was 2x into a Levitate holder when it does
	/// nothing at all, and a greedy policy reading that number won 5% of games
	/// instead of 100%.
	#[test]
	fn move_effectiveness_accounts_for_abilities() {
		let registry = Registry::load();
		// gustling (Electric, Levitate) is immune to Ground despite the chart
		// saying Ground hits Electric for 2x.
		let attacker = mon(&registry, 3); // stonewarden, has earth spike (Ground)
		let levitator = mon(&registry, 6);
		let grounded = mon(&registry, 7); // brackenox, Steel/Ground, no ability

		let against = |defender: CreatureState| {
			let battle_state = BattleState::from(vec![attacker.clone()], vec![defender], vec![0, 1]);
			let mut hooks = HookTable::new();
			hooks.refresh(&battle_state, &registry);
			let encoded = encode_moveset(
				battle_state.get_mon(PositionId(0)), &registry, &battle_state, &hooks,
				Some(PositionId(0)), Some(PositionId(1)),
			);
			// stonewarden's slot 1 is earth spike; the multiplier is the last
			// value in each move's block.
			encoded[1 * MOVE_ENCODING_LEN + (MOVE_ENCODING_LEN - 1)]
		};

		assert_eq!(
			against(levitator), 0.0,
			"a Ground move into Levitate must read as doing nothing"
		);
		assert!(
			against(grounded) > 0.0,
			"and the same move into something without the ability must still read as landing"
		);
	}

	/// An empty roster slot must not blow up or produce a ragged row.
	#[test]
	fn missing_creatures_encode_as_zeroes() {
		let registry = Registry::load();
		assert_eq!(encode_mon(None, &registry), vec![0.0; MON_ENCODING_LEN]);
	}
}
