//! 1v1 positions to solve.
//!
//! One creature per side needs no special-casing anywhere: `Mask` already marks
//! every switch illegal when there is nothing alive on the bench to switch to,
//! so a 1v1 is simply a `BattleState` with short teams.
//!
//! Why start here at all, rather than at a full 6v6:
//!
//! * The right answer can be known independently — see [`known_answer_duel`].
//!   At 6v6 it cannot be, ever, and CFR fails silently, converging smoothly to
//!   the equilibrium of a game that is not the one being solved.
//! * The positions here **terminate on their own**, so the solver needs no depth
//!   limit and no leaf value function, and a wrong answer has exactly one
//!   possible cause rather than two.
//!
//! That second property is a property of the *position*, not of 1v1, and it is
//! worth being precise about because the obvious reasoning is wrong. There is no
//! healing move in the game, so it is tempting to conclude HP falls monotonically
//! and every duel ends. It does not follow. What actually matters is whether a
//! move can be played for *free*, because two creatures alternating free moves
//! cycle through states they have already visited, forever. The solver expands
//! every action at every turn, so such a position presents an unbounded tree.
//!
//! Two moves in the current registry are free in that sense:
//!
//! * *guard* (Protect) always costs nothing.
//! * *decoy* (Substitute) costs a quarter of a health bar — but only the first
//!   time. `Effect::Substitute` returns no events at all when one is already up
//!   or when paying would be fatal, so from then on it is a free no-op too.
//!
//! So no position here is entirely stall-proof, and the solver keeps a node
//! budget and a depth cap for that reason. Avoiding *guard* is enough to make a
//! duel terminate down almost every line; the handful that still stall are cut
//! off by the depth cap and counted, rather than being allowed to hang the solve.

use crate::battle::state::battle_state::BattleState;
use crate::battle::state::creature_state::CreatureState;
use crate::model::pmove::MoveId;
use crate::model::registry::Registry;
use crate::model::speciesdata::SpeciesId;

/// Field layout for a 1v1: roster slot 0 (team zero) against slot 1 (team one).
const ONE_V_ONE_FIELD: [usize; 2] = [0, 1];

fn creature(registry: &Registry, species: u32, moves: &[u32]) -> CreatureState {
	CreatureState::from_species(
		registry,
		SpeciesId(species),
		moves.iter().copied().map(MoveId).collect(),
	)
}

fn duel(team_zero: CreatureState, team_one: CreatureState) -> BattleState {
	BattleState::from(vec![team_zero], vec![team_one], ONE_V_ONE_FIELD.to_vec())
}

/// A position whose answer is known before the solver runs.
///
/// cinderfox (Fire) holds *flame lash* and *mud wave*. Against thornbeast
/// (Grass), flame lash is Fire on Grass with STAB — 3x — for about 102 damage,
/// while mud wave is Ground on Grass, resisted and off the wrong attacking stat,
/// for about 16. Same priority, same targeting, no drawback that could
/// compensate: flame lash strictly dominates, so the equilibrium is pure and
/// obvious.
///
/// That makes this the one battle position where the solver can be *checked*
/// rather than merely observed, which is why it carries the end-to-end test.
///
/// thornbeast is given real options (earth spike is Ground on Fire, and 2HKOs
/// back) so the position is a genuine fight rather than a walkover — cinderfox
/// is faster and wins the race, but only by playing correctly.
pub fn known_answer_duel(registry: &Registry) -> BattleState {
	duel(
		// flame lash, mud wave
		creature(registry, 2, &[3, 7]),
		// thorn whip, earth spike
		creature(registry, 5, &[14, 6]),
	)
}

/// The action index of the dominant move in [`known_answer_duel`], for tests.
pub const KNOWN_ANSWER_DOMINANT_ACTION: usize = 0;

/// A mirror match, for checking the solve is even-handed.
///
/// Both sides are the same creature with the same moves and the same HP, so the
/// position is balanced by construction and its value has to be exactly zero.
/// That is a strong check: anything giving one seat a systematic edge moves the
/// value off zero, and unlike a real matchup there is no argument about what the
/// answer should be.
///
/// The moves are *flame lash* and *decoy*. Substitute was picked over Protect
/// deliberately — see the note at the top of this module — because it is paid for
/// in HP and so cannot stall the battle forever.
///
/// Note that this position's equilibrium turns out to be close to *pure*: flame
/// lash does about 32 against a Substitute that costs 30 and holds 30, so the
/// decoy breaks to a single hit and is not worth the tempo. It is a symmetry
/// test, not a demonstration that the solver can find mixed equilibria — that is
/// covered at the matrix level in [`crate::cfr::matrix`], where rock-paper-scissors
/// has a known mixed answer. A battle position with a genuinely mixed
/// equilibrium has yet to be constructed.
///
/// A small fraction of lines here still stall, because decoy becomes a free
/// no-op once a Substitute is up — see the note at the top of this module. The
/// depth cap catches them and they are reported rather than hidden.
pub fn mirror_duel(registry: &Registry) -> BattleState {
	// flame lash, decoy
	duel(
		creature(registry, 2, &[3, 24]),
		creature(registry, 2, &[3, 24]),
	)
}

/// Field layout for a 2v2: both teams lead with their first creature.
const TWO_V_TWO_FIELD: [usize; 2] = [0, 1];

fn battle(team_zero: Vec<CreatureState>, team_one: Vec<CreatureState>) -> BattleState {
	BattleState::from(team_zero, team_one, TWO_V_TWO_FIELD.to_vec())
}

/// A 2v2 built around the switch-prediction game, with a genuinely mixed answer.
///
/// The three 2v2s already in `example_battles` do not serve. Solving them gives
/// `coverage_test` +0.92 and `sand_vs_levitate` -1.00, both of which one side
/// simply wins — and on a lost side every action is worth the same, so the
/// "equilibrium" there is arbitrary rather than informative. `status_duel` is
/// close at -0.12, but its Team One plays a single move 95% of the time. A
/// position worth studying has to be *balanced* and have an answer that is
/// actually a mixture.
///
/// Both are arranged here rather than hoped for.
///
/// **Balance** comes from making it a mirror. The two teams are identical, so the
/// value is zero by construction and neither seat can be favoured by an accident
/// of the roster.
///
/// **Mixing** comes from building the matchup as matching pennies. Each side
/// leads stonewarden (Rock/Ground) with thornbeast (Grass) in reserve, and the
/// lead holds exactly two moves, each answering one of the opponent's options:
///
/// | | opponent stays (stonewarden) | opponent switches (thornbeast) |
/// |---|---|---|
/// | *aqua pulse* | ~84 — Water is 4x on Rock/Ground | ~12 — resisted |
/// | *cinder blast* | ~12 — resisted | ~56 — Fire is 2x on Grass |
///
/// Each move is strong against exactly what the other is weak against, and
/// switches resolve before attacks, so the punish lands on whoever comes in.
///
/// **What solving it actually gives.** The position is balanced — the mirror
/// guarantees that, and the solver agrees, returning a value within a few
/// hundredths of zero. The answer is mixed rather than pure, but not in the
/// three-way shape the table above suggests: play splits between *aqua pulse* and
/// the switch, and *cinder blast* is left nearly unused. Its accumulated regret
/// is strongly negative while the other two sit close to zero, which is what
/// genuine indifference between two options looks like.
///
/// The reason is worth recording, because it is the thing that makes the position
/// interesting rather than a defect. Switching does not simply win: thornbeast
/// resists aqua pulse, takes about 12 coming in, and threatens a clean one-shot
/// back on stonewarden — but the opponent can see the switch and switch as well,
/// so the one-shot only lands if they stay. The prediction game recurses instead
/// of resolving, and the two options end up close to equal in value. Punishing
/// the switch with cinder blast is worth less than simply taking the 84 on a
/// stay, so it stays out of the mixture.
///
/// **Read the percentages with care.** Because the two live actions are so close
/// in value, the exact split moves between runs and between lookahead depths.
/// What is stable is the balance, which action is discarded, and the fact that
/// the answer is a mixture at all. Sharpening the split needs a better leaf
/// estimate than [`crate::cfr::leaf::HealthHeuristic`], which cannot see a type
/// matchup and so systematically undervalues a switch made for position.
///
/// Two details make the position behave at all. stonewarden is **bulky** (150 HP
/// behind 125 Defence), so the battle lasts long enough for a switch to pay off —
/// an earlier attempt with a frail cinderfox lead collapsed to a pure strategy,
/// because a two-turn race leaves no time to switch. And stonewarden's Sand
/// Stream puts a sandstorm up on entry, chipping the Grass switch-in every turn,
/// so neither side can simply sit there.
///
/// Two moves each also keeps the action space at three, which matters more than
/// it looks: search cost is exponential in the number of actions, so this solves
/// several turns deeper than a four-move position would.
pub fn switch_prediction_2v2(registry: &Registry) -> BattleState {
	// stonewarden: aqua pulse answers the Rock/Ground lead, cinder blast answers
	// the Grass switch-in.
	let lead = || creature(registry, 3, &[12, 4]);
	// thornbeast: STAB Grass, which is 4x back onto stonewarden, plus coverage.
	let reserve = || creature(registry, 5, &[14, 6]);

	battle(vec![lead(), reserve()], vec![lead(), reserve()])
}

/// A full six-a-side mirror, for finding out what happens at scale.
///
/// Both teams hold every creature in the roster with its designed moveset, so the
/// position is balanced by construction and its value has to be zero — the same
/// trick [`mirror_duel`] uses, and the only way to have a known right answer for
/// a position this size.
///
/// Each creature has four moves and five team-mates to switch to, so a node has
/// nine children against the 2v2's three. Search cost is exponential in that, and
/// so is the cost of measuring exploitability, which enumerates both sides'
/// actions rather than sampling one: 81 children per node against 9. A 6v6 is
/// therefore measurable only at a short horizon, and a short horizon is exactly
/// where the leaf estimate is carrying the answer.
pub fn full_team_mirror(registry: &Registry) -> BattleState {
	let team = || -> Vec<CreatureState> {
		(2..8u32)
			.map(|species| {
				CreatureState::from_species(
					registry,
					SpeciesId(species),
					Registry::default_moveset(SpeciesId(species)),
				)
			})
			.collect()
	};
	battle(team(), team())
}

/// A six-a-side that is *not* a mirror, for the team-preview question.
///
/// [`full_team_mirror`] answers "is the solve even-handed"; it cannot answer
/// "which creature should I lead", because in a mirror the lead matrix is
/// antisymmetric and both seats get the same answer by construction. This
/// position exists to make the preview decision a real one, and the two
/// constraints on it pull against each other:
///
/// * The teams have to differ, and there are only six real species, so no pair
///   of six-a-side teams can be built from disjoint species. The difference has
///   to come from *which* species are doubled and from what they are carrying.
/// * The position has to stay roughly level. On a lost side every action is
///   worth the same, so an unbalanced position has an arbitrary equilibrium and
///   teaches the preview nothing.
///
/// **What the balance actually turns on.** Three arrangements were measured at
/// `leads ... 2 200` before this one, and they say the constraint is narrower
/// than it looks:
///
/// * The elemental trio (cinderfox, mireling, gustling) doubled against the
///   ground trio (stonewarden, thornbeast, brackenox) — the only *fully*
///   disjoint split the roster allows — gives +0.049 to the elements. The
///   roster's cycle does not separate into two halves of three.
/// * Under [`crate::cfr::leaf::HealthHeuristic`] at a short horizon, gustling's
///   row beats every other lead in the mirror matrix and its column loses to
///   none. It is a dominant lead: Levitate makes the Ground moves that should
///   punish it do nothing, and *aqua pulse* covers the two Ground types its STAB
///   cannot touch. Give one side a gustling with that moveset and not the other
///   and the preview is decided before the battle starts.
/// * Give *both* sides one and the position is level (+0.003 measured) but the
///   answer is "both lead gustling" — balanced and useless, which is the failure
///   mode this function is trying to avoid rather than a success.
///
/// So the design here is: **exactly one gustling, and it is not allowed universal
/// coverage.** Team one's gustling carries *frost bolt* where the designed set
/// has *aqua pulse*. Electric does nothing to either Ground type, and the swap
/// takes Water's 4x on stonewarden and 2x on brackenox down to Ice's 2x and
/// neutral — enough that both Grounds now trade with it rather than lose to it.
/// Ice is kept rather than dropping the coverage entirely because an
/// Electric-only gustling loses to the Grass lead instead (+0.026 to thornbeast,
/// measured), which only moves the dominant lead somewhere else. That single
/// restriction is what turns the ladder back into a cycle, and the cycle is what
/// makes the preview answer a mixture.
///
/// **The two teams.** Team zero is the slow ground side: two stonewardens in
/// different roles (the designed set with *guard*, and a *blade dance* sweeper
/// set), brackenox, thornbeast, mireling and cinderfox on their designed sets.
/// Team one is the fast offensive side: the restricted gustling, two mirelings
/// (the designed defensive set, and an offensive one), two cinderfoxes (physical
/// Guts, and a special *cinder blast* set that goes at thornbeast's weaker
/// Special Defence), and thornbeast.
///
/// **Measured.** -0.012 to team zero at lookahead 2, holding at -0.014 with three
/// times the iterations and -0.012 at lookahead 3, against a spread of about
/// ±0.07 across the matrix — level enough that neither seat is playing a lost
/// position. Both sides mix: team zero splits thornbeast with brackenox, team one
/// splits gustling with the offensive mireling. Neither cinderfox is ever led,
/// which is the honest result and not a defect: they are bench answers, and the
/// preview only prices the first creature out.
pub fn six_asymmetric(registry: &Registry) -> BattleState {
	battle(
		vec![
			// stonewarden, designed set: sand up, stall behind guard.
			creature(registry, 3, &[5, 6, 16, 25]),
			// stonewarden again, as a sweeper rather than a wall: rock smash for
			// flinch pressure and blade dance instead of the stall.
			creature(registry, 3, &[5, 6, 20, 26]),
			creature(registry, 7, &[16, 6, 5, 26]),
			creature(registry, 5, &[14, 6, 20, 22]),
			creature(registry, 4, &[12, 15, 9, 22]),
			creature(registry, 2, &[3, 5, 13, 24]),
		],
		vec![
			// gustling with frost bolt where the designed set has aqua pulse:
			// still the answer to a Grass lead, no longer an answer to Ground.
			creature(registry, 6, &[10, 15, 19, 24]),
			creature(registry, 4, &[12, 15, 9, 22]),
			// mireling turned offensive: the status and the seed traded for
			// venom fang and dizzy ray.
			creature(registry, 4, &[12, 15, 8, 19]),
			creature(registry, 2, &[3, 5, 13, 24]),
			// cinderfox off its Special Attack instead of its Attack: lower
			// stat, but thornbeast's Special Defence is the weaker of its two.
			creature(registry, 2, &[4, 7, 19, 21]),
			creature(registry, 5, &[14, 6, 20, 22]),
		],
	)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::battle::state::field::PositionId;
	use crate::battle::state::Team;
	use crate::rl::mask::Mask;

	/// The premise of the whole 1v1 approach: with an empty bench there is
	/// nothing to switch to, so the action space collapses to the moveset without
	/// a line of code special-casing it.
	#[test]
	fn a_duel_offers_moves_and_no_switches() {
		let registry = Registry::load();
		let state = known_answer_duel(&registry);

		for (team, position) in [(Team::Zero, PositionId(0)), (Team::One, PositionId(1))] {
			let mask = Mask::from_battle_state(&team, position, &state);
			let legal: Vec<usize> = mask
				.allowed
				.iter()
				.enumerate()
				.filter(|(_, allowed)| **allowed)
				.map(|(index, _)| index)
				.collect();

			assert_eq!(legal, vec![0, 1], "{team:?} should have exactly its two moves");
		}
	}

	#[test]
	fn nobody_starts_a_duel_already_beaten() {
		let registry = Registry::load();
		assert!(known_answer_duel(&registry).outcome().is_none());
		assert!(mirror_duel(&registry).outcome().is_none());
	}

	/// Six a side, so nine legal actions rather than three — the number that makes
	/// a 6v6 expensive to search and much more expensive to measure.
	#[test]
	fn a_full_team_has_every_move_and_every_switch_available() {
		let registry = Registry::load();
		let state = full_team_mirror(&registry);

		let mask = Mask::from_battle_state(&Team::Zero, PositionId(0), &state);
		let legal = mask.allowed.iter().filter(|allowed| **allowed).count();
		assert_eq!(legal, 9, "four moves and five team-mates to switch to");
	}

	#[test]
	fn a_full_team_mirror_is_a_genuine_mirror() {
		let registry = Registry::load();
		let state = full_team_mirror(&registry);

		for index in 0..6 {
			let zero = state.get_mon_from_team(&Team::Zero, index).unwrap();
			let one = state.get_mon_from_team(&Team::One, index).unwrap();
			assert_eq!(zero.species_id, one.species_id);
			assert_eq!(zero.moves, one.moves);
			assert_eq!(zero.current_hp, one.current_hp);
		}
		assert!(state.outcome().is_none());
	}

	/// The point of [`six_asymmetric`] is that it is *not* [`full_team_mirror`],
	/// and the difference is easy to lose: the roster only has six species, so
	/// two teams can look different in the source and still line up creature for
	/// creature once they are laid out in team order. Assert the difference is
	/// real rather than cosmetic, and that both sides are still alive to play —
	/// a position one side has already lost prices every action the same and so
	/// says nothing about which creature to lead.
	#[test]
	fn the_asymmetric_six_is_not_a_mirror() {
		let registry = Registry::load();
		let state = six_asymmetric(&registry);

		let team = |team: Team| -> Vec<(u32, Vec<u32>)> {
			(0..6)
				.map(|index| {
					let mon = state
						.get_mon_from_team(&team, index)
						.expect("six a side means six creatures");
					assert!(mon.current_hp > 0, "{team:?} slot {index} starts fainted");
					(mon.species_id.0, mon.moves.iter().map(|id| id.0).collect())
				})
				.collect()
		};
		let zero = team(Team::Zero);
		let one = team(Team::One);

		assert_ne!(zero, one, "the teams are the same, so this is a mirror");

		// Order is not the difference: a reordered team is the same team, and
		// pairing the leads differently is all `deep leads` does anyway.
		let sorted = |mut members: Vec<(u32, Vec<u32>)>| {
			members.sort();
			members
		};
		assert_ne!(
			sorted(zero),
			sorted(one),
			"the teams differ only in the order they are written down",
		);

		// Neither side is beaten before a move is played.
		assert!(state.outcome().is_none());
	}

	#[test]
	fn the_mirror_duel_is_a_genuine_mirror() {
		let registry = Registry::load();
		let state = mirror_duel(&registry);

		let zero = state.get_mon(PositionId(0)).unwrap();
		let one = state.get_mon(PositionId(1)).unwrap();

		assert_eq!(zero.species_id, one.species_id);
		assert_eq!(zero.moves, one.moves);
		assert_eq!(zero.current_hp, one.current_hp);
	}
}

#[cfg(test)]
mod emit {
	use super::*;
	/// One-off: writes the position to `example_battles` so the other binaries
	/// can load it. Ignored by default; run with `--ignored` to regenerate.
	#[test]
	#[ignore]
	fn write_switch_prediction_json() {
		let registry = Registry::load();
		switch_prediction_2v2(&registry)
			.to_file("example_battles/switch_prediction_2v2.json")
			.unwrap();
	}

	/// The two six-a-side team-preview positions, written out together because
	/// they are read as a pair: the mirror says what a balanced lead matrix looks
	/// like, and the other one is only interesting next to it.
	#[test]
	#[ignore]
	fn write_six_a_side_json() {
		let registry = Registry::load();
		full_team_mirror(&registry)
			.to_file("example_battles/six_mirror.json")
			.unwrap();
		six_asymmetric(&registry)
			.to_file("example_battles/six_asymmetric.json")
			.unwrap();
	}
}
