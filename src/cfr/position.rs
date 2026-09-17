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
}
