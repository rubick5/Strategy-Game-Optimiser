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
use crate::battle::state::roster::RosterId;
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

/// A 1v1 whose whole question is whether setting up is worth the turn.
///
/// This exists because of a gap found in [`crate::cfr::critic::default_curriculum`]:
/// not one of its three positions contains a stat-boosting move, so every
/// stat-stage input to the encoder is zero in every training sample the critic
/// ever sees. Those weights get no gradient, and a critic trained on that
/// curriculum cannot have learned what a boost is worth — it would fail a test of
/// setup pricing for want of ever having been shown one, which says nothing about
/// whether a network could learn it.
///
/// brackenox holds *iron press* and *blade dance*. Blade dance deals no damage
/// and costs the turn, returning a doubled Attack to spend over the turns that
/// follow: exactly the trade a short horizon misprices, and the reason the
/// six-a-side lead matrix ranks a blade dance stonewarden below its guard set in
/// every single column.
///
/// thornbeast answers with *thorn whip* and *earth spike*, Ground being 2x back
/// onto brackenox's Steel, so setting up in front of it is a real gamble rather
/// than a free turn — which is what makes the position worth labelling.
///
/// It terminates on its own, like the other duels here: blade dance restores no
/// health and cannot be played for free, since the opponent keeps attacking
/// through it. So this is an *exactly* solvable position, and its labels carry no
/// debt to any leaf estimate.
pub fn setup_duel(registry: &Registry) -> BattleState {
	duel(
		// iron press, blade dance
		creature(registry, 7, &[16, 26]),
		// thorn whip, earth spike
		creature(registry, 5, &[14, 6]),
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

/// [`six_asymmetric`] with team zero's two leads differing *only* in whether one
/// of them holds *guard*.
///
/// This exists to settle which of two creatures the six-a-side horizon sweep was
/// actually mispricing. That sweep compared a stonewarden carrying *iron press*
/// and *guard* against one carrying *rock smash* and *blade dance*, found the
/// second worse by 0.032 at two turns of lookahead and by only 0.005 at five,
/// and the obvious reading was that the search underprices setup.
///
/// The individual values say otherwise. Across depths two to five the blade
/// dance set moves by 0.008 while the *guard* set falls by 0.038 — nearly five
/// times as far. It is the guard set whose value is collapsing, not the dance
/// set's that is climbing. Which points at Protect rather than at blade dance,
/// and for a reason that fits: Protect blocks everything for a turn at no visible
/// cost, and a search that stops immediately afterwards never pays for it. It
/// does not see the free turn the opponent gets, nor `protect_success_chance`
/// decaying on repeated use.
///
/// The two-set comparison cannot separate those, because the sets differ in two
/// moves at once. Here they differ in exactly one. Both leads are the designed
/// stonewarden; the second has *guard* replaced by *mud wave*, which is plain —
/// no rider, no tempo effect of its own — and redundant with the Ground STAB
/// already in the set, so it is close to an empty slot. That is deliberate: the
/// question is not whether the replacement is any good, it is whether the gap
/// between them *shrinks as the search deepens*.
///
/// A gap that is wide at two turns and narrow at five means Protect was being
/// overvalued by the short horizon. A gap that holds steady means Protect is
/// simply worth that much and the horizon was never the problem.
pub fn guard_probe(registry: &Registry) -> BattleState {
	let mut state = six_asymmetric(registry);
	// Team zero's second slot becomes the first one with guard swapped out, so
	// the pair differs in that move and nothing else. Everything behind them,
	// and the whole of team one, is left exactly as six_asymmetric has it.
	//
	// Roster slots interleave, team zero at even indices, so its second creature
	// is slot two.
	*state
		.roster
		.get_mut_mon(RosterId(2))
		.expect("six_asymmetric gives team zero six creatures") =
		creature(registry, 3, &[5, 6, 16, 7]);
	state
}

/// The control for [`guard_probe`]: the same experiment with no *guard* in it.
///
/// Without this, [`guard_probe`] proves less than it appears to. It shows a gap
/// that is wide at two turns, vanishes at three and returns smaller at four —
/// but a sweep that oscillates like that for *any* pair of movesets would
/// produce the same picture whatever the moves were, and Protect would be
/// convicted on evidence that had nothing to do with it.
///
/// So here both leads hold a plain attack in the slot under test — *mud wave*
/// against *mind shatter*, neither carrying a rider, a boost or a block. If the
/// gap between two ordinary moves is flat across depths, the oscillation in
/// `guard_probe` belongs to Protect. If it oscillates the same way, it is an
/// artifact of the sweep and `guard_probe` says nothing.
pub fn plain_probe(registry: &Registry) -> BattleState {
	let mut state = six_asymmetric(registry);
	// Roster slots interleave, team zero at even indices.
	*state.roster.get_mut_mon(RosterId(0)).expect("team zero has a lead") =
		creature(registry, 3, &[5, 6, 16, 7]);
	*state.roster.get_mut_mon(RosterId(2)).expect("team zero has a second") =
		creature(registry, 3, &[5, 6, 16, 17]);
	state
}

/// A position where setting up is right, and does not pay off for many turns.
///
/// Built to test one thing: can the solver find a move whose entire value lies
/// beyond its horizon? *blade dance* deals no damage and costs the turn. A search
/// that stops before the boost has been spent sees only the cost, so it declines
/// it — which is exactly what the six-a-side lead matrix does, ranking a blade
/// dance stonewarden below its guard set in every single column.
///
/// The trap in building such a position is letting the payoff arrive *too soon*.
/// If setting up wins the very next turn then a two-turn search already sees it
/// and the position tests nothing. So the reward is deliberately placed a long
/// way after the investment, and the type chart is what places it:
///
/// | | iron press (Steel) | earth spike (Ground) |
/// |---|---|---|
/// | gustling | 0.5x | **0x**, via Levitate |
/// | stonewarden | 2x | 2x |
///
/// and back the other way, gustling's *static jolt* is **0x** into brackenox's
/// Ground typing while *dizzy ray* is a resisted 0.5x.
///
/// So the lead is a near-stalemate. brackenox cannot meaningfully hurt gustling —
/// its Ground STAB does literally nothing and its Steel STAB is resisted — and
/// gustling cannot meaningfully hurt it back. That is what makes spending a turn
/// on the boost nearly free, and so correct. But the *reward* is not in that
/// matchup at all: it is behind gustling, in a stonewarden that takes double from
/// both of brackenox's attacks and is far too bulky to break unboosted. The chain
/// is: spend a turn, grind through a creature that resists you, and only then
/// collect.
///
/// Both sides keep a second creature so neither starts in a lost position, and
/// both leads hold two moves rather than four to keep the action space small
/// enough to search several turns deeper than usual.
///
/// **Measured**, 2500 iterations at each horizon:
///
/// | lookahead | blade dance | value |
/// |---|---|---|
/// | 2 | **0%** | +0.868 |
/// | 3 | 99% | +0.904 |
/// | 4 | 93% | +0.938 |
/// | 5 | 100% | +0.955 |
/// | 6-8 | 91-100% | +0.94 to +0.95 |
///
/// Two separate thresholds, and the gap between them is the useful finding. The
/// **decision** is wrong at two turns and right from three: at the horizon this
/// position was built to defeat, the solver plays the move that cannot pay off
/// inside it and declines the one that can. But the **valuation** stays wrong
/// well past that, converging only around five turns — at two turns the position
/// is priced 0.087 below its settled value, and still 0.017 low at four.
///
/// That distinction is what reconciles this with the six-a-side lead matrix,
/// where a blade dance stonewarden is ranked below its guard set in every column.
/// A lead matrix is built out of *values*, not decisions, so it inherits the
/// slower of the two convergences. The horizon sweep in [`crate::cfr::horizon`]
/// agrees: that gap reads -0.032 at two turns and settles near -0.005.
pub fn delayed_setup(registry: &Registry) -> BattleState {
	battle(
		vec![
			// iron press, earth spike, blade dance
			creature(registry, 7, &[16, 6, 26]),
			// a real answer in reserve, so losing brackenox is not losing outright
			creature(registry, 4, &[12, 15]),
		],
		vec![
			// dizzy ray and static jolt: 0.5x and 0x into brackenox respectively
			creature(registry, 6, &[19, 10]),
			// the prize — doubly weak to both of brackenox's attacks, and bulky
			// enough that reaching it unboosted is not enough
			creature(registry, 3, &[5, 16]),
		],
	)
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
/// Its fourth move changes too, and the reason is unrelated: *cinder blast*
/// becomes *decoy*. A gustling holding both frost bolt and cinder blast still
/// answers everything, just one type-chart step sideways, so the coverage cut
/// above would not have bound. Note the cost — decoy is one of the two moves
/// that can be played for free once a Substitute is up, so this lead can stall,
/// and lines through it lean on the depth cap rather than ending.
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
/// times the iterations, -0.012 at lookahead 3 and -0.0119 at lookahead 4 —
/// against a spread of about ±0.07 across the matrix, so neither seat is playing
/// a lost position, and stable enough across four horizons to say the answer does
/// not depend on where the search stops.
///
/// The full lookahead-4 answer, 4000 iterations a matchup, 30 minutes: team zero
/// leads brackenox 49%, thornbeast 43%, cinderfox 8%; team one leads gustling
/// 59%, thornbeast 39%, the offensive mireling 2%. Both supports satisfy the
/// indifference condition to within 0.0002 while every excluded lead is clearly
/// worse, which is what an equilibrium is and is checkable by hand from the
/// printed matrix.
///
/// Neither stonewarden is ever led, and the two of them differ by a mean of only
/// 0.0028 across the whole matrix despite sharing just half a moveset — with the
/// *blade dance* set the worse of the two in every single column. That is the
/// shape a horizon effect makes: four turns is long enough to see the stat boost
/// and too short to see the sweep it buys, so setup is priced at its cost.
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
	// Used only by the tests; guard_probe reaches the roster through RosterId.
	use crate::battle::state::Team;
	use crate::model::pmove::MoveId;
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
		assert!(setup_duel(&registry).outcome().is_none());
	}

	/// The point of [`setup_duel`] is that a boost can actually appear in it. If
	/// the boosting move is ever dropped from the set this silently stops being a
	/// training position about setup and becomes another plain duel.
	#[test]
	fn the_setup_duel_contains_a_boosting_move() {
		let registry = Registry::load();
		let state = setup_duel(&registry);
		let mon = state.get_mon(PositionId(0)).unwrap();

		const BLADE_DANCE: u32 = 26;
		assert!(
			mon.moves.iter().any(|id| id.0 == BLADE_DANCE),
			"no boosting move, so this teaches nothing about setup: {:?}",
			mon.moves,
		);
		assert_eq!(mon.stat_changes, crate::battle::state::stat_stages::StatStages::new());
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

	/// The premise of [`delayed_setup`] is a lead matchup neither side can win
	/// quickly. If a moveset or the type chart ever changes so that one of them
	/// can, the position silently stops testing delayed payoff and starts testing
	/// nothing, while still looking fine. So the interactions it rests on are
	/// asserted rather than left in a comment.
	#[test]
	fn the_setup_position_really_does_stall_in_the_lead() {
		use crate::model::typing::effectiveness;
		let registry = Registry::load();
		let state = delayed_setup(&registry);

		let typing = |position: PositionId| {
			registry.get_species_data(state.get_mon(position).unwrap().species_id).typing.clone()
		};
		let element = |id: u32| registry.get_move(MoveId(id)).element;

		// brackenox's Steel STAB is resisted by the lead it faces.
		assert_eq!(effectiveness(element(16), &typing(PositionId(1))).as_f32(), 0.5);
		// Its Ground STAB is answered by an ability rather than a typing, so the
		// chart alone will not show the zero and the ability is asserted instead.
		assert_eq!(
			registry.get_species_data(state.get_mon(PositionId(1)).unwrap().species_id).ability,
			Some(crate::model::ability::AbilityId::Levitate),
			"without Levitate this is not a stalemate and earth spike simply wins",
		);
		// And the lead's Electric STAB does nothing back.
		assert_eq!(effectiveness(element(10), &typing(PositionId(0))).as_f32(), 0.0);

		// The reward waiting behind it takes double from both attacks.
		let prize = registry
			.get_species_data(state.get_mon_from_team(&Team::One, 1).unwrap().species_id)
			.typing
			.clone();
		assert_eq!(effectiveness(element(16), &prize).as_f32(), 2.0);
		assert_eq!(effectiveness(element(6), &prize).as_f32(), 2.0);

		assert!(state.outcome().is_none());
	}

	/// The whole value of [`guard_probe`] is that its two leads differ in exactly
	/// one move. If a second difference ever creeps in, the experiment silently
	/// goes back to being the confounded two-move comparison it was built to
	/// replace, while still producing numbers.
	#[test]
	fn the_guard_probe_leads_differ_in_exactly_one_move() {
		let registry = Registry::load();
		let state = guard_probe(&registry);

		let moves = |slot: usize| -> Vec<u32> {
			state
				.get_mon_from_team(&Team::Zero, slot)
				.unwrap()
				.moves
				.iter()
				.map(|id| id.0)
				.collect()
		};
		let (with_guard, without) = (moves(0), moves(1));

		assert_eq!(with_guard.len(), without.len());
		let differences =
			with_guard.iter().zip(&without).filter(|(a, b)| a != b).count();
		assert_eq!(differences, 1, "{with_guard:?} against {without:?}");

		const GUARD: u32 = 25;
		assert!(with_guard.contains(&GUARD), "the guard set lost its guard");
		assert!(!without.contains(&GUARD), "the control still holds guard");

		// Same species, so nothing but that move is in play.
		assert_eq!(
			state.get_mon_from_team(&Team::Zero, 0).unwrap().species_id,
			state.get_mon_from_team(&Team::Zero, 1).unwrap().species_id,
		);
		assert!(state.outcome().is_none());
	}

	/// The control only controls if neither of its leads can stall. A rider or a
	/// block creeping into either slot would quietly turn it into a second copy
	/// of the experiment it is supposed to be checking.
	#[test]
	fn the_plain_probe_compares_two_moves_with_no_tricks() {
		let registry = Registry::load();
		let state = plain_probe(&registry);

		// guard, decoy, blade dance, and every move carrying a status rider.
		const NOT_PLAIN: [u32; 12] = [2, 3, 4, 8, 9, 10, 11, 19, 20, 21, 22, 23];
		const STALLING: [u32; 3] = [22, 24, 25];

		let moves = |slot: usize| -> Vec<u32> {
			state
				.get_mon_from_team(&Team::Zero, slot)
				.unwrap()
				.moves
				.iter()
				.map(|id| id.0)
				.collect()
		};
		let (first, second) = (moves(0), moves(1));

		let differences = first.iter().zip(&second).filter(|(a, b)| a != b).count();
		assert_eq!(differences, 1, "{first:?} against {second:?}");

		for (slot, set) in [(0, &first), (1, &second)] {
			for id in set.iter() {
				assert!(!STALLING.contains(id), "slot {slot} can stall on move {id}");
			}
		}
		// The move actually under test, in each set, must be a plain attack.
		let changed: Vec<(u32, u32)> =
			first.iter().zip(&second).filter(|(a, b)| a != b).map(|(a, b)| (*a, *b)).collect();
		let (a, b) = changed[0];
		assert!(!NOT_PLAIN.contains(&a) && !NOT_PLAIN.contains(&b), "{a} against {b} is not plain");

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

	#[test]
	#[ignore]
	fn write_delayed_setup_json() {
		let registry = Registry::load();
		delayed_setup(&registry)
			.to_file("example_battles/delayed_setup.json")
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
