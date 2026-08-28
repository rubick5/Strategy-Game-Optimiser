use rand::{RngCore, seq::IteratorRandom};

use crate::{battle::state::{TEAM_SIZE, Team, battle_state::BattleState, field::PositionId, roster::RosterId}, rl::moveslot::{MAX_DECISION, MOVESLOT_COUNT, Moveslot}};

#[derive(Clone, Copy)]
pub struct Mask {
	pub allowed: [bool; MAX_DECISION]
}

impl Mask {
	pub fn from_battle_state(team: &Team, pos: PositionId, battle_state: &BattleState) -> Self {
		let mut moveslots = [false; MOVESLOT_COUNT];
		let current_mon = battle_state.get_mon(pos).unwrap();
		
		if current_mon.current_hp != 0 {
			// `.min(MOVESLOT_COUNT)` because `moveslots` is a fixed [bool; 4]:
			// giving a creature a fifth move used to panic here rather than
			// failing anywhere near the registry entry that caused it.
			for i in 0..current_mon.moves.len().min(MOVESLOT_COUNT) {
				moveslots[i] = true;
			}
		}

		let mut switches = [true; TEAM_SIZE];

		// Roster layout interleaves the teams, so team index i is roster i*2+offset.
		let team_offset = match team { Team::Zero => 0, Team::One => 1 };
		let active_roster_id = battle_state.field[pos];

		for (index, creature) in battle_state.roster.team(team).iter().enumerate() {
			// Switching to the creature that is already out was legal, and used to
			// be a harmless no-op. It is not harmless any more: with switch hooks
			// in play it became a free action that fires SwitchOut and SwitchIn —
			// a Natural Cure holder could cure its own status every turn by
			// "switching" to itself, which is exactly the kind of degenerate line
			// a policy-gradient agent will find and exploit.
			if RosterId(index * 2 + team_offset) == active_roster_id {
				switches[index] = false;
				continue;
			}
			switches[index] = match creature {
				Some(poke_state) => poke_state.current_hp != 0,
				None => false
			}
		}
		let allowed = std::array::from_fn(|i| if i < moveslots.len() { moveslots[i] } else { switches[i - moveslots.len()] });

		Self {
			allowed,
		}
	}

	pub fn apply(&self, pre_softmax: &mut [f32]) {
		//println!("pre softmax len: {}", pre_softmax.len());
		assert!(pre_softmax.len() == MAX_DECISION);
		for (index, allowed) in self.allowed.iter().enumerate() {
			if !allowed {
				pre_softmax[index] = f32::NEG_INFINITY;
			}
		}
	}

	/**
	 * Returns a random valid action from our mask.
	 * Yields None if there are no valid actions
	 */
	pub fn get_random_valid(&self, rng: &mut dyn RngCore) -> Option<Moveslot> {
		//println!("{:?}", self.allowed);
		self.allowed.iter().enumerate()
			.filter(|(_, b)| **b)
			.map(|(index, _)| Moveslot::from_number(index))
			.choose(rng)
	}

}