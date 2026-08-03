use rand::{Rng, seq::IteratorRandom};

use crate::{battle::state::{BattleState, PositionId, TEAM_SIZE, Team}, rl::agent::{MAX_DECISION, MOVESLOT_COUNT, Moveslot}};

pub struct Mask {
	allowed: [bool; MAX_DECISION]
}

impl Mask {
	pub fn from_battle_state(team: Team, pos: PositionId, battle_state: &BattleState) -> Self {
		let mut moveslots = [false; MOVESLOT_COUNT];
		let current_mon = battle_state.get_mon(pos).unwrap();
		
		if current_mon.current_hp != 0 {
			for i in 0..current_mon.moves.len() {
				moveslots[i] = true;
			}
		}

		

		let mut switches = [true; TEAM_SIZE];

		for (index, mon) in battle_state.roster.team(team).iter().enumerate() {
			switches[index] = match mon {
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
				pre_softmax[index] = -1e9;
			}
		}
	}

	/**
	 * Returns a random valid action from our mask.
	 * Yields None if there are no valid actions
	 */
	pub fn get_random_valid(&self, rng: &mut impl Rng) -> Option<Moveslot> {
		//println!("{:?}", self.allowed);
		self.allowed.iter().enumerate()
			.filter(|(_, b)| **b)
			.map(|(index, _)| Moveslot::from_number(index))
			.choose(rng)
	}

}