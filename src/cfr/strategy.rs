use std::collections::HashMap;

use crate::battle::command::Command;
use crate::battle::state::battle_state::BattleState;
use crate::battle::state::field::PositionId;
use crate::battle::state::{Outcome, Team};
use crate::model::registry::Registry;
use crate::rl::mask::Mask;
use crate::rl::moveslot::Moveslot;

type Infoset = Vec<f32>;
struct InfosetData {
	pub regrets: Vec<(Moveslot, f32)>,
	pub strategy_sum: Vec<(Moveslot, f32)>,
}

type RegretTable = HashMap<Infoset, InfosetData>;
struct TraversalData {
	pub battle: BattleState,
	pub commands_ready: Vec<Command>,
	pub team: Team,
	pub pos: PositionId,
}

fn traversal_thing(pi1: f32, pi2: f32, t: TraversalData, original_tbl: RegretTable, write_tbl: &mut RegretTable, registry: &Registry) -> f32 {
	match t.battle.outcome() {
		Some(Outcome::Win { team}) => {
			match team {
				Team::Zero => 1.0,
				Team::One => -1.0,
			}
		},

		Some(Outcome::Draw) => 0.0,
		None => {
			// need to calculate the pi1s pi2s pics
			// there's a lot of pics during the game but idk if we honestly get into
			// allat
			// need to do regret matching
			// need to continue branching
			let mask = Mask::from_battle_state(&t.team, t.pos, &t.battle);
			let node_values = mask.get_all_valid().iter().map(
				|moveslot| {
					let command = moveslot.to_command(t.pos, &t.battle, registry);
				}
			);
			6.7
		}
	}
}