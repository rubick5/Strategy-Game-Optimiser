use crate::{battle::{command::Command, state::BattleState}, model::registry::Registry, rl::encoder};

pub enum Moveslot {}

impl Moveslot {
	pub fn to_command(&self) -> Command {
		todo!()
	}
}

pub struct Agent {
	// a bunch of weights telling us what to do
	
}

impl Agent {
	pub fn choose_move(&self, battle: BattleState, registry: &Registry) -> Moveslot {
		let representation = encoder::encode(battle, registry);
		todo!()
	}
	pub fn init_random() -> Self {
		Agent {}
	}

	pub fn backprop(&self, move_slot: Moveslot, state: BattleState) {
		todo!()
	}
}