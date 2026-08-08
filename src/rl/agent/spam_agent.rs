use crate::rl::{agent::Agent, mask::Mask, moveslot::Moveslot};
use rand::RngCore;

pub struct SpamAgent {
	pub index: usize
}

impl Agent for SpamAgent {
	fn choose_move(&mut self, _representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		if mask.allowed[self.index] {
			Moveslot::from_number(self.index)
		} else {
			mask.get_random_valid(rng).expect("no actions available and tried to sample random (spamagent)")
		}
	}
}