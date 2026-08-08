use crate::rl::{agent::Agent, mask::Mask, moveslot::Moveslot};
use rand::RngCore;

pub struct RandomAgent {}

impl Agent for RandomAgent {
	fn choose_move(&mut self, _representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		mask.get_random_valid(rng).expect("no actions available and tried to sample random")
	}
}