pub mod train_config;
pub mod bot_agent;
pub mod random_agent;
pub mod spam_agent;
pub mod ppo_agent;

use crate::rl::{mask::Mask, moveslot::Moveslot};
use rand::RngCore;


pub trait Agent {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot;
}

impl <A: Agent + ?Sized> Agent for Box<A> {
	fn choose_move(&mut self, representation: &[f32], mask: &Mask, rng: &mut dyn RngCore) -> Moveslot {
		(**self).choose_move(representation, mask, rng)
	}
}