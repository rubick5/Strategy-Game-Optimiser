use crate::battle::state::{battle_state::BattleState, field::PositionId, non_volatile_status::NonVolatileStatus, roster::RosterId};


pub enum Event {
	DealDamage {
		amount: u32,
		target: PositionId,
	},
	Switch {
		current: PositionId,
		new: RosterId,
	},
	ApplyNonVolStatus {
		status: NonVolatileStatus,
		target: PositionId,
	}
}

impl Event {
	pub fn deal_percent_damage(battle_state: &BattleState, percentage: f32, target: PositionId) -> Result<Self, Box<dyn std::error::Error>> {
		let mon = battle_state.get_mon(target).ok_or("position doesn't exist in the field")?;
		let amount = mon.current_hp as f32 * percentage / 100.0;

		Ok(Event::DealDamage { amount: amount.round() as u32, target })
	}

}