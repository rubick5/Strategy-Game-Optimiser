use eframe::egui;

use crate::{battle::state::BattleState, rl::agent::BotAgent};

pub struct BattleApp {
	pub battle: BattleState,
	pub agent: BotAgent,
	pub log: Vec<String>,
}

impl eframe::App for BattleApp {
	fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
		ui.heading("pokemon battle yay!!");
	}
}

