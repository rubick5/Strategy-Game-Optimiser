use eframe::egui;

use crate::{battle::state::{BattleState, PositionId, Team}, model::registry::Registry, rl::{agent::BotAgent, moveslot::Moveslot}};


pub struct BattleApp {
	pub battle: BattleState,
	pub agent: BotAgent,
	pub log: Vec<String>,
	pub registry: Registry,
	pub player_team: Team,
}

impl eframe::App for BattleApp {
	fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
		let position = match self.player_team {
			Team::One => PositionId(1),
			Team::Zero => PositionId(0),
		};

		ui.heading("pokemon battle yay!!");
		let mut move_selected: Option<Moveslot> = None;
		egui::Grid::new("moves").show(ui, |ui| {
			move_selected = self.generate_moveslot_buttons(ui, position);
		});

		if let Some(moveslot) = move_selected {
			// do some stuff
		}
	}
}

impl BattleApp {
	fn generate_moveslot_buttons(&self, ui: &mut egui::Ui, position: PositionId) -> Option<Moveslot> {
		let mut final_moveslot = None;
		for moveslot in Moveslot::all_moveslots() {
			if ui.button(self.generate_moveslot_name(moveslot, position)).clicked() {
					final_moveslot = Some(moveslot);
			}
			if let Moveslot::Slot(3) = moveslot {
				ui.end_row();
				ui.end_row();
			}
		}
		final_moveslot
	}

	fn generate_moveslot_name(&self, moveslot: Moveslot, position: PositionId) -> String {
		match moveslot {
			Moveslot::Slot(n) => {
				let option_mv = self.battle.get_mon(position).unwrap().moves.get(n);
				if let Some(mv) = option_mv {
					self.registry.get_move(*mv).name.clone()
				} else {
					String::from("empty")
				}
			}
			Moveslot::Switch(n) => {
				if let Some(target) = self.battle.get_mon_from_team(&self.player_team, n) {
					self.registry.get_pokemon(target.species_id).name.clone()
				} else {
					String::from("nobody")
				}
			}
		}
	}

}