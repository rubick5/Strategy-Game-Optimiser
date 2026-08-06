use eframe::egui;

use crate::{battle::state::{BattleState, CreatureState, PositionId, Team}, model::registry::Registry, rl::{agent::BotAgent, mask::Mask, moveslot::Moveslot}};


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

		self.show_battle_state(ui);
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

	fn display_creature_state(&self, ps: &CreatureState) -> String {
		let species_data = self.registry.get_species_data(ps.species_id);
		format!("{}: HP {} out of {}", species_data.name, ps.current_hp, species_data.base_hp)
	}
	fn display_team(&self, team: &Team, ui: &mut egui::Ui) {
		for m in self.battle.field.team(team) {
			let ps = self.battle.roster.get_mon(*m).unwrap();
			ui.strong(self.display_creature_state(&ps));
		}	
	}
	fn show_battle_state(&self, ui: &mut egui::Ui) {

		ui.heading(format!("your active(s):"));
		self.display_team(&self.player_team, ui);

		ui.heading(format!("their active(s):"));
		self.display_team(&self.player_team.other(), ui);
	}

	fn generate_moveslot_buttons(&self, ui: &mut egui::Ui, position: PositionId) -> Option<Moveslot> {
		let mut final_moveslot = None;
		let mask = Mask::from_battle_state(&self.player_team, position, &self.battle);
		for moveslot in Moveslot::all_moveslots() {
			if ui.add_enabled(
				mask.allowed[moveslot.to_number()], 
				egui::Button::new(self.generate_moveslot_name(moveslot, position))).clicked()
			{
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
					self.registry.get_species_data(target.species_id).name.clone()
				} else {
					String::from("nobody")
				}
			}
		}
	}

}