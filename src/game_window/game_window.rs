use eframe::egui;
use rand::RngCore;

use crate::{battle::{engine::engine, state::{Outcome, Team, battle_state::BattleState, creature_state::CreatureState, field::PositionId}}, model::registry::Registry, rl::{agent::Agent, encoder, mask::Mask, moveslot::Moveslot}};


pub struct BattleApp<R>
where
	R: RngCore,
{
	pub battle: BattleState,
	pub agent: Box<dyn Agent>,
	pub log: Vec<String>,
	pub registry: Registry,
	pub player_team: Team,
	pub rng: R,
}

impl <R> eframe::App for BattleApp<R>
where
	R: RngCore
{
	fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
		self.show_battle_state(ui);

		match self.battle.outcome() {
			Some(Outcome::Win { team }) if team == self.player_team => {
				ui.label("you win!!");
			},
			Some(Outcome::Win {team: _}) => {
				ui.label("you lose...");
			},
			Some(Outcome::Draw) => {
				ui.label("it's a draw?");
			}
			None => {
				self.display_move_options(ui);
			}
		}
		
	}
}

impl <R> BattleApp<R>
where
	R: RngCore
{

	fn display_move_options(&mut self, ui: &mut egui::Ui) {
		let player_position = self.battle.field.team_positions(&self.player_team)[0];
		let agent_position = self.battle.field.team_positions(&self.player_team.other())[0];

		let mut move_selected: Option<Moveslot> = None;

		egui::Grid::new("moves").show(ui, |ui| {
			move_selected = self.generate_moveslot_buttons(ui, player_position);
		});

		if let Some(moveslot) = move_selected {
			// do some stuff
			let encoding = encoder::encode(&self.battle, &self.registry, false);
			let agent_mask = Mask::from_battle_state(&self.player_team.other(), agent_position, &self.battle);
			let agent_moveslot = self.agent.choose_move(&encoding, &agent_mask, &mut self.rng);
			let commands = vec![
				agent_moveslot.to_command(agent_position, &self.battle, &self.registry),
				moveslot.to_command(player_position, &self.battle, &self.registry),
			];
			// TODO: HANDLE STEP RESULT FOR REPLACEMENTS SO WE DONT GET HIT ON SWITCH IN
			let step_result = engine::step(self.battle.clone(), commands, &self.registry, &mut self.rng);
			self.battle = step_result.battle_state;
		}
	}

	fn display_creature_state(&self, ps: &CreatureState) -> String {
		let species_data = self.registry.get_species_data(ps.species_id);

		format!("{}: HP {} out of {}, with status: {}", species_data.name, ps.current_hp, species_data.base_hp, ps.non_vol_status)
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