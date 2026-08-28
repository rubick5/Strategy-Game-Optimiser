use eframe::egui;
use rand::RngCore;

use crate::{battle::{command::Command, engine::engine::{self, StepRequest}, state::{Outcome, Team, battle_state::BattleState, creature_state::CreatureState, field::PositionId, weather::TimedWeather}}, model::{pmove::{MoveType, PMove}, registry::Registry, speciesdata::Stat, typing::{self, Typing}}, rl::{agent::Agent, encoder, mask::Mask, moveslot::{MOVESLOT_COUNT, Moveslot}}};

/// Width of the text HP bar, in characters.
const HP_BAR_WIDTH: usize = 20;

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
	pub agent_used: String,
	/// What the engine is waiting for.
	///
	/// This used to be thrown away, which is what let creatures get attacked on
	/// the way in: after a faint the engine returns `NeedsReplacements`, but the
	/// window went straight back to the normal move menu, so the replacement
	/// switch was submitted as an ordinary turn action alongside a freely chosen
	/// attack from the opponent.
	///
	/// Start a new battle at `NeedsActions`.
	pub step_request: StepRequest,
}

impl <R> eframe::App for BattleApp<R>
where
	R: RngCore
{
	fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
		self.show_field(ui);
		ui.separator();

		// Driven by what the engine actually asked for, rather than by
		// re-deriving the situation from the battle state.
		match self.step_request.clone() {
			StepRequest::Finished(Outcome::Win { team }) if team == self.player_team => {
				ui.heading("you win!!");
			}
			StepRequest::Finished(Outcome::Win { team: _ }) => {
				ui.heading("you lose...");
			}
			StepRequest::Finished(Outcome::Draw) => {
				ui.heading("it's a draw?");
			}
			StepRequest::NeedsReplacements(positions) => {
				self.display_replacement_options(ui, &positions);
			}
			StepRequest::NeedsActions => {
				self.display_move_options(ui);
			}
		}
	}
}

impl <R> BattleApp<R>
where
	R: RngCore
{

	pub fn new(battle: BattleState,
		agent: Box<dyn Agent>,
		registry: Registry,
		player_team: Team,
		rng: R,
	) -> Self {
		Self {
			battle,
			agent,
			log: vec![],
			registry,
			rng,
			player_team,
			agent_used: String::from(""),
			step_request: StepRequest::NeedsActions,
		}
	}
	// -----------------------------------------------------------------------
	// Driving the battle
	// -----------------------------------------------------------------------

	/// Run one engine step and remember what it asks for next.
	fn submit(&mut self, commands: Vec<Command>) {
		let step_result = engine::step(self.battle.clone(), commands, &self.registry, &mut self.rng);
		self.battle = step_result.battle_state;
		self.step_request = step_result.step_request;
	}

	/// Ask the agent to pick a replacement for one of its own positions.
	fn agent_replacement(&mut self, pos: PositionId) -> Command {
		let agent_team = self.player_team.other();
		// `replacement: true` matches how the encoder is called during training,
		// so the agent sees the same kind of position it was trained on.
		let encoding = encoder::encode(&self.battle, &self.registry, true, &agent_team);
		let mask = Mask::from_battle_state(&agent_team, pos, &self.battle);
		let chosen = self.agent.choose_move(&encoding, &mask, &mut self.rng);
		chosen.to_command(pos, &self.battle, &self.registry)
	}

	/// The replacement phase: nobody attacks, both sides just send something in.
	fn display_replacement_options(&mut self, ui: &mut egui::Ui, positions: &[PositionId]) {
		let agent_team = self.player_team.other();
		let player_positions: Vec<PositionId> =
			positions.iter().filter(|p| p.team() == self.player_team).copied().collect();
		let agent_positions: Vec<PositionId> =
			positions.iter().filter(|p| p.team() == agent_team).copied().collect();

		// The agent's side needs no input from us, so decide it up front.
		let mut commands: Vec<Command> = Vec::new();
		for pos in &agent_positions {
			commands.push(self.agent_replacement(*pos));
		}

		if player_positions.is_empty() {
			// Only the opponent fainted. Resolve immediately rather than showing
			// the human an empty menu and stalling the battle.
			self.submit(commands);
			return;
		}

		ui.heading("your creature fainted - choose a replacement");

		// Singles, so there is at most one position to fill.
		let pos = player_positions[0];
		let mut chosen: Option<Moveslot> = None;
		egui::Grid::new("replacements").show(ui, |ui| {
			chosen = self.generate_switch_buttons(ui, pos);
		});

		if let Some(moveslot) = chosen {
			commands.push(moveslot.to_command(pos, &self.battle, &self.registry));
			self.submit(commands);
		}
	}

	fn display_move_options(&mut self, ui: &mut egui::Ui) {
		let player_position = self.battle.field.team_positions(&self.player_team)[0];
		let agent_position = self.battle.field.team_positions(&self.player_team.other())[0];

		let mut move_selected: Option<Moveslot> = None;

		ui.heading("your move");
		egui::Grid::new("moves").show(ui, |ui| {
			move_selected = self.generate_moveslot_buttons(ui, player_position);
		});

		if let Some(moveslot) = move_selected {
			// Encoded from the agent's own side, not the player's — the agent here
			// is whichever team the human is not.
			let agent_team = self.player_team.other();
			let encoding = encoder::encode(&self.battle, &self.registry, false, &agent_team);
			let agent_mask = Mask::from_battle_state(&agent_team, agent_position, &self.battle);
			let agent_moveslot = self.agent.choose_move(&encoding, &agent_mask, &mut self.rng);
			self.agent_used = match agent_moveslot {
				Moveslot::Slot(n) => self.battle.field.team_positions(&agent_team).iter().map(|p| { 
					self.registry.get_move(self.battle.get_mon(*p).unwrap().moves[n]).name.clone()
				}).collect::<Vec<String>>().concat(),
				Moveslot::Switch(n) => String::from(format!("switch to {}", n)),
			};
			let commands = vec![
				agent_moveslot.to_command(agent_position, &self.battle, &self.registry),
				moveslot.to_command(player_position, &self.battle, &self.registry),
			];
			self.submit(commands);
		}
	}

	// -----------------------------------------------------------------------
	// Showing the battle
	// -----------------------------------------------------------------------

	fn show_field(&self, ui: &mut egui::Ui) {
		ui.label(format!("agent last used: {}", self.agent_used));
		ui.label(self.describe_field());
		ui.separator();

		ui.heading("THEM");
		self.display_team(&self.player_team.other(), ui);
		self.display_bench(&self.player_team.other(), ui);

		ui.separator();

		ui.heading("YOU");
		self.display_team(&self.player_team, ui);
		self.display_bench(&self.player_team, ui);
	}

	/// Weather and any other field-wide state, on one line.
	fn describe_field(&self) -> String {
		let weather = match self.battle.weather {
			Some(TimedWeather { weather, turns_left }) => {
				format!("{:?} ({} turn{} left)", weather, turns_left, if turns_left == 1 { "" } else { "s" })
			}
			None => String::from("clear"),
		};
		let trick_room = if self.battle.trick_room { "  |  Trick Room" } else { "" };
		format!("weather: {}{}", weather, trick_room)
	}

	fn display_team(&self, team: &Team, ui: &mut egui::Ui) {
		for m in self.battle.field.team(team) {
			let creature = match self.battle.roster.get_mon(*m) {
				Some(creature) => creature,
				None => continue,
			};
			ui.strong(self.describe_headline(creature));
			ui.label(self.describe_hp(creature));
			ui.label(self.describe_stats(creature));
			ui.label(self.describe_conditions(creature));
		}
	}

	/// One line per benched creature, so you can see what you have left to
	/// switch to and roughly how healthy it is.
	fn display_bench(&self, team: &Team, ui: &mut egui::Ui) {
		let active: Vec<_> = self.battle.field.team(team).into_iter().copied().collect();
		let mut lines: Vec<String> = Vec::new();
		for (index, slot) in self.battle.roster.team(team).iter().enumerate() {
			let creature = match slot {
				Some(creature) => creature,
				None => continue,
			};
			let roster_id = crate::battle::state::roster::RosterId(index * 2 + team.roster_offset());
			if active.contains(&roster_id) {
				continue;
			}
			let species = self.registry.get_species_data(creature.species_id);
			let state = if creature.is_alive() {
				format!("{}/{}", creature.current_hp, creature.max_hp)
			} else {
				String::from("fainted")
			};
			lines.push(format!(
				"{} [{}] {}",
				species.name,
				describe_typing(&species.typing),
				state
			));
		}
		if !lines.is_empty() {
			ui.label(format!("  bench: {}", lines.join("   |   ")));
		}
	}

	/// Name, typing and ability.
	fn describe_headline(&self, creature: &CreatureState) -> String {
		let species = self.registry.get_species_data(creature.species_id);
		let ability = match creature.ability {
			Some(ability) => ability.name(),
			None => "no ability",
		};
		format!(
			"{}  [{}]  ability: {}",
			species.name,
			describe_typing(&species.typing),
			ability
		)
	}

	/// A text HP bar plus the raw numbers, because the bar alone hides whether
	/// something survives one more hit.
	fn describe_hp(&self, creature: &CreatureState) -> String {
		let filled = if creature.max_hp == 0 {
			0
		} else {
			(creature.current_hp as usize * HP_BAR_WIDTH) / creature.max_hp as usize
		};
		let bar: String = std::iter::repeat('#')
			.take(filled)
			.chain(std::iter::repeat('.').take(HP_BAR_WIDTH - filled))
			.collect();
		let percent = if creature.max_hp == 0 {
			0
		} else {
			creature.current_hp * 100 / creature.max_hp
		};
		format!("  HP [{}] {}/{} ({}%)", bar, creature.current_hp, creature.max_hp, percent)
	}

	/// Live stats, with any stat stage shown next to the value it produced.
	fn describe_stats(&self, creature: &CreatureState) -> String {
		let mut parts: Vec<String> = Vec::new();
		for (stat, label) in [
			(Stat::Attack, "Atk"),
			(Stat::Defense, "Def"),
			(Stat::SpecialAttack, "SpA"),
			(Stat::SpecialDefense, "SpD"),
			(Stat::Speed, "Spe"),
		] {
			let stage = creature.stat_changes.get(stat);
			let suffix = if stage == 0 {
				String::new()
			} else {
				format!(" ({:+})", stage)
			};
			parts.push(format!("{} {}{}", label, creature.get_stat(stat, &self.registry), suffix));
		}
		format!("  {}", parts.join("  "))
	}

	/// Both kinds of status on one line, since in play you care about the
	/// combination rather than which bucket each one lives in.
	fn describe_conditions(&self, creature: &CreatureState) -> String {
		let mut parts: Vec<String> = Vec::new();
		if creature.non_vol_status.is_afflicted() {
			parts.push(format!("{}", creature.non_vol_status));
		}
		for volatile in creature.volatiles.iter() {
			let mut label = volatile.kind.name().to_string();
			if let Some(turns) = volatile.turns_left {
				label.push_str(&format!(" ({}t)", turns));
			}
			// Substitute stores its remaining HP, which is the thing you actually
			// want to know about it.
			if volatile.kind == crate::battle::state::volatile::VolatileKind::Substitute {
				label.push_str(&format!(" [{} hp]", volatile.value));
			}
			parts.push(label);
		}
		if parts.is_empty() {
			String::from("  status: none")
		} else {
			format!("  status: {}", parts.join(", "))
		}
	}

	// -----------------------------------------------------------------------
	// Buttons
	// -----------------------------------------------------------------------

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

	/// Switch buttons only — a replacement is not a choice between attacking and
	/// switching, so the move slots are not offered at all.
	fn generate_switch_buttons(&self, ui: &mut egui::Ui, position: PositionId) -> Option<Moveslot> {
		let mut final_moveslot = None;
		let mask = Mask::from_battle_state(&self.player_team, position, &self.battle);
		for moveslot in Moveslot::all_moveslots() {
			let slot_number = moveslot.to_number();
			if slot_number < MOVESLOT_COUNT {
				continue;
			}
			if ui.add_enabled(
				mask.allowed[slot_number],
				egui::Button::new(self.generate_moveslot_name(moveslot, position))).clicked()
			{
				final_moveslot = Some(moveslot);
			}
		}
		final_moveslot
	}

	fn generate_moveslot_name(&self, moveslot: Moveslot, position: PositionId) -> String {
		match moveslot {
			Moveslot::Slot(n) => {
				let option_mv = self.battle.get_mon(position).unwrap().moves.get(n);
				match option_mv {
					Some(mv) => self.describe_move(self.registry.get_move(*mv), position),
					None => String::from("-"),
				}
			}
			Moveslot::Switch(n) => {
				match self.battle.get_mon_from_team(&self.player_team, n) {
					Some(target) => {
						let species = self.registry.get_species_data(target.species_id);
						format!(
							"{} [{}]\n{}/{}",
							species.name,
							describe_typing(&species.typing),
							target.current_hp,
							target.max_hp
						)
					}
					None => String::from("-"),
				}
			}
		}
	}

	/// Everything you need to pick a move: type, category, power, riders, and
	/// what the chart says it would do to whatever is currently opposite.
	fn describe_move(&self, mv: &PMove, user: PositionId) -> String {
		let category = match mv.move_type {
			MoveType::Physical => "phys",
			MoveType::Special => "spec",
			MoveType::Status => "status",
		};
		let power = if mv.base_power == 0 {
			String::from("--")
		} else {
			mv.base_power.to_string()
		};

		let mut line = format!("{}\n{} {} {}", mv.name, mv.element.name(), category, power);

		// Effectiveness against the creature opposite, worked out here so you
		// don't have to hold the chart in your head.
		if mv.move_type.is_damaging() {
			let foe_position = self.battle.field.team_positions(&user.team().other()).first().copied();
			if let Some(foe) = foe_position.and_then(|pos| self.battle.get_mon(pos)) {
				let foe_typing = self.registry.get_species_data(foe.species_id).typing;
				let effect = typing::effectiveness(mv.element, &foe_typing);
				let tag = if effect.is_immune() {
					"x0"
				} else if effect.as_f32() >= 3.9 {
					"x4"
				} else if effect.is_super_effective() {
					"x2"
				} else if effect.as_f32() <= 0.3 {
					"x1/4"
				} else if effect.is_resisted() {
					"x1/2"
				} else {
					""
				};
				if !tag.is_empty() {
					line.push_str(&format!("  {}", tag));
				}
			}
		}

		// Riders, so a 30% flinch is visible without reading the registry.
		let riders: Vec<String> = mv
			.effects
			.iter()
			.map(|effect| format!("{} {}%", effect.label(), effect.chance()))
			.collect();
		if !riders.is_empty() {
			line.push('\n');
			line.push_str(&riders.join(", "));
		}
		line
	}
}

/// "Rock/Ground" or just "Fire".
fn describe_typing(typing: &Typing) -> String {
	match typing.secondary {
		Some(second) => format!("{}/{}", typing.primary.name(), second.name()),
		None => typing.primary.name().to_string(),
	}
}
