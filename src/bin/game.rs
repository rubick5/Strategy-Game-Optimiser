use std::error::Error;

use rand::{Rng as _, SeedableRng, rngs::StdRng};
use strat_optimizer::{battle::state::{Team, battle_state::BattleState, creature_state::CreatureState}, game_window::game_window::BattleApp, model::{registry::Registry, speciesdata::SpeciesId}, rl::agent::bot_agent::BotAgent};

/// A three-a-side match-up built from the new roster, so the demo actually shows
/// the abilities off: Sand Stream vs Levitate, with a Rough Skin wall and a Guts
/// attacker on the field to start.
///
/// Each creature gets `Registry::default_moveset`, which is four moves — the cap
/// `Mask` imposes.
pub fn start_battle_state(registry: &Registry) -> BattleState {
	let mon = |id: u32| {
		CreatureState::from_species(registry, SpeciesId(id), Registry::default_moveset(SpeciesId(id)))
	};

	// cinderfox (Guts), stonewarden (Sand Stream), mireling (Natural Cure)
	let team0 = vec![mon(2), mon(3), mon(4)];
	// thornbeast (Rough Skin), gustling (Levitate), brackenox (no ability)
	let team1 = vec![mon(5), mon(6), mon(7)];

	BattleState::from(team0, team1, vec![0, 1])
}

pub fn main() -> Result<(), Box<dyn Error>> {
	let registry = Registry::load();
	let args: Vec<String> = std::env::args().collect();
	let mut rng = rand::rng();
	let rng = StdRng::seed_from_u64(rng.random());

	start_battle_state(&registry).to_file("example_battles/start_battle.json").unwrap();

	// NOTE: an agent.json trained before the encoder change has the wrong input
	// width and will not work here. Retrain first (`cargo run --bin
	// strat-optimizer`), then point this at the fresh file.
	let file_name = args.get(1).map(|s| s.as_str()).unwrap_or("agent.json");
	let battle_app = BattleApp::<StdRng>::new(
		BattleState::from_file("example_battles/start_battle.json").unwrap(),
		Box::new(BotAgent::relu_from_file(file_name)?),
		registry,
		Team::One,
		rng,
	);
	let _eframe_result = eframe::run_native(
		"battle",
		eframe::NativeOptions::default(),
		Box::new(|_cc| Ok(Box::new(battle_app))),
	)?;
	Ok(())

}
