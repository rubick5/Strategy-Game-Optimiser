use std::error::Error;

use poke_sim::{battle::state::{BattleState, PokemonState}, game_window::game_window::BattleApp, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::BotAgent, neural_net::SavedNeuralNet}};

pub fn start_battle_state(registry: &Registry) -> BattleState {
	let ps0 = PokemonState::from_species(registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let ps01 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);
	let ps11 = PokemonState::from_species(registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]);

	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps1_1hp = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	ps1_1hp.current_hp = 1;
	BattleState::from(vec![ps0, ps01], vec![ps1, ps11], vec![0, 1])
}

pub fn main() -> Result<(), Box<dyn Error>> {
	let registry = Registry::load();
	let args: Vec<String> = std::env::args().collect();


	match args.get(1).map(|s| s.as_str()) {
		Some(file_name) => {
			let battle_app = BattleApp {
				battle: start_battle_state(&registry),
				agent: BotAgent::relu_from_file(file_name)?,
				log: vec![],
				};
			let _eframe_result = eframe::run_native(
				"Poke battle",
				eframe::NativeOptions::default(),
				Box::new(|_cc| Ok(Box::new(battle_app))),
			)?;
		}
		None => println!("No file input received...")
	}
	Ok(())
	
}