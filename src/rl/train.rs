use crate::{battle::{command::Command, state::{BattleState, Field, PokemonState, PositionId}}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::{Agent, Moveslot}, encoder, env}};
use rand::Rng;
use crate::battle::state::Team;

/* We want to:
	* Model a strategy as a net
	* Test the strategy against a random strategy
	* See how well our strategy does against the random one
	* Backpropogate our loss
	(We need to figure out what loss means)
*/

pub fn main_loop() {
	let mut rng = rand::rng();
	let registry = Registry::load();
	let agent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let opponent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);

	let mut battle: BattleState = BattleState {
		mons: Field::from(
			vec![
				PokemonState::from_species(&registry, SpeciesId(1), vec![MoveId(0), MoveId(1)]),
			],
		vec![
			PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]),
		])
	};

	let mut done = false;

	let mut actions_and_states: Vec<(Vec<f32>, Moveslot)> = Vec::new();
	let mut battle_reward: f32 = -100.0;

	while !done {
		let encoded = encoder::encode(&battle, &registry);
		let agent_moveslot = agent.choose_move(&encoded, &registry);
		let opponent_moveslot = opponent.choose_move(&encoded, &registry);
		let actions = vec![
			agent_moveslot.to_command(Team::Zero, PositionId(0), &battle),
			opponent_moveslot.to_command(Team::One, PositionId(1), &battle)
		];
		actions_and_states.push((encoded, agent_moveslot));
		(battle, battle_reward, done) = env::step(battle, actions, &registry);
	}
	println!("battle: {:?}, reward: {}", battle, battle_reward);

	// now we have to back propagate on the agent

	// we pop from actions_and_states each time to access the next most recent action and state it came from

	// we call a function from Agent which will update the weights?
	/*
	match actions_and_states.pop() {
		Some((battle_state, move_decision)) => {
			// don't forget to use 'reward' in here somewhere
			agent.backprop(move_decision, battle_state, battle_reward);
		}
		None => {}
	} */
}