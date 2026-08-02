use crate::{battle::{command::Command, state::{BattleState, Field, PokemonState, PositionId}}, model::{pmove::MoveId, registry::Registry, speciesdata::SpeciesId}, rl::{agent::{Agent, Moveslot}, encoder, env, mask::Mask}};
use rand::Rng;
use crate::battle::state::Team;

const DAMPING_CONSTANT: f32 = 0.95;

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
	let mut agent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	let mut battles_won = 0;


	let ps0 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps00 = ps0.clone();
	ps00.current_hp = 1;
	let ps1 = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	let mut ps1_1hp = PokemonState::from_species(&registry, SpeciesId(0), vec![MoveId(0), MoveId(1)]);
	ps1_1hp.current_hp = 1;
	let battle_state_normal = BattleState::from(vec![ps0], vec![ps1], vec![0, 1]);
	let battle_state_1hp = BattleState::from(vec![ps00], vec![ps1_1hp], vec![0, 1]);


	let mut opponent: Agent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
	for i in 0..60_000 {
		//if i % 1000 == 0 {
			//opponent = agent.clone();
		//}
		opponent = Agent::init_random(encoder::TOTAL_ENCODING_LEN, &mut rng);
		let mut battle: BattleState = if rand::random() {
			battle_state_normal.clone()
		} else {
			battle_state_1hp.clone()
		};
		let mut done = false;

		let mut actions_and_states: Vec<(Vec<f32>, Moveslot, Vec<f32>)> = Vec::new();
		let mut battle_reward: f32 = -100.0;

		while !done {
			let agent_mask = Mask::from_battle_state(Team::Zero, PositionId(0), &battle);
			let opponent_mask = Mask::from_battle_state(Team::One, PositionId(1), &battle);

			let encoded = encoder::encode(&battle, &registry);

			let (agent_moveslot, probabilities) = agent.choose_move(&encoded, agent_mask);
			let (opponent_moveslot, _) = opponent.choose_move(&encoded, opponent_mask);

			let actions = vec![
				agent_moveslot.to_command(Team::Zero, PositionId(0), &battle),
				opponent_moveslot.to_command(Team::One, PositionId(1), &battle)
			];
			actions_and_states.push((encoded, agent_moveslot, probabilities));
			(battle, battle_reward, done) = env::step(battle, actions, &registry);
		}
		if battle_reward == 1.0 {
			battles_won += 1;
		}
		if i % 500 == 0 {
			println!("iter {}: battles won: {}", i, battles_won);
			battles_won = 0;
		}
		//println!("battle: {:?}, reward: {}", battle, battle_reward);

		// now we have to back propagate on the agent

		// we pop from actions_and_states each time to access the next most recent action and state it came from

		// we call a function from Agent which will update the weights?
		let mut gt = 1.0;
		loop {
			match actions_and_states.pop() {
				Some((encoding, move_decision, probabilities)) => {
					// don't forget to use 'reward' in here somewhere
					agent.backprop(move_decision, battle_reward, gt, &encoding, &probabilities);
				}
				None => break,
			}
			gt = gt * DAMPING_CONSTANT;
		}

	}
	//println!("neuron one for our agent: {:?}", agent.neuron1.weights);
	//println!("neuron two for our agent: {:?}", agent.neuron2.weights);

	let agent_mask_normal = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_normal);
	let agent_mask_1hp = Mask::from_battle_state(Team::Zero, PositionId(0), &battle_state_1hp);
	
	println!("{:?}", agent.choose_move(&encoder::encode(&battle_state_normal, &registry), agent_mask_normal));
	println!("{:?}", agent.choose_move(&encoder::encode(&battle_state_1hp, &registry), agent_mask_1hp));
}