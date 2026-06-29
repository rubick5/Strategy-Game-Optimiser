use crate::{battle::{command::Command, state::BattleState}, model::registry::Registry, rl::{agent::{Agent, Moveslot}, env}};


/* We want to:
	* Model a strategy as a net
	* Test the strategy against a random strategy
	* See how well our strategy does against the random one
	* Backpropogate our loss
	(We need to figure out what loss means)
 */

fn main_loop() {
	let registry = Registry::load();
	let agent: Agent = Agent::init_random();
	let opponent: Agent = Agent::init_random();

	let mut battle: BattleState = todo!();
	
	let mut done = false;

	let mut actions_and_states: Vec<(BattleState, Moveslot)> = Vec::new();

	while !done {
		let agent_moveslot = agent.choose_move(battle, &registry);
		let opponent_moveslot = opponent.choose_move(battle, &registry);
		let actions = vec![agent_moveslot.to_command(), opponent_moveslot.to_command()];
		actions_and_states.push((battle, agent_moveslot));
		let (battle, reward, done) = env::step(battle, actions, &registry);
	}

	// now we have to back propagate on the agent

	// we pop from actions_and_states each time to access the next most recent action and state it came from

	// we call a function from Agent which will update the weights?
	match actions_and_states.pop() {
		Some((battle_state, move_decision)) => {
			agent.backprop(move_decision, battle_state);
		}
		None => {}
	}
}