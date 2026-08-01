use core::error;
use std::{f32::consts::E, thread::current};

/*

Next steps:
1. fix caching/recalculation of pre-activation inputs for net back propagation
2. 
 */

use rand::{Rng, distr::Uniform};

use crate::rl::agent::Moveslot;

const LEARNING_RATE: f32 = 0.01;

#[derive(Debug, Clone)]
pub struct NeuralNet<F>
where
	F: Fn(f32) -> f32
{
	layers: Vec<(NeuronLayer, Vec<f32>)>,
	hidden_activation: F,
	hidden_activation_prime: F,
	pre_activations: Vec<Vec<f32>>,
}

impl<F> NeuralNet<F>
where
	F: Fn(f32) -> f32,
{
	fn clear_pre_activations(&mut self) {
		self.pre_activations.clear();
	}
	pub fn gen_random(rng: &mut impl Rng, layer_sizes: &[usize], hidden_activation: F, hidden_activation_prime: F) -> Self {
		Self {
			layers: (0..layer_sizes.len()-1).map(|n| (NeuronLayer::gen_random(rng, layer_sizes[n], layer_sizes[n+1]), vec![])).collect(),
			hidden_activation,
			hidden_activation_prime,
			pre_activations: vec![],
		}
	}

	pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
		let mut output = inputs.to_owned();
		let ((output_layer, output_pres), hidden_layers) = self.layers
			.split_last_mut()
			.expect("network needs at least one layer");

		for (layer, pres) in hidden_layers.iter_mut() {
			output = layer.forward(&output);
			*pres = output.clone();
			output = output.iter()
				.map(|x| (self.hidden_activation)(*x))
				.collect();
		}

		output = output_layer.forward(&output);
		*output_pres = output.clone();
		output

	}

	pub fn backward(&mut self, gt: f32, battle_reward: f32, move_chosen: Moveslot, input_received: &[f32], last_probabilities: &[f32]) {
		self.forward(input_received); // sets the pre_activation cache for this decision made
		let move_chosen_index = move_chosen.to_number();
		let mut current_errors: Vec<f32> = (0..last_probabilities.len()).map ( |index| {
			let indicator = if index == move_chosen_index { 1.0 } else { 0.0 };
			battle_reward * gt * (last_probabilities[index] - indicator)
		}).collect();

		let ((output_layer, _), hidden_layers) =
			self.layers
			.split_last_mut()
			.expect("network needs at least one layer");

		current_errors = output_layer.backward(&current_errors);

		for (layer, pres) in hidden_layers.iter_mut().rev() {
			current_errors.iter_mut().enumerate().for_each(|(i, x)|
				*x = *x * (self.hidden_activation_prime)(pres[i])
			);
			current_errors = layer.backward(&current_errors);
		}

	}
}

fn activation_prime(x: f32) -> f32 {
	if x > 0.0 { 1.0 } else { 0.0 }
}

#[derive(Debug, Clone)]
pub struct NeuronLayer {
	most_recent_input: Vec<f32>,
	most_recent_output: Vec<f32>,
	pub neurons: Vec<Neuron>
}

impl NeuronLayer {
	pub fn gen_random(rng: &mut impl Rng, inputs: usize, outputs: usize) -> Self {
		Self {
			most_recent_input: vec![],
			most_recent_output: vec![],
			neurons: (0..outputs).map(|_| Neuron::gen_random(inputs, rng)).collect(),
		}
	}
	pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
		self.most_recent_input = input.to_owned();
		self.most_recent_output = self.neurons.iter().map(|neuron| neuron.forward(&input)).collect();
		self.most_recent_output.clone()
	}

	pub fn backward(&mut self, incoming_error: &[f32]) -> Vec<f32> {
		// we need to calculate the error to give to the next row here before we update the weights

		let mut d_prev: Vec<f32> = vec![0.0; self.neurons[0].weights.len()];
		for neuron_index in 0..self.neurons.len() {
			for index in 0..d_prev.len() {
				d_prev[index] += self.neurons[neuron_index].weights[index] * incoming_error[neuron_index];
			}
		}

		for (index, neuron) in self.neurons.iter_mut().enumerate() {
			neuron.backprop(incoming_error[index], &self.most_recent_input);
		}

		d_prev
	}

}

#[derive(Debug, Clone)]
pub struct Neuron {
	pub weights: Vec<f32>,
	pub bias: f32,
}

impl Neuron {
	pub fn forward(&self, inputs: &[f32]) -> f32 {
		if inputs.len() != self.weights.len() {
			panic!("bad neural net setup: inputs; {} weights: {}", inputs.len(), self.weights.len());
		}
		let mut result = self.bias;
		for i in 0..inputs.len() {
			result += inputs[i] * self.weights[i];
		}
		return result;
	}

	pub fn gen_random(weight_count: usize, rng: &mut impl Rng) -> Self {
		let distribution = Uniform::new(-1.0, 1.0).unwrap();
		let weights = rng.sample_iter(distribution).take(weight_count).collect();
		let bias = 0.0; //rng.random_range(-2.0..2.0);

		Neuron {
			weights,
			bias
		}
	}
	pub fn backprop(&mut self, error: f32, input_given: &[f32]) {
		for index in 0..self.weights.len() {
			self.weights[index] -= LEARNING_RATE * error * input_given[index];
		}
		self.bias -= LEARNING_RATE * error;
	}
}

