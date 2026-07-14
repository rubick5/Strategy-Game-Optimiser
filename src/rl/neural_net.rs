use std::f32::consts::E;

use rand::{Rng, distr::Uniform};

use crate::rl::agent::Moveslot;

const LEARNING_RATE: f32 = 0.01;

#[derive(Debug, Clone)]
pub struct NeuralNet<F>
where
	F: Fn(f32) -> f32
{
	layers: Vec<NeuronLayer>,
	hidden_activation: F,
}

impl<F> NeuralNet<F>
where
	F: Fn(f32) -> f32,
{
	pub fn gen_random(rng: &mut impl Rng, layer_sizes: &[usize], hidden_activation: F) -> Self {
		Self {
			layers: (0..layer_sizes.len()-1).map(|n| NeuronLayer::gen_random(rng, layer_sizes[n], layer_sizes[n+1])).collect(),
			hidden_activation,
		}
	}

	pub fn forward_and_choose(&self, inputs: &[f32]) -> Moveslot {
		let n = softmax_then_select(&self.forward(inputs));
		Moveslot::from_number(n)
	}

	fn forward(&self, inputs: &[f32]) -> Vec<f32> {
		let mut output = inputs.to_owned();
		let (output_layer, hidden_layers) = self.layers
			.split_last()
			.expect("network needs at least one layer");

		for layer in hidden_layers {
			output = layer.forward(&output);
			output = output.iter()
				.map(|x| (self.hidden_activation)(*x))
				.collect();
		}

		output = output_layer.forward(&output);

		output
	}
}

#[derive(Debug, Clone)]
pub struct NeuronLayer {
	pub neurons: Vec<Neuron>
}

impl NeuronLayer {
	pub fn gen_random(rng: &mut impl Rng, inputs: usize, outputs: usize) -> Self {
		Self {
			neurons: vec![Neuron::gen_random(inputs, rng); outputs]
		}
	}
	pub fn forward(&self, input: &[f32]) -> Vec<f32> {
		self.neurons.iter().map(|neuron| neuron.forward(&input)).collect()
	}

	fn backward(&mut self, incoming_error: &[f32], layer_input: &[f32]) -> Vec<f32> {
		// we need to take into account the error created by our neuron in all the incoming errors
		// then move it in the combined direction
		// then we need to return a vector of all the errors of our neurons
		// (but what does error of our neurons mean?????)
		vec![0.0]
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

	pub fn backprop(&mut self, gt: f32, inputs: &[f32], our_action_taken: bool, p_ours: f32) {
		assert!(inputs.len() == self.weights.len());

		// we need to backprop on each weight in the neuron
		let indicator = if our_action_taken { 1.0 } else { 0.0 };

		for index in 0..self.weights.len() {
			self.weights[index] += LEARNING_RATE * gt * inputs[index] * (indicator - p_ours);
		}
		self.bias += LEARNING_RATE * gt * (indicator - p_ours);
	}

}

fn softmax(logits: &[f32]) -> Vec<f32> {
	let divisor: f32 = logits.iter().map(|x| E.powf(*x)).sum();
	logits.iter().map(|x| E.powf(*x) / divisor).collect()
}

fn softmax_then_select(logits: &[f32]) -> usize {
	let probabilities = softmax(logits);
	let random_selection = rand::random();
	let mut counter = 0.0;
	for (index, prob) in probabilities.into_iter().enumerate() {
		counter += prob;
		if counter >= random_selection {
			return index;
		}
	}
	0
}