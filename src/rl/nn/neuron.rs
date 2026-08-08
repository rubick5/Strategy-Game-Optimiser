use rand::{Rng as _, RngCore};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

const CLIP: f32 = 0.1;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

	pub fn gen_random(weight_count: usize, rng: &mut dyn RngCore) -> Self {
		let distribution = Normal::new(0.0, (2.0 / weight_count as f32).sqrt()).unwrap();
		let weights = rng.sample_iter(distribution).take(weight_count).collect();
		let bias = 0.0; //rng.random_range(-2.0..2.0);

		Neuron {
			weights,
			bias
		}
	}
	pub fn backprop(&mut self, error: f32, input_given: &[f32], learning_rate: f32) {
		for index in 0..self.weights.len() {
			self.weights[index] = (self.weights[index] - (learning_rate * error * input_given[index]).clamp(-CLIP, CLIP)).clamp(-10.0, 10.0);
		}
		self.bias -= (learning_rate * error).clamp(-CLIP, CLIP);
	}
}