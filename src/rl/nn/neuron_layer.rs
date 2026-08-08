use serde::Deserialize;
use serde::Serialize;
use crate::rl::nn::neuron::Neuron;
use rand::RngCore;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronLayer {
	most_recent_input: Vec<f32>,
	most_recent_output: Vec<f32>,
	pub neurons: Vec<Neuron>
}

impl NeuronLayer {
	pub fn gen_random(rng: &mut dyn RngCore, inputs: usize, outputs: usize) -> Self {
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

	pub fn backward(&mut self, incoming_error: &[f32], learning_rate: f32) -> Vec<f32> {
		// we need to calculate the error to give to the next row here before we update the weights

		let mut d_prev: Vec<f32> = vec![0.0; self.neurons[0].weights.len()];
		for neuron_index in 0..self.neurons.len() {
			for index in 0..d_prev.len() {
				d_prev[index] += self.neurons[neuron_index].weights[index] * incoming_error[neuron_index];
			}
		}

		for (index, neuron) in self.neurons.iter_mut().enumerate() {
			neuron.backprop(incoming_error[index], &self.most_recent_input, learning_rate);
		}

		d_prev
	}

}
