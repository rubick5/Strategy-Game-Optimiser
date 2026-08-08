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

#[cfg(test)]
mod tests {
	use super::*;
	fn test_neuron1() -> Neuron {
		Neuron {
			weights: vec![1.0, 2.0, 3.0],
			bias: 0.0,
		}
	}

	fn test_neuron2() -> Neuron {
		Neuron {
			weights: vec![4.0, 5.0, 6.0],
			bias: 2.5,
		}
	}


	fn test_neuron3() -> Neuron {
		Neuron {
			weights: vec![7.0, 8.0, 9.0],
			bias: 5.0,
		}
	}

	fn test_neuron_layer() -> NeuronLayer {
		NeuronLayer {
			neurons: vec![test_neuron1(), test_neuron2(), test_neuron3()],
			most_recent_input: vec![],
			most_recent_output: vec![],
		}
	}

	#[test]
	fn test_forward() {
		let input = vec![1.0, 4.0, 5.0];
		let output1 = test_neuron1().forward(&input); // assumes neuron.forward works
		let output2 = test_neuron2().forward(&input);
		let output3 = test_neuron3().forward(&input);
		assert_eq!(test_neuron_layer().forward(&input), vec![output1, output2, output3]);
	}

	#[test]
	fn test_backprop() {
		let mut layer = test_neuron_layer();
		layer.forward(&vec![1.0, 2.0, 3.0]);
		layer.backward(&vec![1.0, 2.0, 3.0], 0.5);

		assert_eq!(layer.neurons, vec![
			Neuron {
				weights: vec![0.9, 1.9, 2.9],
				bias: -0.1,
			},
			Neuron {
				weights: vec![3.9, 4.9, 5.9],
				bias: 2.4,
			},
			Neuron {
				weights: vec![6.9, 7.9, 8.9],
				bias: 4.9,
			},
		]);
	}
}