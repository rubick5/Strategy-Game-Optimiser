use rand::RngCore;
use serde::{Serialize, Deserialize};
use std::{error::Error, fs::File, io::Write};
use crate::rl::nn::neuron_layer::NeuronLayer;

#[derive(Serialize, Deserialize)]
pub struct SavedNeuralNet {
	pub layers: Vec<NeuronLayer>,
}

impl SavedNeuralNet {
	pub fn from_file(file_name: &str) -> Result<Self, Box<dyn Error>> {
		let bytes = std::fs::read(file_name)?;
		let saved: SavedNeuralNet = serde_json::from_slice(&bytes)?;
		Ok(saved)
	}

	pub fn to_file(&self, target: &str) -> Result<(), Box<dyn Error>> {
		let bytes = serde_json::to_string(&self)?;
		let mut file = File::create(target)?;
		Ok(file.write_all(bytes.as_bytes())?)
	}
}

#[derive(Debug, Clone)]
pub struct NeuralNet<F>
where
	F: Fn(f32) -> f32
{
	// layers are stored with a cache of their pre-activations from the
	// most recent forward (we populate the cache by running forward an
	// extra time before back propagation)
	layers: Vec<(NeuronLayer, Vec<f32>)>,
	hidden_activation: F,
	hidden_activation_prime: F,
}

impl<F> NeuralNet<F>
where
	F: Fn(f32) -> f32,
{
	pub fn to_file(&self, target: &str) -> Result<(), Box<dyn Error>> {
		self.to_saved().to_file(target)
	}

	pub fn from_file(file_name: &str, hidden_activation: F, hidden_activation_prime: F) -> Result<Self, Box<dyn Error>> {
		let saved = SavedNeuralNet::from_file(file_name)?;
		Ok(Self::from_saved(saved, hidden_activation, hidden_activation_prime))
	}

	pub fn to_saved(&self) -> SavedNeuralNet {
		SavedNeuralNet { layers: self.layers.clone().into_iter().map(|(l, _)| l).collect() }
	}

	pub fn from_saved(saved: SavedNeuralNet, hidden_activation: F, hidden_activation_prime: F) -> Self {
		Self {
			layers: saved.layers.into_iter().map(|l| (l, vec![])).collect(),
			hidden_activation,
			hidden_activation_prime,
		}
	}
	pub fn gen_random(rng: &mut dyn RngCore, layer_sizes: &[usize], hidden_activation: F, hidden_activation_prime: F) -> Self {
		Self {
			layers: (0..layer_sizes.len()-1).map(|n| (NeuronLayer::gen_random(rng, layer_sizes[n], layer_sizes[n+1]), vec![])).collect(),
			hidden_activation,
			hidden_activation_prime,
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

	pub fn backward_with_decay(&mut self, mut current_errors: Vec<f32>, input_received: &[f32], learning_rate: f32, decay_amount: f32) {
		self.forward(input_received); // sets the pre_activation cache for this decision made
		
		let ((output_layer, _), hidden_layers) =
			self.layers
			.split_last_mut()
			.expect("network needs at least one layer");

		current_errors = output_layer.backward(&current_errors, learning_rate, decay_amount);

		for (layer, pres) in hidden_layers.iter_mut().rev() {
			current_errors.iter_mut().enumerate().for_each(|(i, x)|
				*x = *x * (self.hidden_activation_prime)(pres[i])
			);
			current_errors = layer.backward(&current_errors, learning_rate, decay_amount);
		}
	}

	pub fn backward(&mut self, current_errors: Vec<f32>, input_received: &[f32], learning_rate: f32) {
		self.backward_with_decay(current_errors, input_received, learning_rate, 0.0);
	}
}




