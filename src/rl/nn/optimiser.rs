//! Adam, for training on noisy targets.
//!
//! [`NeuralNet::backward`](super::neural_net::NeuralNet::backward) updates on
//! every sample with a fixed step. That is fine when the targets are clean, and
//! poor when they are not: each individual label yanks the weights toward itself,
//! so with labels that are largely a coin flip the network spends its capacity
//! chasing noise.
//!
//! Two things help, and this module is the second of them. Averaging gradients
//! over a batch cancels noise *within* a step, complementing the averaging of the
//! targets themselves. Adam then adapts the step per weight from the history of
//! gradients it has seen, so a weight with a consistent gradient moves steadily
//! while one whose gradient keeps changing sign barely moves at all — which is
//! exactly the distinction between signal and noise here.
//!
//! This sits alongside the existing update rather than replacing it. Nothing that
//! used `backward` behaves differently.

use crate::rl::nn::neural_net::NeuralNet;

#[derive(Debug, Clone, Default)]
struct Moment {
	weights: Vec<f32>,
	bias: f32,
}

/// Adaptive moment estimation.
///
/// Holds one running mean and one running variance per parameter, so it must be
/// kept across steps — a fresh `Adam` each batch would be plain SGD with extra
/// arithmetic.
pub struct Adam {
	means: Vec<Vec<Moment>>,
	variances: Vec<Vec<Moment>>,
	step: u32,
	pub beta1: f32,
	pub beta2: f32,
	pub epsilon: f32,
	/// Weights are held to this range, matching what `Neuron::backprop` does.
	pub weight_limit: f32,
}

impl Default for Adam {
	fn default() -> Self {
		Adam {
			means: Vec::new(),
			variances: Vec::new(),
			step: 0,
			beta1: 0.9,
			beta2: 0.999,
			epsilon: 1e-8,
			weight_limit: 10.0,
		}
	}
}

impl Adam {
	pub fn new() -> Self {
		Self::default()
	}

	/// Steps taken so far.
	pub fn steps(&self) -> u32 {
		self.step
	}

	/// Apply one batch's averaged gradients.
	///
	/// `gradients` is shaped as [`NeuralNet::gradients`] returns: outermost layer
	/// first, one entry per neuron, each a weight gradient vector and a bias
	/// gradient.
	pub fn apply<F>(
		&mut self,
		net: &mut NeuralNet<F>,
		gradients: &[Vec<(Vec<f32>, f32)>],
		learning_rate: f32,
	) where
		F: Fn(f32) -> f32,
	{
		self.ensure_shape(gradients);
		self.step += 1;

		// Bias correction: both running averages start at zero, so early steps
		// understate the true moments badly. These factors undo that, and fade to
		// one as the step count grows.
		let correct1 = 1.0 - self.beta1.powi(self.step as i32);
		let correct2 = 1.0 - self.beta2.powi(self.step as i32);

		for (layer_index, layer_gradients) in gradients.iter().enumerate() {
			let layer = net.layer_mut(layer_index);
			for (neuron_index, (weight_gradients, bias_gradient)) in
				layer_gradients.iter().enumerate()
			{
				let neuron = &mut layer.neurons[neuron_index];
				let mean = &mut self.means[layer_index][neuron_index];
				let variance = &mut self.variances[layer_index][neuron_index];

				for index in 0..weight_gradients.len() {
					let gradient = weight_gradients[index];
					mean.weights[index] =
						self.beta1 * mean.weights[index] + (1.0 - self.beta1) * gradient;
					variance.weights[index] = self.beta2 * variance.weights[index]
						+ (1.0 - self.beta2) * gradient * gradient;

					let step = learning_rate * (mean.weights[index] / correct1)
						/ ((variance.weights[index] / correct2).sqrt() + self.epsilon);

					neuron.weights[index] =
						(neuron.weights[index] - step).clamp(-self.weight_limit, self.weight_limit);
				}

				mean.bias = self.beta1 * mean.bias + (1.0 - self.beta1) * bias_gradient;
				variance.bias =
					self.beta2 * variance.bias + (1.0 - self.beta2) * bias_gradient * bias_gradient;

				neuron.bias -= learning_rate * (mean.bias / correct1)
					/ ((variance.bias / correct2).sqrt() + self.epsilon);
			}
		}
	}

	fn ensure_shape(&mut self, gradients: &[Vec<(Vec<f32>, f32)>]) {
		if self.means.len() == gradients.len() {
			return;
		}
		let blank: Vec<Vec<Moment>> = gradients
			.iter()
			.map(|layer| {
				layer
					.iter()
					.map(|(weights, _)| Moment { weights: vec![0.0; weights.len()], bias: 0.0 })
					.collect()
			})
			.collect();
		self.means = blank.clone();
		self.variances = blank;
	}
}

/// Add one sample's gradients into a running batch total.
pub fn accumulate(total: &mut Vec<Vec<(Vec<f32>, f32)>>, sample: &[Vec<(Vec<f32>, f32)>]) {
	if total.is_empty() {
		*total = sample.to_vec();
		return;
	}
	for (layer, layer_sample) in total.iter_mut().zip(sample.iter()) {
		for ((weights, bias), (weights_sample, bias_sample)) in
			layer.iter_mut().zip(layer_sample.iter())
		{
			for (value, add) in weights.iter_mut().zip(weights_sample.iter()) {
				*value += add;
			}
			*bias += bias_sample;
		}
	}
}

/// Divide a batch total by how many samples went into it.
pub fn scale(total: &mut [Vec<(Vec<f32>, f32)>], divisor: f32) {
	if divisor == 0.0 {
		return;
	}
	for layer in total.iter_mut() {
		for (weights, bias) in layer.iter_mut() {
			for value in weights.iter_mut() {
				*value /= divisor;
			}
			*bias /= divisor;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use rand::rngs::StdRng;
	use rand::SeedableRng;

	fn identity(x: f32) -> f32 {
		x
	}

	/// The point of the whole thing: fitting a target the network can actually
	/// reach should reduce the error.
	#[test]
	fn adam_reduces_the_error_on_a_learnable_target() {
		let mut rng = StdRng::seed_from_u64(1);
		let mut net = NeuralNet::gen_random(
			&mut rng,
			&[4, 8, 1],
			identity as fn(f32) -> f32,
			identity as fn(f32) -> f32,
		);
		let mut adam = Adam::new();

		let input = vec![0.5, -0.25, 1.0, 0.75];
		let target = 0.6;

		let before = (net.forward(&input)[0] - target).abs();
		for _ in 0..200 {
			let error = 2.0 * (net.forward(&input)[0] - target);
			let gradients = net.gradients(vec![error], &input);
			adam.apply(&mut net, &gradients, 0.01);
		}
		let after = (net.forward(&input)[0] - target).abs();

		assert!(after < before, "error should fall: {before} -> {after}");
		assert!(after < 0.05, "should get close, got {after}");
		assert_eq!(adam.steps(), 200);
	}

	#[test]
	fn accumulating_then_scaling_gives_the_mean() {
		let mut total: Vec<Vec<(Vec<f32>, f32)>> = Vec::new();
		let first = vec![vec![(vec![1.0, 2.0], 3.0)]];
		let second = vec![vec![(vec![3.0, 4.0], 5.0)]];

		accumulate(&mut total, &first);
		accumulate(&mut total, &second);
		scale(&mut total, 2.0);

		assert_eq!(total, vec![vec![(vec![2.0, 3.0], 4.0)]]);
	}

	/// Gradient collection has to agree with the update that is already in use, or
	/// the two paths are training different networks.
	#[test]
	fn collected_gradients_match_the_existing_backward_pass() {
		let mut rng = StdRng::seed_from_u64(2);
		let sizes = [3, 5, 2];
		let mut collecting = NeuralNet::gen_random(
			&mut rng,
			&sizes,
			identity as fn(f32) -> f32,
			identity as fn(f32) -> f32,
		);
		let mut updating = collecting.clone();

		let input = vec![0.3, -0.7, 0.2];
		let errors = vec![0.4, -0.1];

		// A plain step of size `lr` down the collected gradient should land where
		// `backward` lands, since backward does exactly that (its clamps are wide
		// enough not to bite at this scale).
		let learning_rate = 0.01;
		let gradients = collecting.gradients(errors.clone(), &input);
		for layer_index in 0..collecting.layer_count() {
			let layer = collecting.layer_mut(layer_index);
			for (neuron_index, (weight_gradients, bias_gradient)) in
				gradients[layer_index].iter().enumerate()
			{
				let neuron = &mut layer.neurons[neuron_index];
				for index in 0..weight_gradients.len() {
					neuron.weights[index] -= learning_rate * weight_gradients[index];
				}
				neuron.bias -= learning_rate * bias_gradient;
			}
		}

		updating.backward(errors, &input, learning_rate);

		for layer_index in 0..sizes.len() - 1 {
			let mine = &collecting.layer_mut(layer_index).neurons.clone();
			let theirs = &updating.layer_mut(layer_index).neurons;
			for (a, b) in mine.iter().zip(theirs.iter()) {
				for (x, y) in a.weights.iter().zip(b.weights.iter()) {
					assert!((x - y).abs() < 1e-6, "weights diverged: {x} vs {y}");
				}
				assert!((a.bias - b.bias).abs() < 1e-6);
			}
		}
	}
}
