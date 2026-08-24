use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TimedWeather {
	pub weather: Weather,
	pub turns_left: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Weather {
	Sandstorm,
	HarshSun,
	Rain,
	Snow,
}