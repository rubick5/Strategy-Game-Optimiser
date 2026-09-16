//! Solving a matchup to equilibrium.
//!
//! This sits alongside the PPO learner in [`crate::rl`] rather than replacing
//! it. They answer different questions: PPO learns a policy that plays well
//! across many positions, while this takes *one* position and works out what
//! unexploitable play there looks like — the poker-solver use case.
//!
//! The property that makes it possible: with known teams there is no private
//! information, only simultaneous action selection. An information set is
//! therefore just the public battle state, a position can be solved in
//! isolation, and none of the belief-state machinery a poker solver needs
//! applies here.

pub mod infoset;
pub mod key;
pub mod matrix;
pub mod position;
