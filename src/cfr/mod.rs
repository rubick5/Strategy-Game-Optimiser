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
//!
//! The modules, roughly in dependency order:
//!
//! * [`key`] — identifying a decision point so that equal positions share
//!   regrets and unequal ones do not.
//! * [`infoset`] — regret matching and the tables it accumulates into. Knows
//!   nothing about battles.
//! * [`matrix`] — matrix games with known equilibria, driving the same code as
//!   the battle solver. This is the validation harness, and the reason to trust
//!   anything below it.
//! * [`node`] — reading "who decides, and what may they do" out of the engine.
//! * [`leaf`] — estimating positions the search stops short of, which a 2v2
//!   cannot do without.
//! * [`solver`] — external-sampling MCCFR over battle positions.
//! * [`exploit`] — how much a perfect opponent would gain. The one check that
//!   does not need the answer known in advance, and the metric everything else
//!   should be judged against.
//! * [`critic`] — a leaf estimate the solver trains for itself. Does not yet
//!   beat the hand-written one; see its own notes.
//! * [`position`] — the positions to point all this at.

pub mod critic;
pub mod exploit;
pub mod infoset;
pub mod key;
pub mod leaf;
pub mod matrix;
pub mod multivalue;
pub mod node;
pub mod position;
pub mod probe;
pub mod resolve;
pub mod solver;
