//! Concrete effects, written as hooks.
//!
//! Each module here exposes a `hooks(...) -> &'static [HookDef]` function: given
//! a status, a weather, an ability, hand back the static list of moments it cares
//! about. An effect's entire behaviour is one readable table plus the handlers it
//! names — nothing about it is spread through the engine.

pub mod ability;
pub mod status;
pub mod volatile;
pub mod weather;

/// A fraction of a creature's maximum HP, floored at 1.
///
/// Chip damage in the games never rounds down to nothing, and the old
/// `max_hp / 8` could, for a creature with fewer than 8 max HP.
pub(crate) fn fraction_of_max(max_hp: u32, denominator: u32) -> u32 {
	if denominator == 0 {
		return 0;
	}
	(max_hp / denominator).max(1)
}
