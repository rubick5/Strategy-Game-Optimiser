//! The one place hook ordering is decided.
//!
//! Turn order used to be implied by the position of a `match` arm inside
//! `end_turn_resolution::END_OF_TURN_ORDER`, which meant adding an effect meant
//! editing that list. Now an effect declares its own slot on this shared scale
//! and the table sorts everything for you.
//!
//! Lower numbers run first. Gaps of 100 are deliberate: you can slot a new
//! effect between two existing ones without renumbering anything.

// ---------------------------------------------------------------------------
// Residual phase (TriggerKind::Residual)
//
// Mirrors the ordering that `END_OF_TURN_ORDER` was documenting.
// ---------------------------------------------------------------------------

/// Sandstorm / hail chip damage.
pub const WEATHER_DAMAGE: i16 = 100;
/// Future Sight and friends landing.
pub const DELAYED_ATTACKS: i16 = 200;
/// Wish and other queued healing.
pub const DELAYED_HEALING: i16 = 300;
/// Held-item healing (Leftovers).
pub const ITEM_HEALING: i16 = 400;
/// Poison and Bad Poison chip.
pub const POISON: i16 = 500;
/// Burn chip.
pub const BURN: i16 = 600;
/// Wrap / Bind / Whirlpool chip.
pub const BINDING_MOVES: i16 = 700;

// ---------------------------------------------------------------------------
// Turn end (TriggerKind::TurnEnd)
// ---------------------------------------------------------------------------

/// Weather ticking down and expiring. Runs after residuals so the weather that
/// was active during the turn is the one that deals its damage.
pub const WEATHER_COUNTDOWN: i16 = 100;

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Hard vetoes: immunities and "this can't happen" rules. Run first so that a
/// blocked action never reaches the handlers that would have scaled it.
pub const IMMUNITY: i16 = 0;
/// Multiplicative stat and damage modifiers.
pub const MULTIPLIER: i16 = 100;
/// Final clamps and floors, after everything else has had its say.
pub const FINAL_ADJUSTMENT: i16 = 900;

/// Neutral slot for hooks that genuinely do not care when they run.
pub const DEFAULT: i16 = 500;
