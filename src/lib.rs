pub mod model;
pub mod battle;
pub mod rl;
// The only part of this crate that needs a window, and the only reason it would
// need system graphics libraries. Behind a feature so that `cargo build` and
// `cargo test` work on a bare machine — a reviewer should not hit an X11 error
// from a GUI they did not ask for.
#[cfg(feature = "gui")]
pub mod game_window;
pub mod cfr;
