#[derive(Debug, PartialEq, Copy, Clone)]
pub enum NonVolatileStatus {
	None,
	Poison,
	BadPoison,
	Burn,
}