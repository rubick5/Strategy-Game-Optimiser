/// Level every creature is treated as being. Only affects the damage scale.
pub const LEVEL: u32 = 50;

/// Base damage, before type effectiveness and STAB.
///
/// This is the standard Gen-3-onwards formula minus the 0.85-1.0 random roll,
/// which is left out so training stays deterministic.
///
/// The old version was `attack * power / defense` with no scaling at all, which
/// made damage roughly 2.5x too large relative to these HP pools — a 70-power
/// move took 50-110% of a target's health, so most things died in one or two
/// hits. That is what made switching strictly bad (you give up a turn and eat a
/// near-lethal hit) and residual effects pointless (poison ticks 1/8 per turn in
/// a battle that lasts three). At level 50 a neutral hit now costs a defender
/// roughly a quarter of its HP.
pub(crate) fn calculate_damage(attack_stat: u32, defense_stat: u32, base_power: u32) -> u32 {
	if base_power == 0 || defense_stat == 0 {
		return 0;
	}
	let level_factor = 2 * LEVEL / 5 + 2;
	(level_factor * base_power * attack_stat / defense_stat / 50 + 2).max(1)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A neutral hit between evenly matched creatures should be a chunk, not a
	/// kill. This is the property the whole rebalance hangs on.
	#[test]
	fn a_neutral_hit_is_about_a_quarter_of_a_health_bar() {
		let damage = calculate_damage(100, 100, 70);
		let typical_hp = 130;
		let fraction = damage as f32 / typical_hp as f32;
		assert!(
			(0.15..0.35).contains(&fraction),
			"neutral 70-power hit took {:.0}% of a {} HP bar ({} damage)",
			fraction * 100.0, typical_hp, damage
		);
	}

	#[test]
	fn damage_scales_with_the_obvious_things() {
		let base = calculate_damage(100, 100, 70);
		assert!(calculate_damage(200, 100, 70) > base, "more attack should hurt more");
		assert!(calculate_damage(100, 200, 70) < base, "more defense should hurt less");
		assert!(calculate_damage(100, 100, 140) > base, "more power should hurt more");
	}

	#[test]
	fn zero_power_deals_nothing_and_zero_defense_does_not_divide_by_zero() {
		assert_eq!(calculate_damage(100, 100, 0), 0);
		assert_eq!(calculate_damage(100, 0, 70), 0);
	}
}
