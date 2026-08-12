pub(in crate::battle::engine) fn calculate_damage(attack_stat: u32, defense_stat: u32, base_power: u32) -> u32 {
	attack_stat * base_power / defense_stat
}