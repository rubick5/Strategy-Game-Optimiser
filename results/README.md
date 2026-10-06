# Measurement logs

Raw output from the runs the numbers in the top-level README come from. Kept
because a figure without the run behind it is an assertion, and several claims in
this project were overturned by re-reading one of these.

Two of them are mislabelled or incomplete by accident rather than design, and
both are noted below rather than quietly tidied away.

## The solver

| file | what it measured |
|---|---|
| `leads_la4.log` | `six_mirror` lead matrix, lookahead 4, 4000 iterations a matchup. Answer is "gustling 100%" for both seats, which is a correctness check rather than a study result — a mirror's matrix is antisymmetric. 3164s. |
| `leads_asym_la4.log` | `six_asymmetric` lead matrix, same budget. The real preview answer, and the one whose equilibrium indifference was checked by hand. **Quiescence 0.** 1822s. |
| `leads_asym_la4_noquiet.log` | ⚠️ **Misnamed.** Invoked with `--quiescence 0` but run at 2: the flag was parsed, echoed in the header, and ignored by `leads`. Caught because it failed to reproduce `leads_asym_la4.log`, which it should have matched exactly. Fixed in `bfa3c35`; this is the quiescence-2 result. 2131s. |
| `probe_2v2.log` | A PPO learner trained solely to beat the solver, playing whole battles. 49.9%/51.5% over 1500 battles a seat, against 54.3%/55.7% before the horizon work. 460s. |

## The leaf estimate

| file | what it measured |
|---|---|
| `leaf_health_contraction.log` | `contraction` with the health heuristic over 91 exactly-solved duels. Gives the R² of +0.082 and the γ / γ_s figures. 168s. |
| `leaf_resource.log` | Horizon sweep with `ResourceHeuristic`. The null result: counting stat stages, status, Substitute and Leech Seed moved nothing. 2986s. |
| `leaf_critic.log` | ⚠️ **Incomplete** — killed at depth 3. Projected at 83 hours because every leaf was a network forward pass; it is what motivated `Memoised`. Also shows the critic valuing a level six-a-side position at −0.36 against a true ≈ −0.02, since its curriculum tops out at 2v2. |
| `train_critic.log` | One critic training run, for the file the above loads. |

## The horizon

| file | what it measured |
|---|---|
| `bladedance_sweep.log` | The sweep that started as "the search underprices setup" and ended up being about Protect. Depths 2–5, paired seeds. Killed before depth 6, which would have taken ~5.4h for no new information. |
| `guard_probe_d5.log` | `guard_probe` at depth 5, isolating Protect to a single moveset slot. |

## PPO

| file | what it measured |
|---|---|
| `train.log` | The full 5000-batch run, 160,000 battles, 3h52m. Overall win rate against the fixed pool 53.5% → 97.5%, and the exploiter collapse from 90% to 0%. Zero timeouts throughout. |
| `exploit.log` | Probing that agent. Reports 0.0 exploitability — **do not read it that way.** The prober switch-looped (15 of 18 matchups) and lost to a spam bot, so it measures a probe that never learned. The reasoning is in the top-level README. |

## Reproducing

Every file here has its command in the top-level README, and the `Running ...`
line at the top of most of them records the exact invocation. Note that anything
dated before `da15b60` was run without quiescence, which is now on by default —
so re-running will not reproduce those numbers, by design.
