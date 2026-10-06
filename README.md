# strat-optimizer

A Pokémon-style battle engine with **two independent learners** built on top of it:

- **`src/rl/`** — a PPO self-play agent that learns a policy playing well across many positions.
- **`src/cfr/`** — a CFR matchup solver that takes *one* position and works out what unexploitable play there looks like.

They answer different questions and neither replaces the other. PPO is a player.
CFR is a **study tool** — point it at a position and get back equilibrium action
frequencies, the same use case as a poker solver.

---

## Why both

The property that makes the solver possible: with known teams there is **no
private information**, only simultaneous action selection. An information set is
therefore just the public `BattleState`, a position can be solved in isolation,
and none of the belief-state machinery a poker solver needs applies here.

That also buys something poker solvers cannot have: **exploitability is
computable rather than estimated**. A best response can be enumerated exactly, so
"how much would a perfect opponent gain against this?" has a real answer.

---

## Quick start

```bash
cargo build --release
cargo test --release          # 217 tests
```

### Train the PPO agent

```bash
cargo run --release                          # 5,000 batches -> agent.json
cargo run --release -- <batches> <out-path>  # a shorter run
```

Self-play against past selves plus a league of purpose-trained **exploiters**
(retrained every 200 batches). A mature run is roughly 45% exploiters, 47% past
selves, 8% fixed opponents. Win rates against the fixed opponents print as it goes.

The default is 160,000 battles and took **3h52m** measured end to end. Budget
~2.8s per batch rather than the ~1.1s a batch actually costs: the run also
trains 25 exploiters of 300 batches each, and they are more than half the bill.
(A short rehearsal will mislead you here — under 200 batches no exploiter is
built at all, so it clocks in at ~1.4s and suggests two hours.)
A shorter run is a real small run rather than the first slice of a long one:
the opponent mix and the learning-rate decay are scheduled as a *fraction* of
the run, so passing fewer batches compresses the whole curriculum into them.

### Measure how exploitable an agent is

```bash
cargo run --release --bin exploit [agent.json]
```

Trains a throwaway prober whose only job is to find holes in the saved agent,
and reports how far above the 50% reference it got — about 15 minutes. It scores
**the worst of the two seats, not their average**, because an agent is only as
unexploitable as the side it plays worst.

Note the saved file is a bare policy net (the PPO *actor*), which is why a
PPO-trained `agent.json` loads here as a `BotAgent`: both choose moves by
`forward` → `mask.apply` → `softmax_then_select`, so only the weights differ.

### Play against an agent yourself

```bash
cargo run --release --bin game --features gui
```

The graphical client is the only part needing system graphics libraries, so it is
opt-in — everything else, both learners and every measurement included, builds and
tests with no system dependencies.

### Solve a position

```bash
cargo run --release --bin solve           # the built-in 1v1 and 2v2 fixtures
```

### The solver's main entry point

```bash
# Solve one position for as long as you let it, reporting as it goes
cargo run --release --bin deep -- solve <position> <lookahead> <iterations> [report-every]

# Team preview: which creature should each side lead, and how often?
cargo run --release --bin deep -- leads <position> <lookahead> <iterations>

# Diagnostics
cargo run --release --bin deep -- contraction [walk] [truth-iters] [solve-iters] [noise]
cargo run --release --bin deep -- sensitivity <position> <iterations> [noise] [samples]
cargo run --release --bin deep -- horizon <position> <a> <b> <one> [iters] [seeds] [max-depth]
cargo run --release --bin deep -- train-critic <out-path> [rounds]

# How often does a PPO learner playing whole battles beat the solver?
cargo run --release --bin deep -- probe <position> <lookahead> <solver-iters> [batches] [battles]
```

`<position>` is a builtin name or a path to a battle JSON. Any solving command
takes `--leaf health | resource | critic:<path>` and `--quiescence <n>`.

A worked example — 36 lead matchups on a six-a-side, about 30 minutes:

```bash
cargo run --release --bin deep -- leads example_battles/six_asymmetric.json 4 4000
```

---

## How it works

### The engine (`src/battle/`, `src/model/`)

Turn-based, simultaneous-move, with types, abilities, status, volatiles, weather
and speed ties. Every field bottoms out in an integer, so a state hashes exactly —
which is what lets the solver key information sets on the position itself.

### PPO (`src/rl/`)

Actor-critic over a 799-float encoding of the position (per creature: stats, HP
fraction, stat stages, status, ability, typing, volatiles, counters; plus a team
matchup matrix). Action space is 10: four moves plus five switches plus a
replacement slot, masked to what is legal.

`src/rl/exploit.rs` trains a **prober** — an agent whose only job is to find holes
in a fixed opponent. It is the sharpest available check that an agent *is*
exploitable, though not proof that one is not.

### CFR (`src/cfr/`)

**External-sampling MCCFR.** Outcome sampling was tried first and diverged: at
depth 25 the sampling probability reached 1.9e-10, so importance weights hit 5.3e9.
External sampling needs no importance weighting and replaced it.

Roughly in dependency order:

| module | what it does |
|---|---|
| `key` | identifying a decision point; canonicalises volatiles so insertion order cannot split one infoset into several |
| `infoset` | regret matching and the average strategy (only the *average* converges) |
| `matrix` | matrix games with known equilibria, driving the same code — the validation harness |
| `node` | reading "who decides, and what may they do" out of the engine |
| `leaf` | estimating positions the search stops short of |
| `solver` | the search itself |
| `exploit` | exact best-response exploitability |
| `critic` | a leaf estimate the solver trains for itself |
| `contraction` | how much of a leaf's error actually reaches the root |
| `horizon` | does an answer depend on where the search stops? |
| `position` | the positions to point all this at |

---

## Results

### The PPO agent

A full run — 5,000 batches, 160,000 battles, 3h52m, plus 23 minutes to probe it.
Across all of it, **zero timeouts**: every one of those battles reached a real
result. Mean battle length fell from 18.2 turns to 14.0 as the agent sharpened.

**First, a caveat that colours every number below.** The training position
`fair_start_battle` is not fair — not in the way it first appears, and the
difference matters. Its two teams are entirely different creatures,
thornbeast/mireling/stonewarden against cinderfox/brackenox/gustling, and when a
copy of the trained agent plays itself team one wins **71%** to team zero's
**23%**.

The obvious reading, which this README gave at first, is that a shared
perspective-relative policy reveals the *position's* imbalance. That was wrong,
and solving the position says so: at equilibrium the value to team zero is only
about **−0.04 to −0.07**, essentially level. A shared policy reveals imbalance
*under that policy*, including its own asymmetric mistakes — only equilibrium
play reveals the board. So the 71/23 is the agent playing team zero roughly 23
points worse than it needs to, on a position that is close to fair.

Which lands on the same conclusion from the other direction: every PPO figure
below is measured on one position, and the agent has a seat it plays badly.

Win rate against the four fixed opponents, at the random initialisation and at
the checkpoint that was saved:

| opponent | at init | best (batch 4800) |
|---|---|---|
| random | 62% | **94%** |
| spam-0 | 22% | **96%** |
| spam-1 | 58% | **100%** |
| spam-2 | 72% | **100%** |
| **overall** | **53.5%** | **97.5%** |

The more interesting number is the league's own. Every 200 batches an exploiter
is trained from scratch against a frozen copy of the agent, on an **identical
300-batch budget** whichever point of the run it is built at — so its score is a
like-for-like read on whether the agent currently has a hole:

```
  batch  200    90%        batch 3000    73%
  batch  600    85%        batch 3600    72%
  batch 1000    93%        batch 4200    18%
  batch 1800    93%        batch 4600    32%
  batch 2400    83%        batch 4800     0%
```

For most of the run a dedicated exploiter beats the agent 70–93% of the time.
Over the last few hundred batches that collapses. At batch 4800 an exploiter
trained solely to beat this agent **never won a single battle** — and that figure
is the *peak* over the exploiter's own training, not its final score.

The independent probe agrees. 800 batches against the saved agent:

```
                 prober     mirror  exploitable   baseline
  as team zero      5.0%      23.0%         0.0      11.0%
  as team one       2.0%      71.0%         0.0       2.0%

  target 100 / prober 0 / draws 0   over 100 battles
```

**That 0.0 should not be read as "unexploitable".** The prober won 5% as team
zero where spamming a single move wins 11% — it lost to a bot with no policy at
all. The log says why: **15 of its 18 match-ups have a *switch* as the top
action**, against 0 of 18 for the target, and the traces show it pivoting back
and forth — `-> mireling`, `-> thornbeast`, `-> mireling` — absorbing damage
every turn and never attacking.

That is not an exploit that failed, it is a policy that never formed. The prober
won 0–2.5% of battles throughout its 800 batches, so the policy gradient had
almost no signal to work with and what little it had came from a handful of
lucky episodes, whose contents got amplified. A spam bot beats that because it
at least attacks.

The failure is structural and worth stating plainly: **this probe method breaks
down exactly when the target gets strong.** A prober learns from the games it
wins, so against an opponent it cannot beat it cannot bootstrap — and that is
precisely the regime where an exploitability measurement is most wanted. It
worked earlier in the run (exploiters scoring 70–93%) because the agent was
still weak enough to lose sometimes, and it works against the CFR solver, where
the same method climbs from 8.9% to 50.7% — so the method is sound and this
particular application of it is not.

Which also means the 2% here and the 50.7% against the solver **cannot be put
side by side**. One is a prober that learned; the other is a prober that did
not, on a different position, at a seventh of the sample size.

One more thing worth recording rather than celebrating: the saved policy is
close to **deterministic** on one seat, putting 85% of its mass on a single
opening action as team zero. In a simultaneous-move game that is normally the
shape of something exploitable.

**The solver says it is.** Pointed at that exact position at lookahead 6, the
equilibrium for team zero is spread across *every one of its six legal actions*:

```
  solver                          PPO agent
  switch to mireling    30%       one action     85%
  sap seed              24%       everything else 15%
  thorn whip            15%
  rock smash            15%
  switch to stonewarden  9%
  earth spike            7%
```

Nothing above 30%, and that shape is stable from iteration 400 onward even though
the exact split is not — the run hit its 50M node budget at 6800 of 8000
iterations and went memory-bound holding 15M infosets, so the percentages are
provisional and the *qualitative* answer is not.

So the hole is real and the probe missed it, which is the clearest statement of
why that 0.0 should not be read as "unexploitable". Two methods disagreed, and
the one that enumerates beat the one that learns.

### The solver works

| measurement | result |
|---|---|
| 2v2 exact best-response exploitability | **0.071** on a −1..+1 scale |
| PPO prober against the solver, full battles | **50.7%** to the prober (n=3,000) |
| Six-a-side mirror, lead-choice exploitability | **0.00000** |
| Six-a-side asymmetric, lead-choice exploitability | **0.00008** |

For scale, a best-response gap of 0.071 on a range of 2 is 3.5% — poker solvers
target ~0.5% of pot. Different units, but the same order of magnitude rather than
a different universe.

### The prober fights the solver to a draw

The one measurement that ignores the horizon entirely: train a PPO learner whose
only job is to beat the solver, let it play **whole battles**, and see what it
gets. The solver re-solves from wherever it is standing, so there is no stale
table and no coverage gap — `0 blind decisions out of 174,401`.

On `switch_prediction_2v2`, lookahead 4, 1,500 measured battles a seat:

| | prober | mirror (solver vs itself) | exploitable | best fixed baseline |
|---|---|---|---|---|
| as team zero | 49.9% | 52.1% | **0.0 pts** | 8.7% |
| as team one | 51.5% | 45.9% | **5.6 pts** | 6.7% |

**It cannot get above even.** And it used to: the same measurement before the
horizon work read 54.3% and 55.7% across two seeds at the same sample size, so
a dedicated exploiter has gone from winning 55% to 50.7%.

Two things to keep in mind reading that. It is not a perfectly controlled
comparison — quiescence is on now and the solver's budget may differ from the
earlier run — though the sample sizes are identical and the direction is not
subtle. And the 5.6 points is soft: this is a *mirror* position whose true value
is zero, so the reference ought to be 50% and came back 52.1/45.9. Against that
much scatter, a prober at 51.5% is within noise of even.

Reproduce with `deep probe` — 7.7 minutes, because the resolver answers 97.9% of
its positions from cache.

### Six-a-side team preview

`six_asymmetric`, lookahead 4, 4000 iterations a matchup, ~30 minutes:

```
  team zero    brackenox 51.3%   thornbeast 48.7%
  team one     gustling  54.9%   thornbeast 45.1%
  value to team zero  -0.0120
```

Both supports satisfy the equilibrium indifference condition to within 0.0002
while every excluded lead is clearly worse — checkable by hand from the printed
matrix. The value held at −0.012 across lookaheads 2, 3 and 4, so the answer does
not depend on where the search stops.

The mirror (`six_mirror`) answers "gustling 100%" for both seats, value +0.0015.
That is correct and useless: a mirror's lead matrix is antisymmetric, so both
seats get the same answer by construction. It is a *correctness* check, not a
study position.

### What the leaf estimate is worth

Scored against 91 duels solved to terminal, where the true values have spread
sd = 0.9070:

| estimator | RMS error | R² |
|---|---|---|
| health heuristic | 0.8690 | **+0.082** |
| depth-4 solve using it | 0.0198 | **+1.000** |

The hand-written heuristic explains **8%** of the variance in true position
values. A four-turn search built out of it explains essentially all of it.

### Why four attempts to improve the leaf all failed

`γ_s` measures how much of a *known* injected leaf error survives to the root —
output change per unit of input error.

| depth | 1v1 | 2v2 | 6v6 |
|---|---|---|---|
| 1 | 0.471 | 0.423 | 0.234 |
| 2 | 0.095 | 0.253 | 0.103 |
| 3 | — | 0.126 | 0.053 |
| 4 | 0.038 | 0.145 | 0.161 |

Roughly halving per turn of depth, and smaller in bigger games. **At lookahead 4,
six-sevenths of any leaf error is destroyed before it reaches the answer.**

That is one explanation for four separate failures: a learned critic, a ReBeL
bootstrapping loop, multi-valued leaves, and a sibling-difference loss (refuted by
measurement before being built) all landed at parity with the heuristic. They were
all improving a quantity that gets multiplied by ~0.15 on its way to the answer.

Raw estimator quality, on a smaller 68-duel sample:

| leaf | R² |
|---|---|
| health heuristic | +0.094 |
| resource heuristic (adds stat stages, status, Substitute, Leech Seed) | +0.106 |
| trained critic | **+0.344** |

The critic is by far the best *predictor* and has never once produced better
*play*. On a six-a-side it valued a level position at **−0.36** against a true
≈ −0.02, because its training curriculum tops out at 2v2 and a 6v6 encoding is
out of distribution for it.

### The horizon fault was Protect, not setup

A blade-dance (spending a turn increasing your attack rather than attacking) moveset
looked underpriced at shallow depth. Two probes differing in exactly one move settled 
what was actually happening:

| probe | depth 2 | depth 3 | depth 4 |
|---|---|---|---|
| original (two moves differ, incl. blade dance) | −0.0321 | +0.0009 | −0.0110 |
| `guard_probe` (guard vs a plain move) | −0.0310 | +0.0004 | −0.0119 |
| `plain_probe` (two plain moves) | +0.0007 | −0.0003 | −0.0006 |

Deleting blade dance entirely reproduces the original within its error bars, and
two ordinary moves show no gap at any depth. The effect belongs to **Protect**.

The mechanism: Protect's every cost — the free turn the opponent gets, the streak
that makes the next one fail, the lack of progress — falls on the turn *after* it,
and at the horizon there is no turn after. So the last turn of any search offers a
free damage block.

**Quiescence** fixes it: carry a position still in motion past the horizon rather
than pricing it there. The artifact goes from 0.039 to 0.001 at depth 2 while the
control stays flat, for ~25% more time — where buying two real turns of lookahead
costs twenty-five fold. On by default (`DEFAULT_QUIESCENCE = 2`).

Separately, `delayed_setup` shows the solver handles genuine setup fine: given a
position where blade dance is correct, it plays it **0% of the time at lookahead 2
and 99% from lookahead 3**. The decision converges at depth 3; the *valuation*
takes until depth 5. Lead matrices are built from values, which is why they
inherit the slower of the two.

### Speed

| change | effect |
|---|---|
| Memoising an expensive leaf | **26× / 14×** faster at depths 2 / 3 |
| Transposition-aware search | 5–13% — "worth about a tenth of a turn" |

---

## Progression

| era | what it is | status |
|---|---|---|
| `RandomAgent`, `SpamAgent` | uniform / fixed-slot baselines | fixed opponents, unchanged |
| `BotAgent` | the original learner — single net, REINFORCE-style with a running baseline | superseded |
| `PPOAgent` | actor-critic, masked action space, self-play | the player |
| + league | purpose-trained exploiters every 200 batches | current PPO |
| `src/cfr/` | CFR matchup solver | the study tool |

Measured end points, where they exist:

| | overall vs fixed opponents | exploitable by an 800-batch prober |
|---|---|---|
| random initialisation | 53.5% | — |
| `PPOAgent` + league, 5,000 batches | **97.5%** | **0.0 points** |
| `BotAgent` | not measured | not measured |

> ⚠️ **`BotAgent`'s numbers were never recorded and are not reproducible cheaply.**
> The old `src/agent.json` is from August, predates the encoder change
> (655 → 799 inputs) and no longer loads, so the only way to fill that row is to
> train one — another four hours. It is the earlier architecture and nothing
> depends on it, so this is left open rather than guessed at.

The PPO and CFR figures answer different questions on different positions and
are **not comparable to each other**. One is a win rate against a fixed pool on
`fair_start_battle`; the other is a best-response gap on a specific solved
position. Neither is evidence about the other.

---

## Open questions

- **How do you measure the exploitability of a strong agent?** The current probe
  learns from the games it wins, so it stops working precisely when the target
  stops losing — it collapsed into a switch-loop and lost to a spam bot. More
  batches may not fix a policy with no gradient to climb; shaping the prober's
  reward, or seeding it from the target's own weights, probably would. Until
  then the 0.0 is uninformative.
- **Why does the PPO agent play team zero so badly?** The solver puts that seat's
  equilibrium value near level and spreads its play across all six actions; the
  agent loses 23 points from it and plays one action 85% of the time. Training on
  a single position is the obvious suspect, and training across several would test
  it.
- **The lookahead-6 solve did not converge.** It hit the 50M node budget at 6800
  of 8000 iterations and went memory-bound at 15M infosets, each holding a full
  cloned `BattleState`. Digesting the infoset key is the fix, and it is also what
  stands between this and solving six-a-side positions deeper.
- **`BotAgent` has no figures.**
- **A 6v6 curriculum for the critic.** Until it trains on positions the size of
  the ones it is asked about, its 6v6 estimates mean nothing.
- **The depth-3 residual** in the guard sweep sits at −0.012 and is unmoved by
  quiescence of 2, 4 or 6. Unexplained.
- **Head-to-head CFR vs PPO.** Deliberately not built yet.

---

## Licence

All rights reserved — see [LICENSE](LICENSE). The code is published to be read,
not used: no permission is granted to use, copy, modify or distribute it, and a
fork is a copy rather than a licence. Read it, learn from it, ask if you want to
do anything more.

## Layout

```
src/battle/     engine, state, hooks, effects
src/model/      registry: species, moves, types, abilities
src/rl/         PPO, encoder, league, exploiter prober, neural nets
src/cfr/        the matchup solver and its diagnostics
src/bin/        deep (solver CLI), solve, game, exploit
example_battles/  positions, including the six-a-side fixtures
```

## A note on the measurements

Several results in this project were overturned by a second look — a coverage
floor that measurement refuted, a "+0.0107 improvement" that was noise, a
single-seed selection that overfitted, a sibling-difference theory killed before
it was built, and a parity mechanism asserted without evidence and withdrawn. The
diagnostics carry their own controls for that reason: `contraction` reports a
noise floor beside every sensitivity figure, `horizon` reports a standard
deviation beside every difference, and `plain_probe` exists purely to say what
*should not* move. Treat a number here without a control next to it as the weaker
kind of claim.
