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
cargo run --release           # src/main.rs — the training loop
```

Self-play against past selves plus a league of purpose-trained **exploiters**
(retrained every 200 batches). A mature run is roughly 45% exploiters, 47% past
selves, 8% fixed opponents. Win rates against the fixed opponents print as it goes.

### Play against an agent yourself

```bash
cargo run --release --bin game
```

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

### The solver works

| measurement | result |
|---|---|
| 2v2 exact best-response exploitability | **≈ 0.07** on a −1..+1 scale |
| PPO prober against the solver, full game | **≈ 55–45** to the prober |
| Six-a-side mirror, lead-choice exploitability | **0.00000** |
| Six-a-side asymmetric, lead-choice exploitability | **0.00008** |

For scale, a best-response gap of 0.07 on a range of 2 is 3.5% — poker solvers
target ~0.5% of pot. Different units, but the same order of magnitude rather than
a different universe.

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

A blade-dance moveset looked underpriced at shallow depth. Two probes differing in
exactly one move settled what was actually happening:

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

> ⚠️ **The PPO-era win rates are not recorded anywhere.** `src/agent.json` is from
> August and predates the encoder change (655 → 799 inputs), so it no longer
> loads. Reproducing those figures needs a fresh training run — see *Open
> questions*. Every CFR number above was measured directly and is reproducible
> from the commands in this README.

---

## Open questions

- **PPO benchmark figures.** A training run plus `rl::evaluate` against the fixed
  opponents would fill the gap above.
- **A 6v6 curriculum for the critic.** Until it trains on positions the size of
  the ones it is asked about, its 6v6 estimates mean nothing.
- **The depth-3 residual** in the guard sweep sits at −0.012 and is unmoved by
  quiescence of 2, 4 or 6. Unexplained.
- **Head-to-head CFR vs PPO.** Deliberately not built yet.

---

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
