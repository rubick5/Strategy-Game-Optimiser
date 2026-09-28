#!/bin/bash
# Train a PPO agent and then probe it, unattended.
#
#   ./run_overnight.sh [batches] [out-path]
#
# Survives the terminal closing, keeps the Mac awake, and lets the screen go
# dark. Progress lands in train.log and exploit.log.

set -u
BATCHES="${1:-5000}"
OUT="${2:-agent.json}"
cd "$(dirname "$0")"

# Build first. `cargo run` would rebuild mid-script otherwise, and a compile
# error at 2am should stop this now rather than after two hours of training.
cargo build --release || { echo "build failed, not starting"; exit 1; }

{
  echo "=== started $(date) : $BATCHES batches -> $OUT ==="
  # -i prevents the machine idling to sleep. Deliberately NOT -d, which would
  # hold the *display* awake — the whole point is to let the screen go dark
  # while the CPU keeps working. -s covers system sleep on AC power.
  caffeinate -is ./target/release/strat-optimizer "$BATCHES" "$OUT"
  echo "=== training finished $(date), exit $? ==="
} > train.log 2>&1

if [ -f "$OUT" ]; then
  {
    echo "=== probing $OUT, started $(date) ==="
    caffeinate -is ./target/release/exploit "$OUT"
    echo "=== probe finished $(date), exit $? ==="
  } > exploit.log 2>&1
else
  echo "no $OUT was written, so there is nothing to probe" >> exploit.log
fi

echo "=== all done $(date) ===" >> train.log
