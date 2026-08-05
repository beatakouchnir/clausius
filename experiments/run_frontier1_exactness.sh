#!/bin/sh
# Unattended frontier chain. Ordered by value, not by convenience: if this dies
# halfway the exactness verdict and the speed curve are already on disk.
#
# NO pgrep WAITER. An earlier overnight script in this repo blocked on
# `while pgrep -f "stage_a --task gpqa"` and the waiter's own command line
# matched the pattern, so it would have spun until morning. This chain assumes
# the GPU is free and just runs.
#
# One process per configuration, because wrap() mutates the model in place:
# re-wrapping in a single process keeps the old cache referenced and the memory
# saving evaporates. Sequential invocation is the isolation.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1

step() {
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE rc=$? $(date +%H:%M) ==="
}

echo "### frontier chain start $(date) ###"

# ---------------------------------------------------------------- 1. exactness
# The load-bearing claim: policy='exact' fetches on miss, so shrinking the cache
# must not change a single token. If these diverge, accuracy IS capacity-
# dependent and the whole plan needs the sweep back.
step exact-gemma-full $QV -m knowledge.frontier gen --model gemma --capacity 128 --policy exact --tag full
step exact-gemma-c32  $QV -m knowledge.frontier gen --model gemma --capacity 32  --policy exact --tag c32
step exact-gemma-cmp  $QV -m knowledge.frontier compare --model gemma --a full --b c32

step exact-qwen-full  $QV -m knowledge.frontier gen --model qwen --capacity 256 --policy exact --tag full
step exact-qwen-c64   $QV -m knowledge.frontier gen --model qwen --capacity 64  --policy exact --tag c64
step exact-qwen-cmp   $QV -m knowledge.frontier compare --model qwen --a full --b c64

# ------------------------------------------------------------- 2. speed curve
# Replaces the modelled tax with measured tok/s. This is the axis the frontier
# needs and the one my cost model only estimated (and estimated optimistically:
# reads get smaller and more scattered as capacity falls).
for C in 128 96 64 48 32 24; do
  step speed-gemma-c$C $QV -m knowledge.frontier speed --model gemma --capacity $C --policy exact
done
for C in 256 192 128 96 64 48 32; do
  step speed-qwen-c$C $QV -m knowledge.frontier speed --model qwen --capacity $C --policy exact
done

# --------------------------------------------------- 3. the LOSSY arm (static)
# policy='static' preloads the hot experts and zeroes everything else, so this
# is real accuracy-vs-memory. cap-128 is the wiring control: with every expert
# pinned nothing can be zeroed, so it must reproduce the resident accuracy
# (gemma popqa 0.225). If it does not, the pins are wrong and the rest is noise.
# Prior from the traces: the top 25% of experts serve only 50.7% of decisions,
# so low-capacity static should be badly damaged. That is the prediction.
for C in 128 96 64 32; do
  step static-gemma-popqa-c$C $QV -m knowledge.frontier acc --model gemma --task popqa --capacity $C --policy static --n 200
done
for C in 128 64 32; do
  step static-gemma-gsm8k-c$C $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity $C --policy static --n 200
done
for C in 256 128 64; do
  step static-qwen-popqa-c$C $QV -m knowledge.frontier acc --model qwen --task popqa --capacity $C --policy static --n 200
done
for C in 256 128 64; do
  step static-qwen-gsm8k-c$C $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity $C --policy static --n 200
done

echo "### frontier chain done $(date) ###"
