#!/bin/sh
# Follow-up chain, added after the exactness test FAILED its pre-registration:
# 10/16 identical, 6 diverging at tokens 26/132/128/36/21/101. Never at token 0,
# so not a wiring bug — capacity changes gather_qmm's reduction order via
# _gather_sort(x, slots) (slots are LRU positions, not expert ids) and via
# prefill chunking, which only triggers below full capacity.
#
# That makes 'exact' semantically exact but numerically capacity-dependent, and
# it raises two questions the original chain cannot answer.
#
# BOUNDED WAIT, not a pgrep waiter. Exits after 1200 iterations (~6.7 h) even if
# the first chain never prints its marker, so this cannot spin overnight the way
# the pgrep self-match did.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
LOG=records/frontier.log

i=0
while [ $i -lt 1200 ]; do
  grep -q "frontier chain done" "$LOG" && break
  i=$((i + 1))
  sleep 20
done
grep -q "frontier chain done" "$LOG" || {
  echo "### chain 1 never finished after ~6.7h — chain 2 aborting ###"; exit 1; }

step() {
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE $(date +%H:%M) ==="
}

echo "### frontier chain 2 start $(date) ###"

# --------------------------------------------- 1. is it deterministic per cap?
# THE control the first chain lacks. Greedy decoding has no sampling, so running
# the SAME capacity twice must give byte-identical tokens — unless there is
# run-to-run nondeterminism on top of the capacity effect. Identical => the
# divergence is a deterministic function of cache size, which is a reproducible
# property one can validate against. Different => the runtime is simply
# nondeterministic, and no output-diff test can ever validate it.
step determinism-a $QV -m knowledge.frontier gen --model gemma --capacity 64 --policy exact --tag c64a
step determinism-b $QV -m knowledge.frontier gen --model gemma --capacity 64 --policy exact --tag c64b
step determinism-cmp $QV -m knowledge.frontier compare --model gemma --a c64a --b c64b

# ------------------------------------------------- 2. the accuracy NOISE FLOOR
# If capacity perturbs the numerics, accuracy has a floor of wobble that owes
# nothing to information loss. Without this number, no static-policy or top-k
# result can be attributed: a 2pp drop might be the mechanism or might be
# rounding. popqa because it is the cheapest task in the suite (~0.5 s/item).
# Prediction: unbiased scatter around the resident value (gemma 0.225), NOT a
# monotone decline. A monotone decline would mean exact offload does lose
# information and the whole two-family split is wrong.
for C in 128 96 64 48 32; do
  step noise-gemma-popqa-c$C $QV -m knowledge.frontier acc --model gemma --task popqa --capacity $C --policy exact --n 200
done
for C in 256 192 128 64; do
  step noise-qwen-popqa-c$C $QV -m knowledge.frontier acc --model qwen --task popqa --capacity $C --policy exact --n 200
done

echo "### frontier chain 2 done $(date) ###"
