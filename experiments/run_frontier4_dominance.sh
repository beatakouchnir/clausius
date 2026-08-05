#!/bin/sh
# Chain 4. Chain 3's measured memory curve opened a claim that was not reachable
# before: qwen at cap-32 runs in 3.40 GB and gemma at cap-24 in 3.76 GB, both
# BELOW e4b's measured 3.91 GB. If accuracy holds at those rungs — and the noise
# floor says it will, 0.9397/0.9444/0.9447 across a 4x capacity range — then the
# offloaded big model does not merely trade memory for accuracy against the small
# model. It DOMINATES it on both, losing only speed.
#
# That is a strictly stronger statement than "more accurate at comparable
# memory", so it runs first and the expensive knowledge task runs last.
#
# BOUNDED WAIT on chain 3 (1200 x 20s ~= 6.7h), never a pgrep waiter.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
LOG=records/frontier3.log

i=0
while [ $i -lt 1200 ]; do
  grep -q "frontier chain 3 done" "$LOG" && break
  i=$((i + 1))
  sleep 20
done
grep -q "frontier chain 3 done" "$LOG" || {
  echo "### chain 3 never finished — chain 4 aborting ###"; exit 1; }

step() {
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE $(date +%H:%M) ==="
}

echo "### frontier chain 4 start $(date) ###"

# ------------------------------------------- 1. the dominance rungs (below e4b)
# qwen cap-32 = 3.40 GB and cap-48 = 4.46 GB measured; e4b = 3.91 GB.
for C in 32 48; do
  step dom-qwen-gsm8k-c$C $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity $C --policy exact --n 200
  step dom-qwen-popqa-c$C $QV -m knowledge.frontier acc --model qwen --task popqa --capacity $C --policy exact --n 200
done
# gemma cap-24 = 3.76 GB, also below e4b
step dom-gemma-gsm8k-c24 $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity 24 --policy exact --n 200
step dom-gemma-popqa-c24 $QV -m knowledge.frontier acc --model gemma --task popqa --capacity 24 --policy exact --n 200

# --------------------------------------- 2. mmlu_pro — the widest capability gap
# The task where the small model falls furthest behind (measured separately: gemma-26b 0.530
# vs e4b 0.285), so it is where "use the big model, offloaded" should look best.
# n=150 not 200: cap4160 makes this the most expensive task in the suite, and
# R14 forbids shortening the cap to save time — truncation manufactures errors.
step mmlu-e4b $QV -m knowledge.frontier acc --model e4b --task mmlu_pro --policy none --n 150
step mmlu-gemma-c32 $QV -m knowledge.frontier acc --model gemma --task mmlu_pro --capacity 32 --policy exact --n 150
step mmlu-qwen-c64 $QV -m knowledge.frontier acc --model qwen --task mmlu_pro --capacity 64 --policy exact --n 150

step report $QV -m knowledge.frontier report

echo "### frontier chain 4 done $(date) ###"
