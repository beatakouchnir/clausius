#!/bin/sh
# Chain 5. Re-runs the gemma top-k arm after the ranking bug.
#
# gemma4_text's Router selects experts with argpartition, which fixes membership
# but NOT order, so masking by position dropped an arbitrary 2 of 8 rather than
# the 2 weakest. The invalid records are overwritten in place — same filenames,
# so the report self-heals rather than mixing old and new.
#
# A STRONGER CONTROL runs first this time. The old k=8 no-op could not fail on a
# ranking bug, because keeping every position is order-independent. Comparing
# k=7 against k=7 across two processes is also order-independent and equally
# useless. What discriminates is whether the k-th expert dropped is the WEAKEST:
# with correct ranking, k=7 removes the smallest of 8 weights and must perturb
# output far less than the old code's arbitrary drop did. So the control is
# k=7 accuracy on gsm8k — near 0.9146 means ranking works; near 0.10 means it
# is still dropping randomly and the k6/k4 numbers below are again meaningless.
#
# BOUNDED WAIT on chain 4 (1200 x 20s ~= 6.7h), never a pgrep waiter.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
LOG=records/frontier4.log

i=0
while [ $i -lt 1200 ]; do
  grep -q "frontier chain 4 done" "$LOG" && break
  i=$((i + 1))
  sleep 20
done
grep -q "frontier chain 4 done" "$LOG" || {
  echo "### chain 4 never finished — chain 5 aborting ###"; exit 1; }

step() {
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE $(date +%H:%M) ==="
}

echo "### frontier chain 5 start $(date) ###"

# control: drop only the single weakest expert
step topk-gemma-control-k7 $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity 128 --policy exact --topk 7 --n 200

# payload, overwriting the invalid records
for K in 6 4; do
  step topk-gemma-gsm8k-k$K $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity 128 --policy exact --topk $K --n 200
  step topk-gemma-popqa-k$K $QV -m knowledge.frontier acc --model gemma --task popqa --capacity 128 --policy exact --topk $K --n 200
done

# qwen k=7 for a matched comparison against gemma's control
step topk-qwen-control-k7 $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity 256 --policy exact --topk 7 --n 200

step report $QV -m knowledge.frontier report

echo "### frontier chain 5 done $(date) ###"
