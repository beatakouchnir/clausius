#!/bin/sh
# Resolve the topk4 ambiguity. Entropy fired (+0.56) but paired McNemar at
# n=200 gave 8 vs 3 discordant, p=0.227 — direction consistent with mild damage,
# not significant. Two readings with opposite product implications:
#   TRUE POSITIVE  -> entropy resolves damage that accuracy at n=200 cannot,
#                     because a continuous per-token signal is more efficient
#                     than a binary per-item one. That is the selling point.
#   FALSE POSITIVE -> entropy has a false-alarm rate on benign-but-perturbing
#                     configs. That is a limitation the product must disclose.
# Only more items decide it. Accuracy-only (no entropy pass) since the question
# is purely about ground truth.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
i=0
while [ $i -lt 1200 ]; do
  grep -q "regress chain done" records/regress.log && break
  i=$((i + 1)); sleep 20
done
grep -q "regress chain done" records/regress.log || { echo "### regress chain never finished — aborting ###"; exit 1; }
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### regress2 start $(date) ###"
step topk-resolve-ref  $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity 256 --policy exact --n 800
step topk-resolve-k4   $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity 256 --policy exact --topk 4 --n 800
echo "### regress2 done $(date) ###"
