#!/bin/sh
# Item 3, corrected. The grounding-suite version failed for a design reason, not
# a signal reason: its facts (capital of Australia, etc.) are ones qwen already
# knows, so handing it the answer changed answer-NLL by +0.15 nats — there was
# no retrieval benefit for entropy to detect — and the "Fact:" frame moved
# entropy more than the content did (+0.68 vs +0.38).
#
# PopQA's long tail fixes that: accuracy is 0.23-0.29, so most items are ones
# the model CANNOT answer from weights and context has something to add.
#
# And it adds the control the first version lacked: split on parametric NLL, a
# label-free proxy for "did the model already know this". Context can only help
# where it did not, so pooling both kinds dilutes the effect to nothing — which
# is precisely how the first run failed.
#
# BOUNDED WAIT on tier1 (~6.7h ceiling), never a pgrep waiter.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
i=0
while [ $i -lt 1200 ]; do
  grep -q "tier1 done" records/tier1.log 2>/dev/null && break
  i=$((i + 1)); sleep 20
done
grep -q "tier1 done" records/tier1.log 2>/dev/null || \
  echo "### tier1 unfinished after ~6.7h — proceeding, GPU may be contended ###"

step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### tier1b start $(date) ###"
for M in qwen gemma; do
  step ctx-popqa-$M $QV -m knowledge.context --model $M --suite popqa --n 300
done
step ctx-popqa-analyse $QV -m knowledge.context --analyse --suite popqa
echo "### tier1b done $(date) ###"
