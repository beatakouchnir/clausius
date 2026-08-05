#!/bin/sh
# F10 redesigned. Three faults fixed: PopQA long-tail items (something to
# detect), chat-templated prompts via quantize's build_prompt (in-distribution),
# and entropy over the model's OWN generated span rather than one token (the
# likely cause of gemma's chance-level per-item agreement).
#
# Baseline is `irrelevant`, not `nocontext` — it is format-matched to `relevant`
# so only content differs, which is the confound that muddied attempt 2.
QV=${QV:-python}
cd "$(dirname "$0")/.." || exit 1
export QUANTIZE_REPO=${QUANTIZE_REPO:-../quantize}
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]\|Repo card"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### F10 start $(date) ###"
for M in qwen gemma; do
  step f10-$M $QV -m knowledge.retrieval --model $M --n 300
done
step f10-analyse $QV -m knowledge.retrieval --analyse
echo "### F10 done $(date) ###"
