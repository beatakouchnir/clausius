#!/bin/sh
# Chain 3. Closes the two measurement gaps, adds the reference model, and puts a
# number on SwiftLM's top-k knob.
#
# Ordering is deliberate: the two gaps the user explicitly asked for run FIRST
# and unconditionally. The top-k arm is new code, so its wiring control runs
# before its accuracy runs — but after the gaps, so a bug there cannot cost the
# requested work.
#
# No waiters at all: chains 1 and 2 are both finished. Sequential, exits.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1

step() {
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE $(date +%H:%M) ==="
}

echo "### frontier chain 3 start $(date) ###"

# ------------------------------------------------- GAP 1: memory, now measured
# The first sweep's peak_gb was a high-water mark taken from process start, so
# it captured the full model load BEFORE wrap() freed the experts and every
# capacity reported the same number. load_wrapped now calls reset_peak_memory()
# after wrapping and records get_active_memory(), so these rows finally measure
# what a configuration costs to RUN. tok/s is re-measured in the same pass.
for C in 128 96 64 48 32 24; do
  step mem-gemma-c$C $QV -m knowledge.frontier speed --model gemma --capacity $C --policy exact
done
for C in 256 192 128 96 64 48 32; do
  step mem-qwen-c$C $QV -m knowledge.frontier speed --model qwen --capacity $C --policy exact
done

# ------------------------------- GAP 2: exact-policy noise floor on gsm8k
# The floor was measured on popqa only — 64-token caps, the least room for the
# capacity-dependent rounding to compound. gsm8k generates ~250 tokens, so if
# the floor is task-dependent this is where it shows. Also turns the ~0.91 in the
# matched-memory table from inferred into measured.
for C in 128 64 32; do
  step floor-gemma-gsm8k-c$C $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity $C --policy exact --n 200
done
for C in 256 128 64; do
  step floor-qwen-gsm8k-c$C $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity $C --policy exact --n 200
done

# --------------------------------------------- REFERENCE: e4b through this code
# The frontier's small-model comparison point has been quoted from the earlier
# harness (gsm8k 0.785, popqa 0.135). Measuring it through THIS scorer removes a
# cross-harness assumption from the headline claim. e4b is not MoE, so only
# policy=none works — wrap() would correctly refuse it.
step ref-e4b-gsm8k $QV -m knowledge.frontier acc --model e4b --task gsm8k --policy none --n 200
step ref-e4b-popqa $QV -m knowledge.frontier acc --model e4b --task popqa --policy none --n 200

# ------------------------------------ TOP-K WIRING CONTROL (before the payload)
# cut_topk at the NATIVE k=8 must be a no-op. Compared against chain 1's
# `gen.gemma.full.json`, which is the same capacity and policy with no top-k
# wrapper, so identical output proves the mask arithmetic is right. If this
# fails, every topk accuracy number below is meaningless — and the log will say
# so before the user reads the numbers.
step topk-control-gen $QV -m knowledge.frontier gen --model gemma --capacity 128 --policy exact --topk 8 --tag k8
step topk-control-cmp $QV -m knowledge.frontier compare --model gemma --a full --b k8

# ------------------------------------------------- TOP-K ACCURACY (the payload)
# Run at FULL capacity so offload is out of the picture and top-k is the only
# variable. NOTE: tok/s from these runs is NOT comparable to SwiftLM's — this
# implementation masks gate weights while SwitchGLU still gathers all 8 experts,
# so it isolates the ACCURACY cost of the knob and gains none of its speed.
for K in 6 4; do
  step topk-gemma-gsm8k-k$K $QV -m knowledge.frontier acc --model gemma --task gsm8k --capacity 128 --policy exact --topk $K --n 200
  step topk-gemma-popqa-k$K $QV -m knowledge.frontier acc --model gemma --task popqa --capacity 128 --policy exact --topk $K --n 200
done
for K in 6 4; do
  step topk-qwen-gsm8k-k$K $QV -m knowledge.frontier acc --model qwen --task gsm8k --capacity 256 --policy exact --topk $K --n 200
  step topk-qwen-popqa-k$K $QV -m knowledge.frontier acc --model qwen --task popqa --capacity 256 --policy exact --topk $K --n 200
done

# ------------------------------------------------------------------- assemble
step report $QV -m knowledge.frontier report

echo "### frontier chain 3 done $(date) ###"
