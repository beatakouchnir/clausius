#!/bin/sh
# Tier-1 entropy applications: RAG utilisation, quantization equivalence,
# forgetting. All three turned out to need no new artifacts — the quantization
# ladder and the LoRA adapters were already on disk.
#
# Ordered by confidence: item 3 is cheap and self-verifying (its full run is
# also its smoke test), item 2 is the highest-value, item 1 depends on which
# base the adapters were trained against, which could not be checked.
#
# BOUNDED WAIT on chain 4 (~6.7h ceiling), never a pgrep waiter.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1

i=0
while [ $i -lt 1200 ]; do
  grep -q "regress4 done" records/regress4.log 2>/dev/null && break
  i=$((i + 1)); sleep 20
done
grep -q "regress4 done" records/regress4.log 2>/dev/null || \
  echo "### chain 4 unfinished after ~6.7h — proceeding, GPU may be contended ###"

step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
skip() { echo "=== SKIP $1 — $2 ==="; }

echo "### tier1 start $(date) ###"

# ------------------------------------------ REPAIR: the four arms chain 4 lost
# Both crashes were bugs in this repo's own code, now fixed:
#   gemma topk1 — `[..., -k:-k+1]` degenerates to the EMPTY slice `[..., -1:0]`
#                 at k==1. The scores branch was guarded for this; the tuple
#                 branch was not.
#   qwen topk1/2 — the model generates NOTHING at top-k<=2, so `variants` read
#                 `ent[n_prompt-1]` past the end of an array with no generated
#                 positions. Now clamped, and the row is kept with `empty_gen`
#                 set, because an immediate stop is a damage observation rather
#                 than a reason to drop the item and bias the sample.
# top-k 1 and 2 are the most extreme well-formed-computation arms available, so
# they matter to F8c: they are the cheapest remaining shot at confidently-wrong
# damage.
for K in 1 2; do
  step repair-topk$K-qwen  $QV -m knowledge.regress capture --model qwen  --task popqa --capacity 256 --policy exact --topk $K --tag topk$K
  step repair-topk$K-gemma $QV -m knowledge.regress capture --model gemma --task popqa --capacity 128 --policy exact --topk $K --tag topk$K
done
step repair-analyse-qwen  $QV -m knowledge.regress analyse --model qwen  --task popqa
step repair-analyse-gemma $QV -m knowledge.regress analyse --model gemma --task popqa

# ---------------------------------------------------------------- ITEM 3: RAG
# ~315 probes x 3 classes, short prompts — a few minutes per model, so the full
# run doubles as the smoke test for brand-new code. If the tokenisation guard in
# answer_stats rejects everything, the capture will report far fewer rows than
# probes and the analysis will say "no complete triples" rather than lying.
for M in qwen gemma; do
  step ctx-$M $QV -m knowledge.context --model $M
done
step ctx-analyse $QV -m knowledge.context --analyse

# ------------------------------------------ ITEM 2: the quantization ladder
# Same model, same pipeline, five bit-widths. Tests the literature claim
# directly — "quantized variants do not reliably reproduce base-model behavior,
# even when accuracy or perplexity appears preserved", drift growing as
# bit-width falls — and gives F8 a THIRD damage mechanism, independent of
# expert-zeroing and top-k. A dose-response curve is far stronger than a pair:
# d_z should ORDER with bit-width if the detector is measuring damage.
#
# --policy none throughout, so quantization is the only variable; mixing offload
# into one arm would confound it with F2's capacity-dependent numerics.
#
# NOTE when reading the analysis: these tags share the gemma/<task> namespace
# with the offload arms from chains 3-4, so the table will also list those
# against the quant_4bit reference. Their Δacc is still computed correctly; only
# the quant_* rows belong to this experiment.
A=${QUANTIZE_REPO:-../quantize}/artifacts
for T in popqa gsm8k; do
  step quant-4bit-$T $QV -m knowledge.regress capture --model gemma --task $T --policy none --model-path $A/26b-a4b-4bit-g64 --tag quant_4bit
  for Q in 3bit-g64 2bit-g64 mixed_4_6 mixed_3_6; do
    if [ -r "$A/26b-a4b-$Q/config.json" ]; then
      step quant-$Q-$T $QV -m knowledge.regress capture --model gemma --task $T --policy none --model-path $A/26b-a4b-$Q --tag quant_$Q
    else
      skip quant-$Q-$T "unreadable at $A/26b-a4b-$Q"
    fi
  done
  step quant-analyse-$T $QV -m knowledge.regress analyse --model gemma --task $T --ref quant_4bit
done

# --------------------------------------------------- ITEM 1: forgetting monitor
# Base vs LoRA checkpoint on held-out domains the adapter never trained on
# (gsm8k, popqa — the corpus was synthetic entity facts). Reference is the
# adapter-free base; ground truth is accuracy on the same items, which the
# detector never sees.
#
# Which base the adapters were trained against could not be verified, so qwen is
# tried first and a mismatch will surface as a load error in the log rather than
# silently producing numbers.
ARMS=records/corpus/arms
step forget-base-popqa $QV -m knowledge.regress capture --model qwen --task popqa --capacity 256 --policy exact --tag forget_base
for AD in adapter-qa adapter-router adapter-attn; do
  if [ -d "$ARMS/$AD" ]; then
    step forget-$AD-popqa $QV -m knowledge.regress capture --model qwen --task popqa --capacity 256 --policy exact --adapter $ARMS/$AD --tag forget_$AD
  else
    skip forget-$AD "not found at $ARMS/$AD"
  fi
done
step forget-analyse $QV -m knowledge.regress analyse --model qwen --task popqa --ref forget_base

echo "### tier1 done $(date) ###"
