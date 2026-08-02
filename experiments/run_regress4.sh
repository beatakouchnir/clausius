#!/bin/sh
# Adversarial construction: can a config be damaged while getting MORE confident?
# "Ruled out" is not reachable — it is an existential claim over all configs —
# so the substitute is a deliberate attempt to build the counterexample.
#
# SUBSTITUTION routes non-resident experts to a real resident expert instead of
# a zero slot: well-formed arithmetic, wrong weights. If damage can present as
# confidence, this is its shape.
# TOP-K 1/2 leaves the computation entirely well-formed — real expert, real
# magnitudes, less mixture — and is a knob real runtimes expose.
# SHARPEN is the specificity control: greedy output verified bit-identical,
# entropy collapsed. It tests the MIRROR failure — does the detector fire on a
# confidence change that is provably not damage?
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### regress4 start $(date) ###"

# cheap task first: popqa gives the answer in minutes per arm
for M in qwen gemma; do
  [ "$M" = qwen ] && C=64 || C=32
  step sub-$M-popqa   $QV -m knowledge.regress capture --model $M --task popqa --capacity $C --policy static --substitute --tag subst_c$C
  step topk1-$M-popqa $QV -m knowledge.regress capture --model $M --task popqa --capacity 256 --policy exact --topk 1 --tag topk1
  step topk2-$M-popqa $QV -m knowledge.regress capture --model $M --task popqa --capacity 256 --policy exact --topk 2 --tag topk2
  step sharp2-$M      $QV -m knowledge.regress capture --model $M --task popqa --capacity 256 --policy exact --sharpen 2.0 --tag sharpen2
  step sharp3-$M      $QV -m knowledge.regress capture --model $M --task popqa --capacity 256 --policy exact --sharpen 3.0 --tag sharpen3
  step analyse-$M-popqa $QV -m knowledge.regress analyse --model $M --task popqa
done

# then gsm8k, where the detector is strongest — does substitution still evade it?
for M in qwen gemma; do
  [ "$M" = qwen ] && C=64 || C=32
  step sub-$M-gsm8k   $QV -m knowledge.regress capture --model $M --task gsm8k --capacity $C --policy static --substitute --tag subst_c$C
  step topk1-$M-gsm8k $QV -m knowledge.regress capture --model $M --task gsm8k --capacity 256 --policy exact --topk 1 --tag topk1
  step analyse-$M-gsm8k $QV -m knowledge.regress analyse --model $M --task gsm8k
done

echo "### regress4 done $(date) ###"
