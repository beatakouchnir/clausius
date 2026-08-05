#!/bin/sh
# Detector generalisation, threshold calibration, and the terse-damage test.
#
# Everything so far is gsm8k, whose 960-token cap gives a broken model room to
# ramble. popqa caps at 64, so length carries far less information and a
# detector that was quietly riding on verbosity has nowhere to hide.
#
# EXACT arms are the null distribution: F3 established accuracy is flat across
# capacity, so each is a benign config and their d_z spread IS the false-alarm
# threshold — replacing the 0.5 that was eyeballed off six configs.
#
# STATIC arms are damage on a short-output task: the failure mode no test has
# covered, and the one that would be a silent false negative if entropy only
# detects damage that presents as uncertainty.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### regress3 start $(date) ###"

step p-qwen-ref $QV -m knowledge.regress capture --model qwen --task popqa --capacity 256 --policy exact --tag ref
for C in 192 128 96 64 48 32; do
  step p-qwen-exact-c$C $QV -m knowledge.regress capture --model qwen --task popqa --capacity $C --policy exact --tag exact_c$C
done
for C in 128 64; do
  step p-qwen-static-c$C $QV -m knowledge.regress capture --model qwen --task popqa --capacity $C --policy static --tag static_c$C
done
step p-qwen-analyse $QV -m knowledge.regress analyse --model qwen --task popqa

step p-gemma-ref $QV -m knowledge.regress capture --model gemma --task popqa --capacity 128 --policy exact --tag ref
for C in 96 64 48 32 24; do
  step p-gemma-exact-c$C $QV -m knowledge.regress capture --model gemma --task popqa --capacity $C --policy exact --tag exact_c$C
done
for C in 96 64 32; do
  step p-gemma-static-c$C $QV -m knowledge.regress capture --model gemma --task popqa --capacity $C --policy static --tag static_c$C
done
step p-gemma-analyse $QV -m knowledge.regress analyse --model gemma --task popqa

echo "### regress3 done $(date) ###"
