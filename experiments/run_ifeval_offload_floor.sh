#!/bin/sh
# Does the accuracy floor (F3) hold on a LONG-OUTPUT task?
#
# F3 established "exact offload is accuracy-preserving down to 12% residency" on
# popqa and gsm8k. F11 then showed popqa is the LEAST sensitive benchmark
# available — 1.5pp where ifeval and gsm8k lose 18-21pp under quantization — so
# the floor claim now leans mostly on gsm8k alone. That matters because F5's
# dominance result is the strongest thing in the repo and it inherits F3.
#
# IFEval is the right third task: 768-token structured output, and it is the
# capability agents actually depend on. it was measured neutral at ONE
# offload point (0.850 -> 0.875); this sweeps the rungs.
#
# Tags are namespaced `off_*` — the previous run silently overwrote a saved
# reference by sharing the {model}.{task} namespace between experiments.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### ifeval offload ladder start $(date) ###"
step off-if-c128 $QV -m knowledge.regress capture --model gemma --task ifeval --capacity 128 --policy exact --tag off_ref --n 200
for C in 96 64 48 32 24; do
  step off-if-c$C $QV -m knowledge.regress capture --model gemma --task ifeval --capacity $C --policy exact --tag off_c$C --n 200
done
step off-if-analyze $QV -m knowledge.regress analyze --model gemma --task ifeval --ref off_ref
echo "### ifeval offload ladder done $(date) ###"
