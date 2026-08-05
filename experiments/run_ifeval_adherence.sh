#!/bin/sh
# Does compression degrade INSTRUCTION ADHERENCE faster than it degrades the
# factual/reasoning accuracy already measured?
#
# That is the agent-relevant axis. An agent depends on the model emitting a
# required format, using only permitted fields, and stopping where told —
# capabilities IFEval measures directly and MMLU-style accuracy does not. And
# because agentic errors compound (95% per step over 10 steps = 59% end to end),
# a per-step drop too small to matter on a benchmark is decisive over a task.
#
# The same five-point ladder as the popqa/gsm8k runs, so Δacc is directly
# comparable at matched bit-width — which is the whole point. Existing public
# tool-calling evals compare many models at ONE fixed quantization; existing
# quantization benchmarks measure MMLU and perplexity. Nobody sweeps compression
# against the capabilities agents actually run on.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
A=${CLAUSIUS_ARTIFACTS:-../artifacts}
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### ifeval sweep start $(date) ###"
step if-4bit $QV -m knowledge.regress capture --model gemma --task ifeval --policy none --model-path $A/26b-a4b-4bit-g64 --tag quant_4bit --n 200
for Q in mixed_4_6 mixed_3_6 3bit-g64 2bit-g64; do
  step if-$Q $QV -m knowledge.regress capture --model gemma --task ifeval --policy none --model-path $A/26b-a4b-$Q --tag quant_$Q --n 200
done
step if-analyse $QV -m knowledge.regress analyse --model gemma --task ifeval --ref quant_4bit
echo "### ifeval sweep done $(date) ###"
