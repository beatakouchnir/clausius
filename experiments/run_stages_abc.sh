#!/bin/sh
# Unattended chain: finish Stage A, then Stage B, then Stage C.
# Each step is isolated so one failure cannot stop the chain — the log is the
# record and every step brackets itself with STEP/DONE markers.
QV=${QV:-python}          # interpreter with mlx-lm available
Q27=${QWEN_DENSE:-mlx-community/Qwen3.6-27B-4bit}
QMOE=${CLAUSIUS_ARTIFACTS:-../artifacts}/qwen36-35b-a3b-4bit-g64
GMOE=${GEMMA_MOE:?set GEMMA_MOE to a gemma-4-26b-a4b checkpoint}
G31=${GEMMA_DENSE:?set GEMMA_DENSE to a gemma-4-31b checkpoint}
GE4B=mlx-community/gemma-4-e4b-it-4bit

step() {                     # step <label> <command...>
  echo "=== STEP $1 $(date +%H:%M) ==="
  shift
  "$@" 2>&1 | grep -v "examples/s\]"
  echo "=== DONE rc=$? $(date +%H:%M) ==="
}

# wait for the in-flight gpqa(cap4160) run to finish before touching the GPU
while pgrep -f "stage_a --task gpqa" >/dev/null; do sleep 30; done

# ---- Stage A completion -------------------------------------------------
step "A/gpqa-cap8192" $QV -m knowledge.stage_a --task gpqa --n 198 --cap 8192
step "A/mmlu_pro-800" $QV -m knowledge.stage_a --task mmlu_pro --n 800
step "A/analyse"      $QV -m knowledge.stage_a --analyse

# ---- Stage B: entropy vs self-consistency -------------------------------
# popqa is cheap (64 tok) so k=5 is affordable at n=200; mmlu_pro at cap4160
# costs ~6x per item, so n is cut to 120 to keep the arm inside the night.
step "B/popqa"    $QV -m knowledge.stage_b --task popqa      --n 200 --k 5
step "B/omni"     $QV -m knowledge.stage_b --task omniscience --n 200 --k 5
step "B/mmlu_pro" $QV -m knowledge.stage_b --task mmlu_pro   --n 120 --k 5
step "B/analyse"  $QV -m knowledge.stage_b --analyse

# ---- Stage C: architecture portability -----------------------------------
# Two tasks per model, both cheap recall sets, so five models fit. The matched
# pair (QMOE vs Q27) is the clean architecture test: same family, same 4-bit
# quantization, differing only in MoE vs dense.
for M in "$Q27" "$GMOE" "$G31" "$GE4B"; do
  step "C/popqa/$M"  $QV -m knowledge.stage_a --task popqa       --n 250 --model "$M"
  step "C/omni/$M"   $QV -m knowledge.stage_a --task omniscience --n 300 --model "$M"
done
step "C/analyse" $QV -m knowledge.stage_a --analyse

echo "=== OVERNIGHT-COMPLETE $(date +%H:%M) ==="
