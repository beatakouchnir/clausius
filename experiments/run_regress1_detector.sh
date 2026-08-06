#!/bin/sh
# Label-free regression detection. Each arm's TRUE damage is already known from
# the frontier chains; the detector never sees it.
#
# Order: reference first (analysis needs it), then the BENIGN control, which is
# the arm that decides whether this is a damage detector or merely a change
# detector. Broken arms last — if the benign control already fires, the broken
# ones prove nothing and the log shows that early.
QV=${QV:-python}          # interpreter with mlx-lm available
cd "$(dirname "$0")/.." || exit 1
step() { echo "=== STEP $1 $(date +%H:%M) ==="; shift; "$@" 2>&1 | grep -v "examples/s\]"; echo "=== DONE $(date +%H:%M) ==="; }
echo "### regress chain start $(date) ###"

# qwen: reference, benign control, two broken, one mild
step reg-qwen-ref        $QV -m knowledge.regress capture --model qwen --capacity 256 --policy exact --tag ref
step reg-qwen-benign     $QV -m knowledge.regress capture --model qwen --capacity 64  --policy exact --tag exact_c64
step reg-qwen-static128  $QV -m knowledge.regress capture --model qwen --capacity 128 --policy static --tag static_c128
step reg-qwen-static64   $QV -m knowledge.regress capture --model qwen --capacity 64  --policy static --tag static_c64
step reg-qwen-topk4      $QV -m knowledge.regress capture --model qwen --capacity 256 --policy exact --topk 4 --tag topk4
step reg-qwen-analyze    $QV -m knowledge.regress analyze --model qwen

# gemma: architecture independence
step reg-gemma-ref       $QV -m knowledge.regress capture --model gemma --capacity 128 --policy exact --tag ref
step reg-gemma-benign    $QV -m knowledge.regress capture --model gemma --capacity 32  --policy exact --tag exact_c32
step reg-gemma-static32  $QV -m knowledge.regress capture --model gemma --capacity 32  --policy static --tag static_c32
step reg-gemma-analyze   $QV -m knowledge.regress analyze --model gemma

echo "### regress chain done $(date) ###"
