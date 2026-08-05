"""clausius — did the thing you changed break your model?

Named for Rudolf Clausius, who coined the word *entropy* in 1865 — building it
from the Greek `trope` (transformation) and shaping it deliberately to echo
"energy", so the two would sound like the related quantities they are. Entropy
is the signal this library reads.

    from clausius import capture, compare

    ref  = capture("mlx-community/gemma-4-e4b-it-4bit", prompts, tag="v1")
    cand = capture("mlx-community/gemma-4-e4b-it-4bit", prompts, tag="v2",
                   adapter="./my-lora")
    print(compare(ref, cand))
    # REGRESSION  (max d_z = +0.782, threshold 0.3, one-sided)

`prompts` are your own, unlabelled. No gold answers and no judge model: the
comparison is between the model's own per-token entropy distributions on
identical inputs.

See `core` for why each default is what it is — every one of them is a measured
result rather than a taste, and FINDINGS.md has the evidence.
"""
from .core import (CAP_LADDER, DEFAULT_SIGNAL, DEFAULT_THRESHOLD,
                   MIN_PAIRED_ITEMS, SIGNALS, Capture, Result, TruncationCurve,
                   aggregate, capture, compare, top_movers, truncation_curve)

__all__ = ['capture', 'compare', 'Capture', 'Result', 'aggregate',
           'truncation_curve', 'TruncationCurve', 'top_movers',
           'SIGNALS', 'DEFAULT_SIGNAL', 'DEFAULT_THRESHOLD',
           'MIN_PAIRED_ITEMS', 'CAP_LADDER']
__version__ = '0.1.0'
