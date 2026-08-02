"""Read quantize's captured routing traces. Stdlib only, no model, no GPU.

quantize's profiler was built on "CAPTURE ONCE, ANALYSE FOREVER": a GPU run
writes a trace, and every analysis afterwards runs from it with no model load.
This project inherits that discipline for a harder reason than convenience —
loading a 19 GB MoE perturbs anything else running on the machine. MLX uses
unified memory, so pinning to `mx.cpu` does NOT isolate you: same RAM pool,
same memory bus. The axis that matters is loads-a-model vs doesn't, and
everything in this module is on the safe side of it.

Two trace kinds, both written by quantize, both domain-labelled:

  expert-trace/1   which experts were SELECTED, per token   (gemma + qwen)
  gate-trace/1     the top-32 experts by GATE SCORE, ranked (qwen only)

Only DECODE records are usable. A prefill record covers many tokens at once
(`r` > 1) and its expert list is the union over all of them — comparing that
against a per-token set would be comparing two different measurements.
Batched records are excluded for the same reason.

THE UNIT OF INDEPENDENCE IS THE PROMPT, NOT THE TOKEN. Each domain is eight
prompts generated to ~120 tokens, so a trace holds ~900 decode records but only
**8 independent samples**. Tokens inside one generation share a topic and route
alike; treating them as independent inflates any significance test by two
orders of magnitude. `segment()` recovers the prompt boundaries exactly rather
than assuming a fixed block size: the capture loop increments its pass counter
on prefill too, so consecutive prompts leave a gap of 2 in the pass ids while
tokens within a generation are contiguous.

Point at a quantize checkout other than ../quantize with QUANTIZE_REPO.
"""
import gzip
import json
import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent.parent / 'quantize'


def repo():
    return Path(os.environ.get('QUANTIZE_REPO', _DEFAULT))


def artifact(name=''):
    """Path to a checkpoint under the quantize checkout's artifacts/.

    Exists so no module hardcodes an absolute path to somebody's home
    directory. Override the checkout with QUANTIZE_REPO; these defaults only
    say which checkpoint an experiment used, and are not expected to resolve on
    a machine that does not have it.
    """
    return str(repo() / 'artifacts' / name)


def records_dir():
    d = repo() / 'records'
    if not d.is_dir():
        raise SystemExit(f"no quantize records at {d}; set QUANTIZE_REPO")
    return d


# The traces this project reads, and the ablation record each one pairs with.
# The pairing is checkpoint-sensitive and easy to get wrong: quantize holds
# BOTH a base and an instruct ablation for gemma, and the gemma trace was
# captured on the INSTRUCT checkpoint. W5.2 found layer identity does not
# transfer base->instruct (top-4 overlap 1/4), so pairing the it-trace with
# the base ablation would silently compare two different models.
SOURCES = {
    'qwen': {
        'trace': 'expert_trace.qwen36-35b-a3b-4bit-g64.jsonl.gz',
        'gate': 'gate.qwen36-35b-a3b-4bit-g64.jsonl.gz',
        'ablation': 'ablate_layers.qwen36-35b-a3b-4bit-g64.json',
        'ablation_kind': 'layers',
        'label': 'qwen3.6-35b-a3b (instruct, 4bit-g64)',
    },
    'gemma': {
        'trace': 'expert_trace.jsonl.gz',
        'gate': None,
        'ablation': 'ablation.json',          # chat/instruct — matches the trace
        'ablation_kind': 'branch',
        'label': 'gemma-4-26b-a4b (instruct, 4bit-g64)',
    },
}


def _lines(path):
    with gzip.open(path, 'rt') as f:
        json.loads(f.readline())
        for line in f:
            yield json.loads(line)


def load(path, kind):
    """(meta, {domain: [prompt, ...]}), prompt = {layer: [expert-tuple/token]}.

    Layers within a prompt are aligned by construction: a pass is kept only if
    every instrumented layer recorded it, so prompt[l][i] and prompt[l'][i] are
    the same decode step.
    """
    with gzip.open(path, 'rt') as f:
        meta = json.loads(f.readline())
    n_layers = meta['n_layers']
    by = {}
    for r in _lines(path):
        if kind == 'expert' and (r.get('r') != 1 or r.get('b', 1) != 1):
            continue                           # decode, unbatched only
        by.setdefault(r['d'], {}).setdefault(r['p'], {})[r['l']] = tuple(r['e'])

    out = {}
    for d, passes in by.items():
        keep = sorted(p for p, ls in passes.items() if len(ls) == n_layers)
        out[d] = [{l: [passes[p][l] for p in seg] for l in range(n_layers)}
                  for seg in segment(keep)]
    return meta, out


def segment(passes):
    """Split an ascending pass-id list into per-prompt runs.

    Contiguous ids are one generation; any gap starts a new prompt. Verified
    against the capture: eight prompts per domain, seven gaps, every gap of
    size 2 (the skipped id is that prompt's prefill).
    """
    if not passes:
        return []
    segs, cur = [], [passes[0]]
    for a, b in zip(passes, passes[1:]):
        if b - a == 1:
            cur.append(b)
        else:
            segs.append(cur)
            cur = [b]
    segs.append(cur)
    return segs
