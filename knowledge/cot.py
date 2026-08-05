"""R14 — where do you measure entropy when the answer follows a reasoning chain?

R13 measured entropy at the LAST PROMPT TOKEN, which is the right place for
short-answer recall: the very next token is the answer, so the distribution
there is uncertainty about the answer itself. On a chain-of-thought task that
alignment breaks — the next token starts the *reasoning*, and the answer arrives
hundreds of tokens later.

Worse, there is a specific reason to expect the obvious fix to fail. Measuring
at the answer token means measuring after the chain is written, and by then the
model has committed: given a confident but WRONG chain, the final token is
nearly deterministic. That is exactly the dangerous case, so answer-token
entropy may be least informative precisely where it matters most.

So this compares every aggregation on identical generations:

  first     at the last prompt token — R13's measure, the baseline to beat
  answer    at the token emitting the final answer — the direct analogue
  mean/max  over generated tokens — "was the model uncertain anywhere?"
  min/last  completeness
  p90       90th-percentile token entropy — max is a single noisy token, p90 is
            the same idea with the outlier removed
  gen_len   generation length. A DUMB BASELINE that must be beaten: longer
            chains may simply be harder problems, and a signal that cannot beat
            "how much did it write" is not reading uncertainty at all.

ONE CAPTURE, MANY ANALYSES. The generation is expensive and the aggregation is
free, so the run records per-token entropy across the whole sequence and every
variant is computed offline (profile_experts.py's discipline).

CONTEXT FROM A PRIOR CALIBRATION STUDY. An n=740 study measured 8-vote
self-consistency at AURC **0.087 on math** and 0.259 on MCQ, so on this task the
incumbent is strong, not weak. Entropy does not need to beat it here — Stage B
does that comparison properly — but a variant that cannot even beat generation
length is not worth carrying into the long experiment.

Needs mlx-lm. Reuses the vendored GSM8K loader, prompt builder and scorer.

Usage:
  python3 -m knowledge.cot --capture --n 200
  python3 -m knowledge.cot
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT
from .popqa import task_suite

CAP = OUT / 'cot.capture.json'


def variants(ent, n_prompt, ans_pos):
    """Every aggregation of a per-token entropy sequence, computed offline."""
    gen = ent[n_prompt - 1:]
    if len(gen) == 0:
        gen = ent[-1:]
    return {
        'first': float(ent[n_prompt - 1]),
        'answer': float(ent[ans_pos]) if ans_pos is not None else float('nan'),
        'mean': float(np.mean(gen)),
        'max': float(np.max(gen)),
        'min': float(np.min(gen)),
        'last': float(gen[-1]),
        'p90': float(np.percentile(gen, 90)),
        'mean_top10': float(np.mean(np.sort(gen)[-max(1, len(gen) // 10):])),
        'gen_len': float(len(gen)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--task', default='gsm8k')
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.capture:
        return capture(a)

    d = json.loads(CAP.read_text())
    rows = d['rows']
    y = np.array([0 if r['correct'] else 1 for r in rows])
    print(f"{d['model']} · {d['task']} · {len(rows)} items · "
          f"accuracy {1 - y.mean():.3f} · {int(y.sum())} errors")
    if y.sum() < 10 or (1 - y).sum() < 10:
        print("  WARNING: one class is tiny; AUC will be unstable.")

    from .detect import auc
    keys = [k for k in rows[0]['ent'] if k != 'gen_len']
    print(f"\n  {'variant':12s} {'AUC vs error':>13s}  {'n valid':>8s}")
    res = {}
    for k in keys + ['gen_len']:
        v = np.array([r['ent'][k] for r in rows])
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            print(f"  {k:12s} {'—':>13s}  {int(ok.sum()):8d}")
            continue
        s = auc(v[ok & (y == 1)], v[ok & (y == 0)])
        res[k] = round(s, 4)
        tag = '  <-- DUMB BASELINE' if k == 'gen_len' else (
            '  <-- R13 measure' if k == 'first' else '')
        print(f"  {k:12s} {s:13.4f}  {int(ok.sum()):8d}{tag}")

    best = max((v, k) for k, v in res.items() if k != 'gen_len')
    print(f"\n  best variant: {best[1]} at {best[0]:.4f}")
    print(f"  dumb baseline (gen_len): {res.get('gen_len', float('nan')):.4f}")
    print(f"  R13's measure (first):   {res.get('first', float('nan')):.4f}")

    # the same AURC implementation, so the number is comparable with that n=740 study
    try:
        from . import _gl
        corr = 1 - y
        print(f"\n  {'variant':12s} {'AURC':>8s} {'E-AURC':>8s}   "
              f"(prior study: 8-vote self-consistency AURC 0.087 on math, "
              f"0.259 on MCQ)")
        # E-AURC subtracts the best achievable AURC at this base rate, so it is
        # the metric that compares fairly ACROSS tasks with different accuracy.
        for k in sorted(res, key=lambda k: res[k], reverse=True)[:4]:
            v = np.array([r['ent'][k] for r in rows])
            ok = np.isfinite(v)
            rc = _gl.risk_coverage(-v[ok], corr[ok])
            print(f"  {k:12s} {rc['aurc']:8.4f} {rc['e_aurc']:8.4f}")
    except Exception as e:
        print(f"\n  (calibration metrics unavailable: {e})")

    dest = OUT / 'cot.json'
    dest.write_text(json.dumps({'task': d['task'], 'model': d['model'],
                                'n': len(rows), 'accuracy': float(1 - y.mean()),
                                'auc': res}, indent=1))
    print(f"\n  → {dest}")


def find_answer_pos(tok, ids, n_prompt, text):
    """Index of the position that EMITS the final answer token.

    the vendored number scorer takes the last number after an `ANSWER:` marker,
    so the answer is located by finding that number's character offset in the
    generated text and mapping it back to a token — the same character-offset
    approach that fixed the multi-token failures in generated.py.
    """
    m = re.findall(r'ANSWER:\s*([^\n]+)', text)
    tail = m[-1] if m else text.strip()[-40:]
    nums = re.findall(r'-?\d[\d,]*\.?\d*', tail.replace(',', ''))
    if not nums:
        return None
    want = nums[-1].rstrip('.')
    at = text.rfind(want)
    if at < 0:
        return None
    acc, pos = '', None
    for i in range(n_prompt, len(ids)):
        piece = tok.decode([ids[i]])
        if len(acc) + len(piece) > at:
            pos = i - 1
            break
        acc += piece
    return pos


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate

    suite = task_suite()
    items = suite.load_items(a.task, a.n, a.seed)
    print(f"{len(items)} {a.task} items", flush=True)
    print("loading …", flush=True)
    model, tok = load(a.model)

    out = []
    for i, it in enumerate(items):
        pr = suite.build_prompt(tok, it, think=False)
        text = generate(model, tok, prompt=pr,
                        max_tokens=it.get('max_tokens', 512), verbose=False)
        ok = bool(suite.score(it, text))
        ids = tok.encode(pr + text)
        n_prompt = len(tok.encode(pr))

        # one teacher-forced pass over prompt+generation gives per-token
        # entropy everywhere; generation itself does not expose it
        lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        pv = mx.exp(lp)
        ent = np.asarray((-mx.sum(pv * lp, axis=-1))[0].tolist())

        ap_ = find_answer_pos(tok, ids, n_prompt, text)
        if ap_ is not None:
            ap_ = min(ap_, len(ent) - 1)
        out.append({'correct': ok, 'n_prompt': n_prompt,
                    'n_gen': len(ids) - n_prompt,
                    'ent': variants(ent, n_prompt, ap_),
                    'answer_pos_found': ap_ is not None})
        if (i + 1) % 25 == 0:
            acc = np.mean([r['correct'] for r in out])
            print(f"  {i + 1}/{len(items)}  accuracy {acc:.3f}", flush=True)

    acc = float(np.mean([r['correct'] for r in out]))
    found = float(np.mean([r['answer_pos_found'] for r in out]))
    CAP.write_text(json.dumps({'model': a.model.rstrip('/').split('/')[-1],
                               'task': a.task, 'accuracy': acc,
                               'rows': out}))
    print(f"\n  {len(out)} items · accuracy {acc:.3f} · "
          f"answer position located in {found:.1%} → {CAP}")


if __name__ == '__main__':
    main()
