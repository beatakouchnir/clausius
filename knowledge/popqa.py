"""R13 — error prediction on PopQA, a benchmark with a real error rate.

The previous attempt authored its own question set and produced ZERO model
errors: qwen answered all 44 factual questions correctly and refused all 16
invented entities rather than fabricating. Two apparent errors were both bugs in
my own scoring. With no errors there is nothing to predict, and hand-authoring
harder questions would make ME the ground-truth bottleneck — on genuinely hard
facts I am less reliable too, and mislabelling a correct answer as an error is
worse than having no data.

PopQA solves both problems. It is long-tail entity QA — the shape of question a
user actually asks — and it was measured separately: **0.29 on qwen, 0.225 on
gemma**, i.e. a 71-77% error rate, with curated alias sets as ground truth.

REUSED FROM the vendored `_vendor/suite.py`, not reimplemented: the dataset and split, the
prompt and instruction wording, and the alias scorer. Matching that harness is
what makes the error rate here comparable to the numbers already in
records/suite.json. the vendored code is read-only here.; nothing is written
back.

WHAT IS COMPARED, all read at one forward pass over the prompt:

  entropy      the incumbent. One scalar, no router seam, works on dense models,
               and it beat routing outright in R6c.
  top1_prob    the other free scalar.
  routing      a classifier over selected experts, trained to predict error.

SPLIT BY QUESTION, and note the limitation: PopQA items are independent, so
there is no passage or entity grouping to leak through — but a subject entity
can recur across questions, so the split groups on the subject where one is
recoverable.

Needs mlx-lm and datasets.

Usage:
  python3 -m knowledge.popqa --capture --n 400
  python3 -m knowledge.popqa
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT

CAP = OUT / 'popqa.capture.json'


def task_suite():
    """The vendored task/scorer suite — the same code that produced
    records/suite.json, not a re-implementation that might disagree about what
    counts as correct. Vendored at `_vendor/suite.py`; see that file's header."""
    from ._vendor import suite
    return suite


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.capture:
        return capture(a)

    d = json.loads(CAP.read_text())
    rows = d['rows']
    y = np.array([0 if r['correct'] else 1 for r in rows])       # 1 = ERROR
    ent = np.array([r['entropy'] for r in rows])
    top1 = np.array([r['top1_prob'] for r in rows])
    X = np.array([r['experts'] for r in rows], dtype=np.int64)
    grp = np.array([r['group'] for r in rows])
    E = d['n_experts']

    print(f"{d['model']} · PopQA · {len(rows)} items · "
          f"accuracy {1 - y.mean():.3f} · **{int(y.sum())} errors "
          f"({y.mean():.1%})**")
    print(f"  (the recorded suite run gives popqa/qwen accuracy 0.29)\n")

    from .detect import auc
    from .meter import counts, score

    # grouped CV so a repeated subject cannot be memorised
    pred = np.zeros(len(y))
    folds = np.unique(grp)
    rng = np.random.default_rng(a.seed)
    assign = {g: i % 5 for i, g in enumerate(rng.permutation(folds))}
    fold = np.array([assign[g] for g in grp])
    for f in range(5):
        te, tr = fold == f, fold != f
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        C = counts(X[tr], y[tr], E, n_cls=2)
        sc = score(C, X[te])
        pred[te] = sc[:, 1] - sc[:, 0]

    res = {'n': len(y), 'n_errors': int(y.sum()),
           'accuracy': round(float(1 - y.mean()), 4),
           'entropy': round(auc(ent[y == 1], ent[y == 0]), 4),
           'neg_top1': round(auc(-top1[y == 1], -top1[y == 0]), 4),
           'routing': round(auc(pred[y == 1], pred[y == 0]), 4)}
    print(f"  {'signal':12s} {'AUC vs error':>13s}")
    for k in ('entropy', 'neg_top1', 'routing'):
        print(f"  {k:12s} {res[k]:13.4f}")
    best = max(res['entropy'], res['neg_top1'])
    print(f"\n  best scalar {best:.4f} · routing {res['routing']:.4f} "
          f"· lift {res['routing'] - best:+.4f}")
    res['lift'] = round(res['routing'] - best, 4)

    # combination, and the confident quadrant where entropy is blind
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)   # noqa: E731
    comb = z(pred) + z(ent)
    res['combined'] = round(auc(comb[y == 1], comb[y == 0]), 4)
    print(f"  combined (routing + entropy) {res['combined']:.4f}")

    conf = ent <= np.median(ent)
    print(f"\n  CONFIDENT half (entropy <= median) — where entropy is blind:")
    for lbl, m in (('confident', conf), ('uncertain', ~conf)):
        if len(np.unique(y[m])) < 2:
            continue
        e_ = auc(ent[m & (y == 1)], ent[m & (y == 0)])
        r_ = auc(pred[m & (y == 1)], pred[m & (y == 0)])
        print(f"    {lbl:10s} n={int(m.sum()):3d} errors {int(y[m].sum()):3d}"
              f"   entropy {e_:.3f}   routing {r_:.3f}")
        res[f'{lbl}_entropy'] = round(e_, 4)
        res[f'{lbl}_routing'] = round(r_, 4)

    dest = OUT / 'popqa.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  AUC 0.5 = chance. Higher = better at flagging a wrong answer.")
    print(f"\n  → {dest}")


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate
    from .seam import find_gates, gate_output, describe

    suite = task_suite()
    items = suite.load_items('popqa', a.n, a.seed)
    print(f"{len(items)} PopQA items from the vendored loader", flush=True)

    print("loading …", flush=True)
    model, tok = load(a.model)
    n_moe, E, _tk = describe(model)
    sink = {'on': False, 'rows': {}}
    restore = []
    for li, holder, name, gate in find_gates(model):
        th, tn, tg = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if inner is not None and callable(inner):
            th, tn, tg = gate, 'proj', inner

        class Tap:
            def __init__(self, inner, idx):
                self.inner, self.idx = inner, idx

            def __call__(self, x, *aa, **kw):
                out = self.inner(x, *aa, **kw)
                if sink['on']:
                    rk, _s, _k = gate_output(out, a.top_k)
                    mx.eval(rk)
                    sink['rows'][self.idx] = np.asarray(
                        rk.reshape(-1, rk.shape[-1]).tolist(), dtype=np.int64)
                return out

            def __getattr__(self, n):
                return getattr(object.__getattribute__(self, 'inner'), n)

        setattr(th, tn, Tap(tg, li))
        restore.append((th, tn, tg))

    out = []
    try:
        for i, it in enumerate(items):
            # the vendored prompt builder, with think=False to match the
            # runs already in records/suite.json. Rebuilding the prompt here
            # would risk a different accuracy than the 0.29 being compared to.
            pr = suite.build_prompt(tok, it, think=False)
            text = generate(model, tok, prompt=pr,
                            max_tokens=it.get('max_tokens', 48), verbose=False)
            ok = bool(suite.score(it, text))

            ids = tok.encode(pr)
            sink['rows'] = {}
            sink['on'] = True
            lg = model(mx.array([ids])).astype(mx.float32)
            sink['on'] = False
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            pv = mx.exp(lp)
            t = len(ids) - 1
            # group on the question's subject where recoverable, so a repeated
            # entity cannot straddle the CV split
            m = re.search(r"of ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*)*)",
                          it['prompt'])
            out.append({
                'question': it['prompt'][:120], 'answer': text.strip()[:60],
                'correct': ok, 'group': (m.group(1) if m else it['prompt'][:18]),
                'entropy': float(-mx.sum(pv[0, t] * lp[0, t])),
                'top1_prob': float(mx.max(pv[0, t])),
                'experts': [sink['rows'][l][t][:a.top_k].tolist()
                            for l in range(n_moe)]})
            if (i + 1) % 50 == 0:
                acc = np.mean([r['correct'] for r in out])
                print(f"  {i + 1}/{len(items)}  running accuracy {acc:.3f}",
                      flush=True)
    finally:
        for h, n, g in restore:
            setattr(h, n, g)

    acc = float(np.mean([r['correct'] for r in out]))
    CAP.write_text(json.dumps({'model': a.model.rstrip('/').split('/')[-1],
                               'n_layers': n_moe, 'n_experts': E,
                               'accuracy': acc, 'rows': out}))
    print(f"\n  {len(out)} items · accuracy {acc:.3f} · "
          f"{sum(1 for r in out if not r['correct'])} errors → {CAP}")


if __name__ == '__main__':
    main()
