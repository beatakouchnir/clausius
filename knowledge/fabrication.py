"""Does routing distinguish retrieval from FABRICATION?

The earlier captures could not ask this. They were built from facts the model
reliably knows — the computation suite produced 2 failures in 228 — so there
was nothing to predict. This suite elicits failure instead, through IDENTICAL
templates that vary only the entity:

  known      well-attested; the model should retrieve
  obscure    real but rarely attested; the model may or may not know
  fictional  invented; there is no fact, so a confident answer IS fabrication

The headline test is known vs fictional. Two things decide whether it means
anything:

  LEAVE ONE ENTITY OUT. Both wordings of "the capital of Verdania" are the same
  entity. Splitting them across train and test would let the classifier
  recognise the entity rather than the condition.

  THE `obscure` MIDDLE CLASS IS THE REAL CONTROL. A detector that merely fires
  on rare token sequences should group `obscure` with `fictional`, since both
  are unfamiliar strings. A detector that reads whether retrieval is SUCCEEDING
  should track whether the model actually got the obscure fact right. Which way
  `obscure` falls is the informative result, and it is why the class exists.

Needs numpy. No model, no GPU.

Usage:
  python3 -m knowledge.fabrication
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .meter import (load, featurise, counts, score, balanced_acc,
                    scalar_baseline, OUT)


def loo_by_entity(recs, meta, top_k, pos_cls, neg_cls):
    """Leave-one-entity-out predictions over a two-class subset."""
    sub = [r for r in recs if r['cls'] in (pos_cls, neg_cls)]
    X = featurise(sub, meta['n_layers'], top_k, 'answer')
    y = np.array([1 if r['cls'] == pos_cls else 0 for r in sub])
    ents = sorted({r['entity'] for r in sub})
    e = np.array([ents.index(r['entity']) for r in sub])
    C_all = counts(X, y, meta['n_experts'])
    pred = np.zeros(len(y), dtype=np.int64)
    for i in range(len(ents)):
        te = e == i
        C = C_all - counts(X[te], y[te], meta['n_experts'])
        if (C.sum(axis=(1, 2)) <= 0).any():
            C = C_all
        pred[te] = score(C, X[te]).argmax(axis=1)
    return sub, y, pred, X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trace', default=str(
        OUT / 'probe_gate.hallucination.qwen36-35b-a3b-4bit-g64.jsonl.gz'))
    ap.add_argument('--top-k', type=int, default=8)
    a = ap.parse_args()

    meta, recs = load(a.trace)
    print(f"{meta['model'].split('/')[-1]} · hallucination · {len(recs)} probes")
    for cls in ('known', 'obscure', 'fictional'):
        s = [r for r in recs if r['cls'] == cls]
        ok = sum(r['correct'] for r in s)
        note = ' (correctness meaningless — no true answer)' if cls == 'fictional' else ''
        print(f"  {cls:10s} n={len(s):3d}  answered correctly {ok}/{len(s)}{note}")

    res = {'trace': Path(a.trace).name, 'top_k': a.top_k}

    # ---- headline: known vs fictional --------------------------------------
    sub, y, pred, _ = loo_by_entity(recs, meta, a.top_k, 'known', 'fictional')
    acc = balanced_acc(y, pred)
    lb = scalar_baseline(sub, y, 'n_prompt')
    print(f"\n  known vs fictional (leave-one-entity-out)")
    print(f"    balanced accuracy {acc:.3f}   length-only baseline {lb:.3f}"
          f"   n={len(sub)}, entities={len({r['entity'] for r in sub})}")
    res['known_vs_fictional'] = {'balanced_accuracy': round(acc, 4),
                                 'length_only': round(lb, 4), 'n': len(sub)}

    # ---- the control: where does `obscure` fall? ---------------------------
    tr = [r for r in recs if r['cls'] in ('known', 'fictional')]
    Xtr = featurise(tr, meta['n_layers'], a.top_k, 'answer')
    ytr = np.array([1 if r['cls'] == 'known' else 0 for r in tr])
    C = counts(Xtr, ytr, meta['n_experts'])
    ob = [r for r in recs if r['cls'] == 'obscure']
    pv = score(C, featurise(ob, meta['n_layers'], a.top_k, 'answer')).argmax(1)
    frac = float(pv.mean())
    right = [r['correct'] for r in ob]
    fr_ok = float(np.mean([p for p, c in zip(pv, right) if c])) if any(right) else float('nan')
    fr_no = float(np.mean([p for p, c in zip(pv, right) if not c])) if not all(right) else float('nan')
    print(f"\n  obscure probes scored by that known/fictional profile:")
    print(f"    read as 'known' overall           {frac:.3f}  (n={len(ob)})")
    print(f"    ... among those answered RIGHT    {fr_ok:.3f}  "
          f"(n={sum(right)})")
    print(f"    ... among those answered WRONG    {fr_no:.3f}  "
          f"(n={len(ob) - sum(right)})")
    print(f"\n    A rarity detector puts obscure near fictional (low).")
    print(f"    A retrieval-success detector splits it by correctness.")
    res['obscure'] = {'n': len(ob), 'frac_known': round(frac, 4),
                      'frac_known_when_correct': None if fr_ok != fr_ok else round(fr_ok, 4),
                      'frac_known_when_wrong': None if fr_no != fr_no else round(fr_no, 4),
                      'n_correct': int(sum(right))}

    dest = OUT / 'fabrication.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
