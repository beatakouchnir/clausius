"""Rung 3 — does an individual FACT have a routing address, or just a topic?

The naive test would have looked like a success and meant nothing. Ask "which
fact is this?" across paraphrases of *the capital of Australia* and every one
of them contains the word Australia; a classifier reading subject matter scores
well and has told you nothing about facts.

The crossed grid (see `probes.py`) removes topic from both directions:

  within entity    France's capital vs France's currency vs France's language.
                   Vocabulary is nearly identical; only the fact differs. A
                   topic reader is blind here.
  within relation  the capital of France vs of Japan vs of Brazil. The template
                   is identical; only the entity differs. A question-form
                   reader is blind here.

Two further requirements, both of which this enforces:

  HOLD OUT A WHOLE PARAPHRASE. Train on two wordings, test on the third. A
  split that mixes wordings of the same cell lets the classifier match surface
  form, which is the thing being controlled for.

  BEAT BAG-OF-WORDS. The prompt text itself identifies the fact — that is what
  a prompt is for. The claim "routing addresses the fact" only means something
  if routing does it at least as well as reading the words, since otherwise the
  routing pattern is a lossy re-encoding of the prompt and adds nothing.

Needs numpy. No model, no GPU.

Usage:
  python3 -m knowledge.identity
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .meter import load, featurise, counts, score, OUT
from .probes import all_probes


def nb_words(train_txt, train_y, test_txt, n_cls, alpha=0.3):
    """Multinomial naive Bayes over prompt words — the baseline to beat."""
    vocab = sorted({w for t in train_txt for w in t.lower().split()})
    idx = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((n_cls, len(vocab)))
    for t, c in zip(train_txt, train_y):
        for w in t.lower().split():
            if w in idx:
                C[c, idx[w]] += 1
    logp = np.log((C + alpha) / (C.sum(1, keepdims=True) + alpha * len(vocab)))
    out = []
    for t in test_txt:
        ws = [idx[w] for w in t.lower().split() if w in idx]
        out.append(int(logp[:, ws].sum(1).argmax()) if ws else 0)
    return np.array(out)


def evaluate(recs, texts, meta, key, group, top_k):
    """Leave-one-paraphrase-out accuracy, routing vs words, within each group.

    `key` is what is being identified; `group` is what is held constant. With
    key='relation', group='entity' the classifier must tell France's capital
    from France's currency — topic controlled. With key='entity',
    group='relation' it must tell the capital of France from the capital of
    Japan — template controlled.
    """
    rt, wd, n = [], [], 0
    for g in sorted({r[group] for r in recs}):
        sub = [r for r in recs if r[group] == g]
        labels = sorted({r[key] for r in sub})
        if len(labels) < 2:
            continue
        for held in sorted({r['para'] for r in sub}):
            tr = [r for r in sub if r['para'] != held]
            te = [r for r in sub if r['para'] == held]
            if not tr or not te:
                continue
            ytr = np.array([labels.index(r[key]) for r in tr])
            yte = np.array([labels.index(r[key]) for r in te])
            Xtr = featurise(tr, meta['n_layers'], top_k, 'answer')
            Xte = featurise(te, meta['n_layers'], top_k, 'answer')
            C = counts(Xtr, ytr, meta['n_experts'], n_cls=len(labels))
            rt.append((score(C, Xte).argmax(1) == yte).sum())
            wd.append((nb_words([texts[r['probe_id']] for r in tr], ytr,
                                [texts[r['probe_id']] for r in te],
                                len(labels)) == yte).sum())
            n += len(te)
    return sum(rt) / n, sum(wd) / n, n, len(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trace', default=str(
        OUT / 'probe_gate.grid.qwen36-35b-a3b-4bit-g64.jsonl.gz'))
    ap.add_argument('--suite', default='grid', choices=('grid', 'grid2'))
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--lenient', action='store_true',
                    help='use correct_lenient (spelling/case mismatches kept)')
    ap.add_argument('--all-probes', action='store_true')
    a = ap.parse_args()

    meta, recs = load(a.trace)
    key = 'correct_lenient' if a.lenient else 'correct'
    if not a.all_probes:
        recs = [r for r in recs if r.get(key, r['correct'])]
    texts = {p['probe_id']: p['stem'] for p in all_probes(a.suite)}
    print(f"{meta['model'].split('/')[-1]} · grid · {len(recs)} probes "
          f"({len({r['entity'] for r in recs})} entities x "
          f"{len({r['relation'] for r in recs})} relations)\n")

    res = {'trace': Path(a.trace).name, 'conditions': []}
    print(f"  {'condition':38s} {'n':>5s} {'classes':>8s} {'chance':>8s} "
          f"{'routing':>9s} {'words':>8s}")
    for r in recs:
        r['_ent'] = f"{r['domain']}.{r['entity']}"
        r['_rel'] = f"{r['domain']}.{r['relation']}"
    for key, group, label in (
            ('relation', '_ent', 'identify RELATION, entity fixed'),
            ('entity', '_rel', 'identify ENTITY, relation fixed')):
        acc, wacc, n, k = evaluate(recs, texts, meta, key, group, a.top_k)
        print(f"  {label:38s} {n:5d} {k:8d} {1 / k:8.3f} "
              f"{acc:9.3f} {wacc:8.3f}")
        res['conditions'].append({
            'identify': key, 'held_constant': group, 'n': n, 'n_classes': k,
            'chance': round(1 / k, 4), 'routing_accuracy': round(acc, 4),
            'bagofwords_accuracy': round(wacc, 4)})

    # per-domain, because a pooled number hides whether the effect is uniform
    # or carried by one domain. R5's grid was countries only, so the pooled
    # figure there WAS the country figure; here they can diverge.
    doms = sorted({r['domain'] for r in recs})
    if len(doms) > 1:
        print(f"\n  per domain:")
        print(f"  {'domain':10s} {'n':>5s} {'rel routing':>12s} {'rel words':>10s}"
              f" {'ent routing':>12s} {'ent words':>10s}")
        for d in doms:
            sub = [r for r in recs if r['domain'] == d]
            ra, rw, rn, _ = evaluate(sub, texts, meta, 'relation', '_ent', a.top_k)
            ea, ew, en, _ = evaluate(sub, texts, meta, 'entity', '_rel', a.top_k)
            print(f"  {d:10s} {rn:5d} {ra:12.3f} {rw:10.3f} {ea:12.3f} {ew:10.3f}")
            res['conditions'].append({
                'domain': d, 'n': rn,
                'relation_routing': round(ra, 4), 'relation_words': round(rw, 4),
                'entity_routing': round(ea, 4), 'entity_words': round(ew, 4)})

    dest = OUT / f'identity.{a.suite}.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
