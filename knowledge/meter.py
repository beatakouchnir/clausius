"""R2 — can routing alone tell where an answer came from? Classifier + controls.

Two suites run through this, and the first one FAILED in an instructive way.

`mechanism` (recall vs derivation, topic held constant) scored 1.000 — and
0.995 when relabelled "is the answer numeric", which is the same information.
Restricted to numeric answers only it fell to chance. Controlling topic had
reintroduced a confound elsewhere: every derivation in that suite yields a
number, most recall answers are words, and the router was reading the FORM of
the next token.

`grounding` fixes it by holding the answer token itself identical across
classes — same question, same answer, differing only in whether the fact is
available in context. See `probes.py`.

Every condition reports a LENGTH-ONLY baseline beside its accuracy, because
routing at the answer position turns out to be strongly sensitive to how many
tokens preceded it: `parametric vs distractor` scores 0.962 from routing and
0.984 from length alone. A condition whose length baseline matches its accuracy
is a length detector and is flagged as such.

The classifier is deliberately small. With 35 facts and 40x256 = 10,240 routing
features, anything with fitted per-feature weights would memorise the training
facts; a naive-Bayes profile over expert usage has no free parameters beyond a
smoothing constant, and it is continuous with R1's top-K set methodology.

  P(expert | class, layer)  estimated from training probes, Laplace-smoothed
  score(probe, class)       sum of log P over the experts actually selected at
                            the answer-token position, across all layers
  prediction                argmax, UNIFORM prior — a prior would let the
                            classifier bank the class imbalance instead of
                            reading the router

EVALUATION IS LEAVE-ONE-FACT-OUT, not leave-one-probe-out. A fact's three
paraphrases and its derive items are the same fact; splitting them across train
and test would let the model recognise the fact rather than the mechanism —
the same independence trap that pinned every R1 p-value to the floor when
tokens were permuted instead of prompts.

THE CONDITIONS THAT DECIDE THE GROUNDING RESULT:

  contextual vs distractor   THE CLAIM. Both carry a prepended fact, both emit
                             the same answer token, lengths match — only one
                             context contains the answer.
  parametric vs distractor   FORMAT control. Both require retrieval, differing
                             only by an irrelevant prefix. It scores high, but
                             its length baseline scores higher, which is how we
                             know that row is prompt shape rather than routing.

Needs numpy. No model, no GPU.

Usage:
  python3 -m knowledge.meter --trace records/probe_gate.<model>.jsonl.gz
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'records'


def load(path):
    with gzip.open(path, 'rt') as f:
        meta = json.loads(f.readline())
        return meta, [json.loads(l) for l in f]


def featurise(recs, n_layers, top_k, pos):
    """(N, n_layers, top_k) expert ids at the decisive position.

    `predict_pos` is the step whose routing decides the FIRST answer token —
    W5's "answer token", the position where recall either happens or does not.
    Whole-sequence routing would drown it in the shared instruction wrapper,
    which is identical for both classes by construction.
    """
    X = np.zeros((len(recs), n_layers, top_k), dtype=np.int64)
    for i, r in enumerate(recs):
        for l in range(n_layers):
            rows = r['ranks'][str(l)]
            if pos == 'mean':      # union over the answer span, first token in
                span = rows[r['predict_pos']:] or [rows[-1]]
                sel = [e for row in span for e in row[:top_k]]
                X[i, l] = np.array(sel[:top_k] if len(sel) >= top_k
                                   else (sel * top_k)[:top_k])
            else:
                X[i, l] = np.array(rows[r['predict_pos']][:top_k])
    return X


def counts(X, y, n_experts, n_cls=2):
    """C[c, layer, expert] = times class c selected that expert there."""
    L, k = X.shape[1], X.shape[2]
    C = np.zeros((n_cls, L, n_experts), dtype=np.float64)
    lay = np.repeat(np.arange(L)[None, :], k, axis=0).T.ravel()
    for i in range(len(X)):
        np.add.at(C[y[i]], (lay, X[i].ravel()), 1.0)
    return C


def score(C, X_test, alpha=1.0):
    """Log-likelihood of each test probe under each class profile."""
    tot = C.sum(axis=2, keepdims=True)
    logp = np.log((C + alpha) / (tot + alpha * C.shape[2]))
    L = X_test.shape[1]
    out = np.zeros((len(X_test), C.shape[0]))
    for c in range(C.shape[0]):
        for i in range(len(X_test)):
            out[i, c] = logp[c][np.arange(L)[:, None], X_test[i]].sum()
    return out


def scalar_baseline(sub, y, field):
    """Best accuracy obtainable from ONE scalar cue alone, no cross-validation.

    Deliberately optimistic — the threshold is chosen on the test data itself —
    because it is used to rule a confound OUT. If routing beats this by a wide
    margin the signal is not length; if it does not, the condition is measuring
    how many tokens preceded the answer and nothing more.

    This is not hypothetical. `parametric vs distractor` scores 0.962 from
    routing and 0.984 from length: that condition is a length detector.
    """
    xs = np.array([r.get(field, 0) for r in sub])
    best = 0.0
    for t in np.unique(xs):
        best = max(best, balanced_acc(y, (xs >= t).astype(int)),
                   balanced_acc(y, (xs < t).astype(int)))
    return best


def balanced_acc(y, pred):
    accs = [float((pred[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return sum(accs) / len(accs) if accs else float('nan')


def loo_by_fact(X, y, facts, n_experts, alpha=1.0):
    """Leave-one-fact-out predictions. Folds subtract, never rebuild."""
    C_all = counts(X, y, n_experts)
    pred = np.zeros(len(y), dtype=np.int64)
    for f in np.unique(facts):
        te = facts == f
        C = C_all - counts(X[te], y[te], n_experts)
        if (C.sum(axis=(1, 2)) <= 0).any():      # a class emptied by the fold
            C = C_all
        pred[te] = score(C, X[te], alpha).argmax(axis=1)
    return pred


def permutation_null(X, y, facts, n_experts, n_perm, seed, alpha=1.0):
    """Null accuracy from shuffled class labels, same CV structure."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        out.append(balanced_acc(yp, loo_by_fact(X, yp, facts, n_experts, alpha)))
    return np.sort(np.array(out))


def transfer(train, test, meta, pos_cls, neg_cls, top_k, pos, drop_facts=()):
    """Train a profile on one suite, apply it to another. No CV needed — the
    test probes come from a different suite entirely.

    The generalisation test the within-suite numbers cannot give. R3 scores
    0.982, but three suites in a row have shown that a number can be high for a
    reason specific to how the probes were written. If a profile trained on
    R3's numeric arithmetic still labels a *word*-answer retrieval probe from
    another suite as `retrieved`, the meter is reading mechanism rather than
    template.

    `drop_facts` removes test probes whose underlying fact also appears in the
    training suite. Six historical facts (Westphalia, the Moon landing, ...)
    were written into both, and leaving them in would let the profile recognise
    a fact it was trained on rather than generalise to a new one.
    """
    tr = [r for r in train if r['cls'] in (pos_cls, neg_cls)]
    X = featurise(tr, meta['n_layers'], top_k, pos)
    y = np.array([1 if r['cls'] == pos_cls else 0 for r in tr])
    C = counts(X, y, meta['n_experts'])

    te = [r for r in test if r['fact_id'] not in drop_facts]
    Xt = featurise(te, meta['n_layers'], top_k, pos)
    pred = score(C, Xt).argmax(axis=1)
    out = {}
    for cls in sorted({r['cls'] for r in te}):
        m = np.array([r['cls'] == cls for r in te])
        out[cls] = {'n': int(m.sum()),
                    f'frac_{pos_cls}': round(float(pred[m].mean()), 4)}
    return out


def condition(recs, meta, name, pos_cls, neg_cls, top_k, pos, n_perm, seed,
              extra=None):
    sub = [r for r in recs if r['cls'] in (pos_cls, neg_cls)
           and (extra is None or extra(r))]
    if len({r['cls'] for r in sub}) < 2:
        print(f"  {name}: only one class present, skipped")
        return None
    X = featurise(sub, meta['n_layers'], top_k, pos)
    y = np.array([1 if r['cls'] == pos_cls else 0 for r in sub])
    fnames = sorted({r['fact_id'] for r in sub})
    facts = np.array([fnames.index(r['fact_id']) for r in sub])
    E = meta['n_experts']

    pred = loo_by_fact(X, y, facts, E)
    acc = balanced_acc(y, pred)
    null = permutation_null(X, y, facts, E, n_perm, seed)
    p = (int((null >= acc).sum()) + 1) / (len(null) + 1)
    lb = scalar_baseline(sub, y, 'n_prompt')
    nb = scalar_baseline(sub, y, 'n_numbers')
    worst = max(lb, nb)
    print(f"  {name:34s} n={len(sub):4d} ({int((y == 1).sum())}/"
          f"{int((y == 0).sum())})  facts={len(fnames):3d}  "
          f"bal.acc {acc:.3f}   null {np.median(null):.3f} "
          f"[p95 {null[int(.95 * len(null))]:.3f}]   p {p:.4f}"
          f"   len {lb:.3f}  digits {nb:.3f}"
          f"{'  <-- CUE' if worst >= acc - .05 else ''}")
    return {'name': name, 'classes': [pos_cls, neg_cls],
            'n': len(sub), 'n_facts': len(fnames),
            'balanced_accuracy': round(acc, 4),
            'length_only_baseline': round(lb, 4),
            'digitcount_only_baseline': round(nb, 4),
            'null_median': round(float(np.median(null)), 4),
            'null_p95': round(float(null[int(.95 * len(null))]), 4),
            'p_value': round(p, 4)}


def per_layer(recs, meta, pos_cls, neg_cls, top_k, pos):
    """Single-layer accuracy — where in depth does the signal live?"""
    sub = [r for r in recs if r['cls'] in (pos_cls, neg_cls)]
    X = featurise(sub, meta['n_layers'], top_k, pos)
    y = np.array([1 if r['cls'] == pos_cls else 0 for r in sub])
    fnames = sorted({r['fact_id'] for r in sub})
    facts = np.array([fnames.index(r['fact_id']) for r in sub])
    out = []
    for l in range(meta['n_layers']):
        pred = loo_by_fact(X[:, l:l + 1, :], y, facts, meta['n_experts'])
        out.append(round(balanced_acc(y, pred), 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trace', default=str(
        OUT / 'probe_gate.qwen36-35b-a3b-4bit-g64.jsonl.gz'))
    ap.add_argument('--top-k', type=int, default=8,
                    help='experts per token to use as features. The model '
                         'routes to 8; larger reaches into experts the router '
                         'ranked but did not select.')
    ap.add_argument('--pos', default='answer', choices=('answer', 'mean'))
    ap.add_argument('--perms', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all-probes', action='store_true',
                    help='include probes the model got wrong (default: only '
                         'correct ones, so the contrast is clean)')
    a = ap.parse_args()

    meta, recs = load(a.trace)
    ok = recs if a.all_probes else [r for r in recs if r['correct']]
    print(f"{meta['model'].split('/')[-1]} · {meta['n_layers']} layers · "
          f"{meta['n_experts']} experts · top_k feature {a.top_k} · "
          f"pos={a.pos}")
    print(f"{len(ok)}/{len(recs)} probes used"
          f"{'' if a.all_probes else ' (correct only)'}\n")

    res = {'trace': Path(a.trace).name, 'top_k': a.top_k, 'pos': a.pos,
           'perms': a.perms, 'n_used': len(ok), 'conditions': []}

    classes = {r['cls'] for r in ok}
    print("  condition                          size          "
          "facts     accuracy   permutation null")
    add = lambda *args, **kw: res['conditions'].append(  # noqa: E731
        condition(ok, meta, *args, a.top_k, a.pos, a.perms, a.seed, **kw))

    if 'retrieved' in classes:
        # R3: both classes emit the SAME number for a given fact, so answer
        # form and magnitude cannot leak. The residual cue is that `computed`
        # must carry an operand — hence the digit-count baseline.
        add('retrieved vs computed (CLAIM)', 'retrieved', 'computed')
        for k in ('year', 'count'):
            add(f'claim, {k} facts only', 'retrieved', 'computed',
                extra=lambda r, kk=k: r['dkind'] == kk)
        head = ('retrieved', 'computed')
    elif 'contextual' in classes:
        # the claim: both sides carry a prepended fact, only one contains the
        # answer, and the answer TOKEN is identical across classes
        add('contextual vs distractor (CLAIM)', 'contextual', 'distractor')
        # the control that matters: both sides require retrieval and differ
        # only by an irrelevant prefix. Above chance here means the meter is
        # reading prompt FORMAT or length, not retrieval — and would void the
        # claim above, which differs in exactly the same superficial way.
        add('parametric vs distractor (FORMAT)', 'parametric', 'distractor')
        add('contextual vs parametric (confounded)', 'contextual', 'parametric')
        for at in ('word', 'num'):
            add(f'claim, {at}-answer only', 'contextual', 'distractor',
                extra=lambda r, t=at: r['atype'] == t)
        head = ('contextual', 'distractor')
    else:
        add('matched (mechanism)', 'recall', 'derive',
            extra=lambda r: r['matched'])
        add('unmatched (topic baseline)', 'recall', 'derive',
            extra=lambda r: r['cls'] == 'recall' or not r['matched'])
        for dk in ('arith', 'year', 'letters'):
            add(f'matched, {dk} only', 'recall', 'derive',
                extra=lambda r, d=dk: r['matched'] and (r['cls'] == 'recall'
                                                        or r['dkind'] == d))
        head = ('recall', 'derive')
    res['conditions'] = [c for c in res['conditions'] if c]

    pl = per_layer(ok, meta, head[0], head[1], a.top_k, a.pos)
    res['per_layer_matched'] = pl
    best = sorted(range(len(pl)), key=lambda i: -pl[i])[:8]
    print(f"\n  per-layer accuracy (matched): best 8 layers "
          f"{[(l, pl[l]) for l in best]}")
    print(f"  median layer {np.median(pl):.3f} · max {max(pl):.3f} "
          f"· min {min(pl):.3f}")

    # keyed on suite AND model: keying on suite alone means a second
    # architecture silently overwrites the first one's result, which is exactly
    # how the routing records got clobbered twice
    stem = Path(a.trace).name.replace('probe_gate.', '').replace('.jsonl.gz', '')
    dest = OUT / f"meter.{stem}.json"
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
