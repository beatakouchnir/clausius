"""Score every signal family against the membership benchmark, head to head.

Reports AUC per signal family, broken out by duplication dose. Two settings:

  single-model      only the fine-tuned model is queried. This is what a real
                    auditor can do, and it is the honest headline.
  reference-based   the signal minus its value on the BASE model. Strictly
                    stronger and strictly less applicable — it needs the
                    un-finetuned weights. Reported as an upper bound on what
                    the signal contains, never as the headline.

DIRECTIONS ARE FIXED A PRIORI, not chosen from the results. Members are
hypothesised to have lower perplexity, higher Min-K% and lower entropy, so the
scores are negated accordingly before AUC. Picking the orientation that flatters
each signal after seeing the data would inflate every number and is exactly the
kind of quiet degree of freedom that makes published MIA results irreproducible.

LEARNED DETECTORS ARE FIT ON THE `fit` SPLIT ONLY and scored on `eval`. The
split was drawn at random when the corpus was generated, independently of
membership.

THE BLIND BASELINE IS THE VALIDITY CHECK. If it is not at chance, nothing else
on the page means anything, so it is printed first and flagged loudly.

Needs numpy. No model, no GPU.

Usage:
  python3 -m knowledge.detect --sig records/corpus/sig.router.npz \
      --base records/corpus/sig.base.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .corpus import blind_baseline, CORPUS, DUP_SCHEDULE


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def auc(pos, neg):
    """P(score(member) > score(non-member)), ties counted as half."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _sigmoid(z):
    """Overflow-safe. The naive form raised divide-by-zero and invalid-value
    warnings on the real feature matrix, which silently poisons the fit."""
    z = np.clip(z, -30, 30)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def logreg(Xtr, ytr, Xte, l2=1.0, iters=3000, lr=0.1):
    """Plain L2 logistic regression. Standardisation uses TRAIN statistics
    only — fitting the scaler on the test set leaks the test distribution.

    Near-constant columns are dropped rather than standardised: layer 0's
    cosine-to-previous is identically zero by construction, and dividing it by
    a floored standard deviation manufactures a huge spurious feature.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-9
    if not keep.any():
        return np.zeros(len(Xte))
    A = (Xtr[:, keep] - mu[keep]) / sd[keep]
    B = (Xte[:, keep] - mu[keep]) / sd[keep]
    w, b = np.zeros(A.shape[1]), 0.0
    # numpy 2.0.2 raises a spurious "divide by zero encountered in matmul" on
    # perfectly finite inputs — a BLAS status flag, not a real division. It is
    # suppressed here, but the RESULT is asserted finite, so a genuine
    # divergence (which an earlier version of this function did produce) still
    # fails loudly instead of silently poisoning the fit.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(iters):
            p = _sigmoid(A @ w + b)
            w -= lr * (A.T @ (p - ytr) / len(ytr) + l2 * w / len(ytr))
            b -= lr * (p - ytr).mean()
        out = B @ w + b
    if not np.isfinite(out).all():
        raise FloatingPointError("logreg diverged — non-finite scores")
    return out


def nb_counts(Xtr, ytr, Xte, alpha=1.0):
    """Naive Bayes over per-layer expert usage counts; score is the LLR.

    The same estimator the meter used, so a routing result here is continuous
    with R1-R7 rather than a new modelling choice smuggled in at the end.
    """
    L, E = Xtr.shape[1], Xtr.shape[2]
    out = np.zeros((2, L, E))
    for c in (0, 1):
        out[c] = Xtr[ytr == c].sum(0)
    lp = np.log((out + alpha) / (out.sum(2, keepdims=True) + alpha * E))
    return (Xte * lp[1]).sum((1, 2)) - (Xte * lp[0]).sum((1, 2))


def evaluate(docs, sig, base, name):
    member = np.array([d['member'] for d in docs])
    dup = np.array([d['dup'] for d in docs])
    is_eval = np.array([d['split'] == 'eval' for d in docs])
    fit = ~is_eval

    # --- scalar families, no fitting, direction fixed in advance ------------
    scalars = {'perplexity': -sig['ppl'], 'min-k%': sig['mink'],
               'entropy': -sig['ent']}
    if base is not None:
        scalars.update({
            'perplexity (ref)': -(sig['ppl'] - base['ppl']),
            'min-k% (ref)': sig['mink'] - base['mink'],
            'entropy (ref)': -(sig['ent'] - base['ent'])})

    # --- learned families ---------------------------------------------------
    learned = {}
    R = sig['routing'].astype(np.float64)
    H = sig['hidden'].reshape(len(docs), -1).astype(np.float64)
    learned['routing'] = (R, nb_counts)
    learned['hidden'] = (H, None)
    if base is not None:
        learned['routing (ref)'] = (R - base['routing'].astype(np.float64),
                                    None)
        learned['hidden (ref)'] = (
            H - base['hidden'].reshape(len(docs), -1).astype(np.float64), None)

    rows = {}
    for label, s in scalars.items():
        rows[label] = {'all': auc(s[is_eval & member], s[is_eval & ~member])}
        for d in DUP_SCHEDULE:
            rows[label][d] = auc(s[is_eval & (dup == d)],
                                 s[is_eval & ~member])

    for label, (X, est) in learned.items():
        if est is nb_counts:
            sc = np.zeros(len(docs))
            sc[is_eval] = nb_counts(X[fit], member[fit].astype(int), X[is_eval])
        else:
            flat = X if X.ndim == 2 else X.reshape(len(docs), -1)
            sc = np.zeros(len(docs))
            sc[is_eval] = logreg(flat[fit], member[fit].astype(float),
                                 flat[is_eval])
        rows[label] = {'all': auc(sc[is_eval & member], sc[is_eval & ~member])}
        for d in DUP_SCHEDULE:
            rows[label][d] = auc(sc[is_eval & (dup == d)],
                                 sc[is_eval & ~member])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sig', required=True, help='signals from the tuned model')
    ap.add_argument('--base', default=None, help='signals from the base model')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()

    docs = json.loads((CORPUS / 'manifest.json').read_text())
    sig = np.load(a.sig)
    base = np.load(a.base) if a.base else None
    name = a.name or Path(a.sig).stem

    bl = blind_baseline(docs)
    ok = abs(bl - 0.5) < 0.05
    print(f"blind bag-of-words membership: {bl:.3f}  "
          f"[{'PASS — benchmark valid' if ok else 'FAIL — READ NOTHING BELOW'}]\n")

    rows = evaluate(docs, sig, base, name)
    hdr = ['all'] + list(DUP_SCHEDULE)
    print(f"  {'signal':22s} " + ' '.join(f"{('x' + str(h)) if h != 'all' else 'ALL':>7s}"
                                          for h in hdr))
    for label, r in rows.items():
        cells = ' '.join(f"{r[h]:7.3f}" for h in hdr)
        star = '  <-- reference setting' if '(ref)' in label else ''
        print(f"  {label:22s} {cells}{star}")
    print(f"\n  AUC 0.500 = chance. Columns are duplication dose; x1 is the "
          f"realistic\n  contamination case and the one that matters.")

    dest = CORPUS.parent / f'membership.{name}.json'
    dest.write_text(json.dumps(
        {'signals': name, 'blind_baseline': round(bl, 4),
         'rows': {k: {str(kk): (None if vv != vv else round(vv, 4))
                      for kk, vv in v.items()} for k, v in rows.items()}},
        indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
