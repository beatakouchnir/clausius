"""Stage D — product metrics. AUC ranks; a buyer asks a different question.

Stages A-C established that `p90` of per-token entropy predicts error at
0.71-0.92 AUC across 6 task configurations and 5 models, architecture-
independently. None of that tells anyone whether to deploy it, because AUC is a
ranking statistic and a deployment has a review budget.

The question a buyer actually asks is: **"if I have a human check 20% of
answers, what fraction of the errors do I catch?"** That is error-catch at a
fixed budget, and it is what this reports, alongside:

  catch@budget   errors caught when flagging the top 10/20/30% by entropy.
                 Compared against flagging AT RANDOM, which catches exactly the
                 budget fraction — the honest floor, since a manager reviewing
                 20% at random catches 20% of errors for free.
  AURC / E-AURC  selective-prediction risk. E-AURC subtracts the best AURC
                 achievable at that base rate, so it compares across tasks with
                 different accuracies. a prior study measured 8-vote
                 self-consistency at AURC 0.087 (math) / 0.259 (MCQ).
  ECE            calibration, raw and out-of-fold recalibrated. The vendored
                 Phase 2 finding applies directly: recalibration is a MONOTONIC
                 map, so it fixes ECE and **cannot change ranking** — AUC,
                 AURC and catch@budget are all unmoved by it. Reported to show
                 the confidence number is honest, not to improve detection.

Entropy is not a probability, so it is mapped to one by out-of-fold Platt
scaling before any calibration metric is computed. Ranking metrics use the raw
signal.

All metrics come from the vendored `_vendor/calibration.py` rather than being
reimplemented, so the numbers stay comparable with its n=740 study.

No GPU, no model — reads the Stage A/B captures.

Usage:
  python3 -m knowledge.stage_d
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .meter import OUT

BUDGETS = (0.10, 0.20, 0.30)


def ceiling_at(y, budget):
    """Best catch rate ANY signal could achieve at this budget.

    Flagging fraction b of n items can surface at most b*n errors, so with an
    error rate p the ceiling is min(b/p, 1). This is the number that makes
    catch@budget interpretable: on AA-Omniscience p=0.86, so flagging 20% can
    catch at most 23% of errors NO MATTER how good the signal is. Reading a
    22.8% catch as "barely better than random" would be wrong — it is 98% of
    what is achievable.
    """
    p = y.mean()
    return min(budget / p, 1.0) if p > 0 else 1.0


def catch_at(score_err, y, budget):
    """Fraction of errors caught by flagging the top `budget` by score.

    `score_err` is oriented so HIGHER = more error-like. Ties are broken
    stably, so a signal with no information degrades to the random floor rather
    than to something accidentally better.
    """
    n = len(y)
    k = max(1, int(round(budget * n)))
    order = np.argsort(-score_err, kind='stable')[:k]
    return float(y[order].sum() / max(y.sum(), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--signal', default='p90',
                    help='entropy variant to evaluate (Stage A winner: p90)')
    a = ap.parse_args()

    from . import _gl
    files = sorted(OUT.glob('stage_a.*.json'))
    rows_out = {}
    print(f"  signal: {a.signal} entropy · budgets flag the most error-like "
          f"answers for review")
    print(f"  {'task':13s} {'model':22s} {'errs':>5s} | "
          f"{'err%':>5s} {'@20%':>6s} {'ceil':>6s} {'of ceil':>8s} | "
          f"{'E-AURC':>7s} | {'ECE':>6s} {'ECEcal':>7s}")
    for f in files:
        if f.name == 'stage_a.json':
            continue
        d = json.loads(f.read_text())
        rows = [r for r in d['rows']
                if not r['abstained'] and not r['truncated']]
        if len(rows) < 40:
            continue
        y = np.array([0.0 if r['correct'] else 1.0 for r in rows])
        if len(np.unique(y)) < 2 or y.sum() < 10:
            continue
        ent = np.array([r['ent'][a.signal] for r in rows])
        ok = np.isfinite(ent)
        ent, y = ent[ok], y[ok]

        catches = [catch_at(ent, y, b) for b in BUDGETS]
        ceils = [ceiling_at(y, b) for b in BUDGETS]
        frac = [c / cl if cl > 0 else float('nan')
                for c, cl in zip(catches, ceils)]
        corr = 1.0 - y                       # 1 = answer was correct
        conf = -ent                          # higher = more confident
        rc = _gl.risk_coverage(conf, corr)
        # map to a probability so ECE means something; out-of-fold so the
        # recalibration is not scored on the data that fitted it
        z = (conf - conf.mean()) / (conf.std() + 1e-9)
        raw = 1 / (1 + np.exp(-z))
        try:
            cal = _gl.oof_recalibrated(raw, corr)
            ece_cal = _gl.ece(cal, corr)
        except Exception:
            ece_cal = float('nan')
        ece_raw = _gl.ece(raw, corr)

        label = f"{d['task']}/{d.get('cap', '?')}"
        print(f"  {label:13s} {d['model'][:22]:22s} {int(y.sum()):5d} | "
              f"{y.mean():5.0%} {catches[1]:6.1%} {ceils[1]:6.1%} "
              f"{frac[1]:8.0%} | {rc['e_aurc']:7.3f} | "
              f"{ece_raw:6.3f} {ece_cal:7.3f}")
        rows_out[f"{d['task']}/{d['model']}"] = {
            'n': len(y), 'n_errors': int(y.sum()),
            'error_rate': round(float(y.mean()), 4),
            'catch_at': {f'{int(b * 100)}%': round(c, 4)
                         for b, c in zip(BUDGETS, catches)},
            'ceiling_at': {f'{int(b * 100)}%': round(c, 4)
                           for b, c in zip(BUDGETS, ceils)},
            'frac_of_ceiling_at': {f'{int(b * 100)}%': round(c, 4)
                                   for b, c in zip(BUDGETS, frac)},
            'aurc': round(rc['aurc'], 4), 'e_aurc': round(rc['e_aurc'], 4),
            'ece_raw': round(ece_raw, 4),
            'ece_recalibrated': None if ece_cal != ece_cal else round(ece_cal, 4),
            'selective_accuracy_at_50pct': round(
                _gl.selective_accuracy_at(conf, corr, 0.5), 4)}

    print(f"\n  'ceil' = best any signal could do at 20% budget = min(0.2/err, 1)."
          f"\n  'of ceil' = how much of that is achieved. Random review catches "
          f"the\n  budget fraction (20%), so compare BOTH: floor 20%, and the "
          f"ceiling.")
    print(f"  Recalibration is a monotonic map: it moves ECE, and cannot move "
          f"catch@budget or AURC.")
    dest = OUT / 'stage_d.json'
    dest.write_text(json.dumps({'signal': a.signal, 'budgets': list(BUDGETS),
                                'results': rows_out}, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
