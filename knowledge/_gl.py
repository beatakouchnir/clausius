"""Thin re-export of ghostlight's calibration metrics.

ghostlight already has a 564-line, numpy-only calibration suite (ECE, MCE,
Brier, risk-coverage, AURC, Platt scaling, bootstrap CI, out-of-fold
recalibration) plus a published n=740 study on gemma-4-26b-a4b. Reimplementing
AURC here would risk a number that is not comparable with the 0.087 / 0.259 it
recorded for 8-vote self-consistency on math / MCQ.

ghostlight is read-only to this project, exactly as quantize is: imported by
path, never written to. Override the location with GHOSTLIGHT_REPO.
"""
import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent.parent / 'ghostlight'


def _load():
    repo = Path(os.environ.get('GHOSTLIGHT_REPO', _DEFAULT))
    if not (repo / 'harness' / 'calibration.py').exists():
        raise ImportError(f"ghostlight calibration not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from harness import calibration
    return calibration


_c = _load()
risk_coverage = _c.risk_coverage
ece = _c.ece
brier = _c.brier
summary = _c.summary
vote_signals = _c.vote_signals
selective_accuracy_at = _c.selective_accuracy_at
coverage_at_risk = _c.coverage_at_risk
platt_scale = _c.platt_scale
oof_recalibrated = _c.oof_recalibrated
cross_val_ece = _c.cross_val_ece
reliability_bins = _c.reliability_bins
