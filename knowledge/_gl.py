"""Calibration metrics — ECE, MCE, Brier, risk-coverage, AURC, recalibration.

A 564-line, numpy-only calibration suite, alongside an n=740 study on
gemma-4-26b-a4b. Reimplementing AURC here would risk a number that is not
comparable with the 0.087 / 0.259 that study recorded for 8-vote
self-consistency on math / MCQ, so the implementation is shared rather than
rewritten.

It used to be imported by path from a sibling repository that is not published.
The suite is now vendored at `_vendor/calibration.py`, and this module is a flat
re-export of it, so callers are unchanged.
"""
from ._vendor import calibration as _c

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
