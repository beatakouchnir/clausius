"""Tests for the comparison logic.

None of these load a model. `compare()` and `aggregate()` are pure numpy, which
is deliberate — it means CI needs no accelerator, and anyone re-analysing the
published measurement corpus can do it on a laptop.

Each test guards a decision that came out of a measurement rather than a
preference, and several encode bugs that actually happened.
"""
import numpy as np
import pytest

from clausius import Capture, aggregate, compare
from clausius.core import DEFAULT_THRESHOLD


def make(rows_ent, prompts=None, truncated=None, tag='t'):
    """A Capture from a list of per-item `max`-entropy values."""
    rows = []
    for i, e in enumerate(rows_ent):
        rows.append({
            'ent': {'first': e, 'mean': e, 'max': e, 'p90': e,
                    'mean_top10': e, 'gen_len': 100.0},
            'truncated': bool(truncated[i]) if truncated else False,
            'empty_gen': False})
    return Capture(model='m', tag=tag,
                   prompts=prompts or [f'p{i}' for i in range(len(rows_ent))],
                   rows=rows)


# --- aggregate ------------------------------------------------------------

def test_aggregate_handles_empty_generation():
    """A config can be damaged badly enough to generate nothing.

    Top-k <= 2 on one model emits an immediate stop, and an unguarded index into
    the empty generated span raised IndexError and killed four experiment arms.
    An immediate stop is an observation about the damage, not a reason to crash.
    """
    ent = np.array([1.0, 2.0, 3.0])
    out = aggregate(ent, n_prompt=len(ent) + 1)   # no generated positions
    assert np.isfinite(out['max'])
    assert out['gen_len'] >= 1


def test_aggregate_uses_generated_span_only():
    ent = np.concatenate([np.zeros(10), np.full(5, 4.0)])
    out = aggregate(ent, n_prompt=11)
    assert out['max'] == pytest.approx(4.0)
    assert out['mean'] == pytest.approx(4.0)
    assert out['gen_len'] == pytest.approx(5)


# --- pairing guards -------------------------------------------------------

def test_rejects_different_lengths():
    with pytest.raises(ValueError, match='different lengths'):
        compare(make([1.0] * 30), make([1.0] * 29))


def test_rejects_different_prompts():
    """Silently comparing two prompt sets would compare datasets, not configs."""
    a = make([1.0] * 30, prompts=[f'a{i}' for i in range(30)])
    b = make([1.0] * 30, prompts=[f'b{i}' for i in range(30)])
    with pytest.raises(ValueError, match='different prompts'):
        compare(a, b)


def test_rejects_too_few_surviving_items():
    trunc = [True] * 25 + [False] * 5
    with pytest.raises(ValueError, match='survive truncation'):
        compare(make([1.0] * 30, truncated=trunc),
                make([1.0] * 30, truncated=trunc))


# --- verdicts -------------------------------------------------------------

def test_identical_captures_are_clean():
    rng = np.random.default_rng(0)
    base = rng.normal(3.0, 1.0, 40).tolist()
    r = compare(make(base), make(base))
    assert not r.flagged
    assert r.effect == pytest.approx(0.0, abs=1e-6)


def test_consistent_entropy_rise_is_flagged():
    rng = np.random.default_rng(1)
    base = rng.normal(3.0, 1.0, 60)
    r = compare(make(base.tolist()), make((base + 1.0).tolist()))
    assert r.flagged and r.verdict == 'REGRESSION'


def test_noise_without_shift_is_not_flagged():
    """A benign config perturbs individual items without shifting the centre.

    This is the real benign case: offload capacity changes alter ~25% of
    generations textually while accuracy and entropy stay put.
    """
    rng = np.random.default_rng(2)
    base = rng.normal(3.0, 1.0, 200)
    jitter = base + rng.normal(0.0, 0.5, 200)   # symmetric, no mean shift
    r = compare(make(base.tolist()), make(jitter.tolist()))
    assert not r.flagged


# --- one-sided vs two-sided ----------------------------------------------

def test_entropy_drop_is_clean_one_sided_but_flagged_two_sided():
    """The logit-sharpening case, which is why one-sided is the default.

    Scaling logits leaves greedy output bit-identical — accuracy delta exactly
    zero — while collapsing entropy. A two-sided test calls that a regression.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(3.0, 1.0, 60)
    sharpened = base - 1.5
    assert not compare(make(base.tolist()), make(sharpened.tolist())).flagged
    assert compare(make(base.tolist()), make(sharpened.tolist()),
                   one_sided=False).flagged


# --- truncation -----------------------------------------------------------

def test_truncated_items_are_dropped_from_either_side():
    trunc = [False] * 50 + [True] * 10
    r = compare(make([1.0] * 60), make([1.0] * 60, truncated=trunc))
    assert r.n_compared == 50 and r.n_dropped_truncated == 10


def test_keep_truncated_is_opt_in():
    trunc = [False] * 50 + [True] * 10
    r = compare(make([1.0] * 60), make([1.0] * 60, truncated=trunc),
                drop_truncated=False)
    assert r.n_compared == 60


# --- documented constants -------------------------------------------------

def test_threshold_matches_the_measured_null():
    """13 configurations known harmless produced |d_z| <= 0.10; 0.3 is ~3x that.

    If this constant is ever changed, the calibration evidence in FINDINGS.md
    (F8b) has to change with it.
    """
    assert DEFAULT_THRESHOLD == 0.3


def test_all_signals_reported():
    r = compare(make([1.0] * 40), make([2.0] * 40))
    for s in ('max', 'p90', 'mean', 'mean_top10', 'first', 'gen_len'):
        assert s in r.detail


def test_round_trips_through_disk(tmp_path):
    a, b = make([1.0] * 40), make([2.0] * 40)
    pa, pb = a.save(tmp_path / 'a.json'), b.save(tmp_path / 'b.json')
    assert compare(pa, pb).flagged
