"""Tests for the comparison logic.

None of these load a model. `compare()` and `aggregate()` are pure numpy, which
is deliberate — it means CI needs no accelerator, and anyone re-analysing the
published measurement corpus can do it on a laptop.

Each test guards a decision that came out of a measurement rather than a
preference, and several encode bugs that actually happened.
"""
import json

import numpy as np
import pytest

from clausius import Capture, aggregate, capture, compare, truncation_curve
from clausius.cli import main as cli_main
from clausius.core import DEFAULT_THRESHOLD, MIN_PAIRED_ITEMS


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


# --- truncation curve -----------------------------------------------------

def with_lengths(lengths, cap):
    """A Capture carrying explicit generated lengths and the cap it used."""
    rows = [{'ent': {'first': 1.0, 'mean': 1.0, 'max': 1.0, 'p90': 1.0,
                     'mean_top10': 1.0, 'gen_len': float(L)},
             'truncated': L >= cap - 2, 'empty_gen': False}
            for L in lengths]
    return Capture(model='m', tag='t',
                   prompts=[f'p{i}' for i in range(len(lengths))],
                   rows=rows, meta={'max_tokens': cap})


def test_curve_reproduces_a_measured_tighter_cap():
    """The projection is exact downward, not an estimate.

    Measured: a 60-prompt capture at cap 1536 predicted 47/60 truncated at cap
    512, and a separate capture actually run at 512 truncated exactly 47. That
    equality is the whole basis for reporting the curve instead of re-running,
    so it is pinned here.
    """
    cap = with_lengths([1000.0] * 47 + [100.0] * 13, cap=1536)
    row = next(r for r in truncation_curve(cap).rows if r['cap'] == 512)
    assert row['truncated'] == 47
    assert row['survivors'] == 13


def test_curve_never_extrapolates_above_the_cap_used():
    """Items that hit the cap have no recorded true length.

    Reporting a count above the cap used would be inventing data: those items
    wanted at least the cap and might have wanted ten times it.
    """
    curve = truncation_curve(with_lengths([100.0] * 30, cap=1024))
    assert max(r['cap'] for r in curve.rows) == 1024
    assert all(r['cap'] <= 1024 for r in curve.rows)


def test_curve_counts_are_a_partition():
    lengths = [50.0, 300.0, 700.0, 1500.0] * 10
    curve = truncation_curve(with_lengths(lengths, cap=1536))
    for r in curve.rows:
        assert r['truncated'] + r['survivors'] == curve.n_items


def test_curve_flags_a_capture_that_cannot_be_compared():
    """A candidate can only truncate more, so a doomed reference is knowable now.

    This is the eight-minute failure the curve exists to prevent: capture the
    reference, capture the candidate, then learn at compare time that the pair
    was never viable.
    """
    doomed = truncation_curve(with_lengths([2000.0] * 47 + [100.0] * 13, cap=1536))
    assert doomed.survivors == 13
    assert not doomed.usable

    fine = truncation_curve(with_lengths([100.0] * 47 + [2000.0] * 13, cap=1536))
    assert fine.survivors == 47
    assert fine.usable


def test_curve_floor_matches_compares_floor():
    """One constant, so the warning and the refusal can never disagree."""
    assert truncation_curve(with_lengths([1.0] * 30, cap=512)).floor \
        == MIN_PAIRED_ITEMS


def test_curve_requires_the_cap_it_is_relative_to():
    bare = Capture(model='m', tag='t', prompts=['p'],
                   rows=[{'ent': {'gen_len': 10.0}, 'truncated': False}])
    with pytest.raises(ValueError, match='max_tokens'):
        truncation_curve(bare)


# --- the preloaded-model extension point ----------------------------------
# capture() cannot be exercised without mlx, but its argument contract can, and
# that contract is the whole interface for patched runtimes, custom caches and
# offload wrappers. These run on a stock CI runner.

def test_capture_needs_a_model_or_a_preloaded_one():
    with pytest.raises(ValueError, match='either `model`'):
        capture(None, ['p'])


def test_capture_rejects_model_obj_without_tokenizer():
    """The tokenizer is not optional even when the model is already loaded.

    It renders the chat template and counts prompt tokens, and that count is
    what separates prompt positions from generated ones. Without it the entropy
    would be aggregated over the wrong span.
    """
    with pytest.raises(ValueError, match='without `tokenizer`'):
        capture(None, ['p'], model_obj=object())


# --- the CI contract ------------------------------------------------------
# `compare` exits non-zero on a regression so it drops into CI without glue.
# That is a promise made in the README, so it is pinned here rather than left
# to a hand-run smoke test.

def test_cli_exits_nonzero_on_regression(tmp_path):
    rng = np.random.default_rng(7)
    base = rng.normal(3.0, 1.0, 40)
    a = make(base.tolist()).save(tmp_path / 'ref.json')
    b = make((base + 1.2).tolist()).save(tmp_path / 'cand.json')
    assert cli_main(['compare', str(a), str(b)]) == 1


def test_cli_exits_zero_when_clean(tmp_path):
    rng = np.random.default_rng(8)
    base = rng.normal(3.0, 1.0, 40)
    a = make(base.tolist()).save(tmp_path / 'ref.json')
    b = make(base.tolist()).save(tmp_path / 'cand.json')
    assert cli_main(['compare', str(a), str(b)]) == 0


def test_cli_json_output_is_machine_readable(tmp_path, capsys):
    rng = np.random.default_rng(9)
    base = rng.normal(3.0, 1.0, 40)
    a = make(base.tolist()).save(tmp_path / 'ref.json')
    b = make((base + 1.2).tolist()).save(tmp_path / 'cand.json')
    cli_main(['compare', str(a), str(b), '--json'])
    out = json.loads(capsys.readouterr().out)
    assert out['verdict'] == 'REGRESSION' and out['flagged'] is True
    assert out['n_compared'] == 40


# --- the curve travels with the capture ------------------------------------
# capture() cannot run without mlx, but what it attaches must survive a save and
# a reload, because the whole point is that a caller who never touches the CLI
# still has the curve weeks later.

def test_curve_as_dict_round_trips_through_disk(tmp_path):
    # lengths must actually reach the cap, or nothing is censored and the
    # capture is perfectly usable — survivors is counted at the cap USED
    cap = with_lengths([2000.0] * 47 + [100.0] * 13, cap=1536)
    cap.meta['truncation'] = truncation_curve(cap).as_dict()
    reloaded = Capture.load(cap.save(tmp_path / 'c.json'))
    t = reloaded.meta['truncation']
    assert t['cap_used'] == 1536 and t['n_items'] == 60
    assert t['survivors'] == 13 and t['usable'] is False
    assert next(r for r in t['rows'] if r['cap'] == 512)['truncated'] == 47


def test_curve_as_dict_matches_the_curve():
    curve = truncation_curve(with_lengths([100.0] * 30, cap=1024))
    d = curve.as_dict()
    assert d['survivors'] == curve.survivors
    assert d['usable'] == curve.usable
    assert d['floor'] == MIN_PAIRED_ITEMS
