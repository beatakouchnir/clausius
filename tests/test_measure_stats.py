# SPDX-License-Identifier: Apache-2.0
"""Exact McNemar + paired comparison + E14 extraction rules."""

import json

import pytest

from clausius.measure.stats import mcnemar_exact, paired_comparison


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    # 9 vs 1: two-sided exact binomial
    assert mcnemar_exact(9, 1) == pytest.approx(2 * (1 + 10) * 0.5 ** 10)
    assert mcnemar_exact(70, 41) < 0.01  # the clausius F14 shape
    assert mcnemar_exact(17, 21) > 0.5   # bf16 vs q8: not distinguishable


def test_paired_comparison(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows_a = [{"id": f"i{n}", "correct": n < 8} for n in range(10)]
    rows_b = [{"id": f"i{n}", "correct": n < 5} for n in range(10)]
    a.write_text("\n".join(json.dumps(r) for r in rows_a))
    b.write_text("\n".join(json.dumps(r) for r in rows_b))
    out = paired_comparison(a, b)
    assert out["n_pairs"] == 10
    assert out["b"] == 3 and out["c"] == 0  # items 5,6,7: A right, B wrong
    assert out["acc_a"] == 0.8 and out["acc_b"] == 0.5
