# SPDX-License-Identifier: Apache-2.0
"""Paired statistics for strategy comparisons.

McNemar over discordant pairs is the workhorse: strategies run the same
items, so the informative cells are b (A right, B wrong) and c (the
reverse). Exact two-sided binomial — no chi-square approximations at
n=100-cell sizes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p over the b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def load_manifest(path: str | Path) -> dict[str, dict]:
    recs = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        recs[r["id"]] = r
    return recs


def paired_comparison(manifest_a: str | Path, manifest_b: str | Path) -> dict:
    """b = A correct & B wrong; c = reverse; exact McNemar p."""
    a, bm = load_manifest(manifest_a), load_manifest(manifest_b)
    ids = sorted(set(a) & set(bm))
    b = sum(1 for i in ids if a[i]["correct"] and not bm[i]["correct"])
    c = sum(1 for i in ids if bm[i]["correct"] and not a[i]["correct"])
    return {
        "n_pairs": len(ids),
        "acc_a": round(sum(a[i]["correct"] for i in ids) / len(ids), 4),
        "acc_b": round(sum(bm[i]["correct"] for i in ids) / len(ids), 4),
        "b": b,
        "c": c,
        "p": round(mcnemar_exact(b, c), 4),
    }


def synthesize_gate(s0_manifest: str | Path, esc_manifest: str | Path,
                    thresholds: list[float]) -> list[dict]:
    """S4 constructed from recorded cells: the gate reads S0's
    per-item mean entropy; above threshold, the item escalates — outcome
    and cost swap to the escalation arm's record, plus S0's cost (the gate
    always pays the first pass). One S0-with-signal cell + one escalation
    cell yield the whole threshold sweep."""
    s0 = load_manifest(s0_manifest)
    esc = load_manifest(esc_manifest)
    ids = sorted(set(s0) & set(esc))
    missing = [i for i in ids if "mean_entropy" not in (s0[i].get("meta") or {})]
    assert not missing, f"S0 records lack the signal: {missing[:3]}"
    out = []
    for th in thresholds:
        fired = correct = 0
        wall = 0.0
        for i in ids:
            e = s0[i]["meta"]["mean_entropy"]
            wall += s0[i]["wall_s"]
            if e > th:
                fired += 1
                correct += esc[i]["correct"]
                wall += esc[i]["wall_s"]
            else:
                correct += s0[i]["correct"]
        out.append({"threshold": th, "n": len(ids),
                    "gate_rate": round(fired / len(ids), 4),
                    "accuracy": round(correct / len(ids), 4),
                    "mean_wall_s": round(wall / len(ids), 3)})
    return out
