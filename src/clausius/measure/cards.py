# SPDX-License-Identifier: Apache-2.0
"""Manifests -> card rows (the rows `clausius plan` reads).

A card row is one measured configuration. Nothing is estimated: accuracy and its
Wilson interval, wall per item, truncation rate, and mean tokens out are
reductions of the per-item records; model, quantization, task, strategy, and cap
are supplied by the caller, because the manifest does not know them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .runner import wilson_ci


def row_from_manifest(path: str | Path, *, model: str, task: str, strategy: str, cap: int,
                      thinking: bool = False, effort: str | None = None, k: int | None = None,
                      quant: str = "", engine: str = "", checkpoint: str = "",
                      program: str = "", expected_n: int | None = None) -> dict | None:
    """One card row, or None if the manifest is empty or shorter than expected_n
    (a cell still running is not a result)."""
    path = Path(path)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(recs)
    if n == 0 or (expected_n and n < expected_n):
        return None
    k_correct = sum(bool(r["correct"]) for r in recs)
    lo, hi = wilson_ci(k_correct, n)
    return {
        "model": model, "checkpoint": checkpoint, "quant": quant, "engine": engine,
        "task": task, "strategy": strategy, "thinking": thinking, "effort": effort, "k": k, "cap": cap,
        "n": n, "accuracy": round(k_correct / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "wall_s": round(sum(r["wall_s"] for r in recs) / n, 1),
        "truncation": round(sum(bool(r.get("truncated")) for r in recs) / n, 4),
        "tokens_out": round(sum(r["tokens_out"] for r in recs) / n),
        "source": {"program": program, "cell": path.stem, "dir": path.parent.name,
                   "measured": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime)), "partial": False},
    }


def rows_from_dir(path: str | Path, decode, **common) -> list[dict]:
    """Every manifest in a directory that `decode(cell_name)` maps to
    (task, strategy, thinking, effort, k, cap); others are skipped."""
    rows = []
    for p in sorted(Path(path).glob("*.jsonl")):
        dec = decode(p.stem)
        if not dec:
            continue
        task, strategy, thinking, effort, k, cap = dec
        row = row_from_manifest(p, task=task, strategy=strategy, thinking=thinking, effort=effort, k=k, cap=cap, **common)
        if row:
            rows.append(row)
    return rows
