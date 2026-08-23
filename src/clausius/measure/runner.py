# SPDX-License-Identifier: Apache-2.0
"""Cell runner: per-item JSONL manifests, resume, honest summaries.

A cell = (task, strategy, model/endpoint). Every item lands as one JSONL
record the moment it finishes (crash-safe resume by item id — the E14
manifest lesson). The summary reports truncation as a first-class number,
never as silence.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    """One measured item: a stable id (manifests resume by it), the prompt, the gold."""

    id: str
    prompt: str
    gold: str


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_cell(cell_name: str, items, strategy_fn, scorer, out_dir: str | Path,
             progress=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"{cell_name}.jsonl"

    done = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
        if done:
            progress(f"[{cell_name}] resuming: {len(done)} items already done")

    t_start = time.perf_counter()
    with open(manifest, "a") as sink:
        for i, item in enumerate(items):
            if item.id in done:
                continue
            run = strategy_fn(item.prompt)
            correct, extracted = (False, run.answer)
            if run.answer is not None:
                correct = scorer(run.answer, item.gold)
            rec = {
                "id": item.id,
                "gold": item.gold,
                "answer": run.answer,
                "extracted": extracted,
                "correct": bool(correct),
                "n_calls": len(run.calls),
                "wall_s": round(run.wall_s, 3),
                "tokens_out": run.tokens_out,
                "tokens_prompt": run.calls[0].prompt_tokens if run.calls else 0,
                "truncated": run.truncated,
                "meta": run.meta,
            }
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            done[item.id] = rec
            if (i + 1) % 10 == 0:
                progress(f"[{cell_name}] {i + 1}/{len(items)}")

    recs = [done[it.id] for it in items if it.id in done]
    n = len(recs)
    k = sum(r["correct"] for r in recs)
    lo, hi = wilson_ci(k, n)
    walls = sorted(r["wall_s"] for r in recs)
    summary = {
        "cell": cell_name,
        "n": n,
        "accuracy": round(k / n, 4) if n else None,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "mean_wall_s": round(sum(walls) / n, 3) if n else None,
        "p50_wall_s": round(walls[n // 2], 3) if n else None,
        "mean_tokens_out": round(sum(r["tokens_out"] for r in recs) / n, 1) if n else None,
        "truncation_rate": round(sum(r["truncated"] for r in recs) / n, 4) if n else None,
        "mean_calls": round(sum(r["n_calls"] for r in recs) / n, 2) if n else None,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / f"{cell_name}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    progress(f"[{cell_name}] acc={summary['accuracy']} "
             f"ci={summary['acc_ci95']} wall={summary['mean_wall_s']}s "
             f"trunc={summary['truncation_rate']}")
    return summary
