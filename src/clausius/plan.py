# SPDX-License-Identifier: Apache-2.0
"""`clausius plan` — how to run a model, from measured cells.

Every row in the packaged card data is one measured configuration: model,
quantization, task, strategy (greedy / thinking at an effort / sampling), output
cap, n, accuracy with a 95% Wilson interval, wall-clock per item, truncation rate,
and the source cell. `plan` lays the rows out per task and marks one pick per
task by a fixed rule. It never interpolates: a configuration that was not
measured is not on the card.

The pick rule is deliberately conservative. A row whose truncation rate exceeds
``CAP_BOUND`` is not eligible — its accuracy is partly a property of the cap, not
the strategy (the AIME cap ladder is the worked example: the same arm read 0.20
at 2,048 tokens and 0.55 at 8,192). Among eligible rows the pick is the highest
accuracy; a tie within ``TIE`` goes to the cheaper wall-clock.
"""

from __future__ import annotations

import json
from importlib import resources

CAP_BOUND = 0.5  # truncation rate above which a row is reported as cap-bound, not picked
TIE = 0.01  # accuracy difference treated as a tie; then cheaper wall-clock wins


def load() -> dict:
    """The packaged card data (schema 1)."""
    text = resources.files("clausius").joinpath("data/cards.json").read_text()
    return json.loads(text)


def models(data: dict | None = None) -> dict[str, int]:
    data = data or load()
    out: dict[str, int] = {}
    for r in data["rows"]:
        out[r["model"]] = out.get(r["model"], 0) + 1
    return out


def rows_for(model: str, task: str | None = None, data: dict | None = None) -> list[dict]:
    data = data or load()
    rows = [r for r in data["rows"] if r["model"] == model and (task is None or r["task"] == task)]
    return sorted(rows, key=lambda r: (r["task"], r["thinking"], r["strategy"], r["cap"]))


def label(r: dict) -> str:
    if r["thinking"]:
        eff = r.get("effort") or "default"
        return f"thinking ({eff})" if eff != "default" else "thinking"
    return r["strategy"]


def pick(rows: list[dict]) -> dict | None:
    """The recommended row among those measured for one task, or None if every row is cap-bound."""
    eligible = [r for r in rows if r["truncation"] <= CAP_BOUND]
    if not eligible:
        return None
    best = max(eligible, key=lambda r: r["accuracy"])
    near = [r for r in eligible if best["accuracy"] - r["accuracy"] <= TIE]
    return min(near, key=lambda r: r["wall_s"])


def render(model: str, task: str | None = None, data: dict | None = None) -> str:
    data = data or load()
    rows = rows_for(model, task, data)
    if not rows:
        known = ", ".join(sorted(models(data)))
        return f"no measured rows for {model!r}; models on the card: {known}"
    first = rows[0]
    lines = [f"clausius plan — {model} ({first['quant']}, {first['engine']}; {data['hardware']})",
             "  rows are measured cells, not estimates; a row marked cap-bound truncated more than "
             f"{int(CAP_BOUND * 100)}% of items and its accuracy is partly the cap's"]
    for t in sorted({r["task"] for r in rows}):
        trs = [r for r in rows if r["task"] == t]
        chosen = pick(trs)
        lines.append(f"\n  {t}")
        lines.append(f"    {'strategy':28s} {'cap':>6s} {'n':>4s} {'accuracy':>9s} {'95% CI':>16s} {'s/item':>7s} {'trunc':>6s}")
        for r in trs:
            lo, hi = r["ci95"]
            mark = "◀ pick" if r is chosen else ("cap-bound" if r["truncation"] > CAP_BOUND else "")
            lines.append(f"    {label(r):28s} {r['cap']:>6d} {r['n']:>4d} {r['accuracy']:>9.3f} "
                         f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>16s} {r['wall_s']:>7.0f} {r['truncation']:>6.2f}  {mark}")
        if chosen is None:
            lines.append("    every measured row is cap-bound — raise the cap before reading this task")
    srcs = sorted({(r["source"]["program"], r["source"]["measured"]) for r in rows})
    lines.append("\n  sources: " + "; ".join(f"{p} ({d})" for p, d in srcs))
    return "\n".join(lines)


def render_models(data: dict | None = None) -> str:
    data = data or load()
    lines = [f"models on the card (exported {data['exported']}):"]
    for m, n in sorted(models(data).items()):
        tasks = sorted({r["task"] for r in data["rows"] if r["model"] == m})
        lines.append(f"  {m:20s} {n:3d} rows  {', '.join(tasks)}")
    return "\n".join(lines)
