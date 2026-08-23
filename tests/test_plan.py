"""`clausius plan`: the packaged card data is well-formed, the pick rule is the
documented one, and the CLI renders without a model loaded."""
import json
import subprocess
import sys

import pytest

from clausius import plan

REQUIRED = {"model", "quant", "engine", "task", "strategy", "thinking", "cap", "n",
            "accuracy", "ci95", "wall_s", "truncation", "tokens_out", "source"}


def test_card_data_is_well_formed():
    data = plan.load()
    assert data["schema"] == 1 and data["rows"]
    for r in data["rows"]:
        assert REQUIRED <= set(r), r
        lo, hi = r["ci95"]
        assert 0 <= lo <= r["accuracy"] <= hi <= 1
        assert 0 <= r["truncation"] <= 1 and r["n"] > 0 and r["cap"] > 0
        assert r["source"]["cell"] and r["source"]["measured"]


def test_pick_prefers_accuracy_then_cost_and_skips_cap_bound():
    rows = [
        {"accuracy": 0.80, "wall_s": 100, "truncation": 0.10},
        {"accuracy": 0.805, "wall_s": 300, "truncation": 0.10},  # within TIE of the first, costlier
        {"accuracy": 0.95, "wall_s": 10, "truncation": 0.70},    # cap-bound: never picked
    ]
    assert plan.pick(rows) is rows[0]
    assert plan.pick([rows[2]]) is None


def test_pick_is_highest_accuracy_outside_the_tie():
    rows = [{"accuracy": 0.60, "wall_s": 5, "truncation": 0.0},
            {"accuracy": 0.70, "wall_s": 50, "truncation": 0.0}]
    assert plan.pick(rows) is rows[1]


def test_render_marks_one_pick_per_task_and_the_aime_cap_ladder():
    text = plan.render("qwen3.5-35b-a3b")
    assert "aime-2024-25" in text and "math500-l5" in text
    # the 2048-cap AIME greedy row truncated 80% and must be reported cap-bound, never picked
    line = next(l for l in text.splitlines() if "aime" not in l and " 2048 " in l and "greedy" in l and "0.200" in l)
    assert "cap-bound" in line and "pick" not in line


def test_unknown_model_lists_known_ones():
    text = plan.render("no-such-model")
    assert "no measured rows" in text and "qwen3.8-27b" in text


@pytest.mark.parametrize("args", [[], ["--model", "qwen3.8-27b"], ["--model", "qwen3.8-27b", "--json"]])
def test_cli_plan_runs_without_a_model(args):
    r = subprocess.run([sys.executable, "-m", "clausius.cli", "plan", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    if "--json" in args:
        assert json.loads(r.stdout)["rows"]
    else:
        assert "qwen3.8-27b" in r.stdout
