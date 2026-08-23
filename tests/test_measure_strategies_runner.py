# SPDX-License-Identifier: Apache-2.0
"""Strategy mechanics + runner resume, on a fake client. No network."""

import json
from dataclasses import dataclass

from clausius.measure.client import Completion
from clausius.measure.runner import run_cell, wilson_ci
from clausius.measure.scoring import extract_gsm8k
from clausius.measure.strategies import s0_greedy, s1_thinking, s2_self_consistency


@dataclass
class FakeClient:
    scripts: list  # texts, or (text, entropies) tuples

    def __post_init__(self):
        self.n_calls = 0
        self.kwargs_seen = []

    def complete(self, messages, max_tokens, temperature=0.0,
                 enable_thinking=False, signal=False, seed=None, retries=3):
        self.last_messages = messages
        self.kwargs_seen.append({"temperature": temperature,
                                 "enable_thinking": enable_thinking,
                                 "signal": signal, "seed": seed})
        item = self.scripts[self.n_calls % len(self.scripts)]
        text, ents = item if isinstance(item, tuple) else (item, None)
        self.n_calls += 1
        return Completion(text=text, prompt_tokens=10,
                          completion_tokens=len(text.split()),
                          wall_s=0.01, finish_reason="stop",
                          token_entropies=ents if signal else None)


def test_s0_is_greedy_single_call():
    c = FakeClient(["#### 42"])
    run = s0_greedy(c, "q", extract_gsm8k, 64)
    assert run.answer == "42" and c.n_calls == 1
    assert c.kwargs_seen[0]["temperature"] == 0.0
    assert not c.kwargs_seen[0]["enable_thinking"]


def test_s1_requests_thinking_and_scores_after_think():
    c = FakeClient(["bad guess 7</think>#### 42"])
    run = s1_thinking(c, "q", extract_gsm8k, 64)
    assert run.answer == "42"
    assert c.kwargs_seen[0]["enable_thinking"] is True
    assert run.meta["thought"] is True


def test_s2_majority_and_mechanism_counter():
    c = FakeClient(["#### 1", "#### 2", "#### 2", "#### 2", "#### 9"])
    run = s2_self_consistency(c, "q", extract_gsm8k, 64, k=5)
    assert run.answer == "2" and len(run.calls) == 5 and c.n_calls == 5
    assert run.meta["votes"] == {"1": 1, "2": 3, "9": 1}
    assert all(kw["temperature"] == 0.7 for kw in c.kwargs_seen)


def test_runner_resume_and_summary(tmp_path):
    from clausius.measure.runner import Item

    items = [Item(id=f"i{n}", prompt="q", gold="42") for n in range(4)]
    c = FakeClient(["#### 42", "#### 41", "#### 42", "#### 42"])

    def strat(prompt):
        return s0_greedy(c, prompt, extract_gsm8k, 64)

    def scorer(ans, gold):
        return ans == gold

    s1 = run_cell("cell", items, strat, scorer, tmp_path, progress=lambda *a: None)
    assert s1["n"] == 4 and s1["accuracy"] == 0.75
    assert s1["truncation_rate"] == 0.0
    calls_after_first = c.n_calls

    # resume: nothing re-runs
    s2 = run_cell("cell", items, strat, scorer, tmp_path, progress=lambda *a: None)
    assert c.n_calls == calls_after_first
    assert s2["accuracy"] == 0.75
    # manifest holds exactly one record per item
    lines = (tmp_path / "cell.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert {json.loads(ln)["id"] for ln in lines} == {i.id for i in items}


def test_wilson_sane():
    lo, hi = wilson_ci(75, 100)
    assert 0.65 < lo < 0.75 < hi < 0.84


def test_s3b_picks_min_entropy_sample():
    from clausius.measure.strategies import s3b_best_of_n_entropy

    c = FakeClient([("#### 1", [2.0, 2.0]), ("#### 2", [0.1, 0.1]),
                    ("#### 3", [1.0, 1.0])])
    run = s3b_best_of_n_entropy(c, "q", extract_gsm8k, 64, n=3)
    assert run.answer == "2" and run.meta["picked"] == 1
    assert all(kw["signal"] for kw in c.kwargs_seen)
    seeds = [kw["seed"] for kw in c.kwargs_seen]
    assert len(set(seeds)) == 3 and all(s is not None for s in seeds)


def test_s0_signal_records_entropy_meta():
    c = FakeClient([("#### 42", [0.5, 1.5])])
    run = s0_greedy(c, "q", extract_gsm8k, 64, signal=True)
    assert run.meta["mean_entropy"] == 1.0 and run.meta["max_entropy"] == 1.5


def test_synthesize_gate_sweep(tmp_path):
    import json

    from clausius.measure.stats import synthesize_gate

    s0 = tmp_path / "s0.jsonl"
    esc = tmp_path / "esc.jsonl"
    rows0, rowse = [], []
    # 4 items: low-entropy ones S0 gets right; high-entropy ones S0 wrong,
    # escalation fixes two of them
    spec = [("i0", 0.2, True, True), ("i1", 0.3, True, False),
            ("i2", 2.0, False, True), ("i3", 3.0, False, True)]
    for iid, ent, ok0, oke in spec:
        rows0.append({"id": iid, "correct": ok0, "wall_s": 1.0,
                      "meta": {"mean_entropy": ent}})
        rowse.append({"id": iid, "correct": oke, "wall_s": 5.0, "meta": {}})
    s0.write_text("\n".join(json.dumps(r) for r in rows0))
    esc.write_text("\n".join(json.dumps(r) for r in rowse))

    sweep = synthesize_gate(s0, esc, [10.0, 1.0, 0.0])
    never, mid, always = sweep
    assert never["gate_rate"] == 0.0 and never["accuracy"] == 0.5
    assert never["mean_wall_s"] == 1.0
    assert mid["gate_rate"] == 0.5 and mid["accuracy"] == 1.0
    assert mid["mean_wall_s"] == 3.5   # all pay S0; half add escalation
    assert always["gate_rate"] == 1.0 and always["accuracy"] == 0.75
    assert always["mean_wall_s"] == 6.0
