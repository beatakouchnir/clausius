# SPDX-License-Identifier: Apache-2.0
"""Strategy arms S0–S2. Configs were frozen before any model ran.

Every strategy returns the full call record — the refuse-to-fake rule:
the runner asserts the mechanism actually ran (k samples drawn means k
call records with tokens, not a wrapper that no-op'd).
"""

from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass, field

from clausius.measure.client import ChatClient, Completion


@dataclass
class StrategyRun:
    answer: str | None
    calls: list[Completion] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def wall_s(self) -> float:
        return sum(c.wall_s for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def truncated(self) -> bool:
        return any(c.truncated for c in self.calls)


def _messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _sample_seed(prompt: str, i: int) -> int:
    """Stable per-(item, sample) seed: reproducible cells, diverse samples.
    Exists because the serving path once returned IDENTICAL draws at any
    temperature (the void-cells incident, a dated amendment)."""
    return (zlib.crc32(prompt.encode()) ^ (i * 0x9E3779B1)) & 0x7FFFFFFF


def s0_greedy(client: ChatClient, prompt: str, extract, max_tokens: int,
              signal: bool = False) -> StrategyRun:
    c = client.complete(_messages(prompt), max_tokens=max_tokens,
                        temperature=0.0, signal=signal)
    meta = {}
    if signal and c.token_entropies:
        meta = {"mean_entropy": round(c.mean_entropy, 5),
                "max_entropy": round(max(c.token_entropies), 5)}
    return StrategyRun(answer=extract(c.text), calls=[c], meta=meta)


def s1_thinking(client: ChatClient, prompt: str, extract, max_tokens: int) -> StrategyRun:
    c = client.complete(_messages(prompt), max_tokens=max_tokens,
                        temperature=0.0, enable_thinking=True)
    think_tokens = None
    if "</think>" in c.text:
        # crude but honest share estimate: chars scale with tokens well enough
        # for a share metric; exact split lands with tokenizer-side metering
        pre = c.text.split("</think>")[0]
        think_tokens = round(c.completion_tokens * len(pre) / max(1, len(c.text)))
    return StrategyRun(answer=extract(c.text), calls=[c],
                       meta={"think_tokens_est": think_tokens,
                             "thought": "</think>" in c.text})


def s2_self_consistency(client: ChatClient, prompt: str, extract,
                        max_tokens: int, k: int = 5) -> StrategyRun:
    calls, answers = [], []
    for i in range(k):
        c = client.complete(_messages(prompt), max_tokens=max_tokens,
                            temperature=0.7, seed=_sample_seed(prompt, i))
        calls.append(c)
        a = extract(c.text)
        if a is not None:
            answers.append(a)
    assert len(calls) == k, f"mechanism check: drew {len(calls)} of {k} samples"
    if not answers:
        return StrategyRun(answer=None, calls=calls, meta={"k": k, "votes": {}})
    votes = Counter(answers)
    top, top_n = votes.most_common(1)[0]
    return StrategyRun(answer=top, calls=calls,
                       meta={"k": k, "votes": dict(votes), "margin": top_n / k})


JUDGE_TEMPLATE = (
    "You are grading candidate answers to a problem.\n\nProblem:\n{prompt}"
    "\n\nCandidates:\n{candidates}\n\nReply with only the number of the "
    "best candidate, in the form: BEST: <number>"
)


def s3a_best_of_n_judge(client: ChatClient, prompt: str, extract,
                        max_tokens: int, n: int = 5) -> StrategyRun:
    import re

    calls = [client.complete(_messages(prompt), max_tokens=max_tokens,
                             temperature=0.7, seed=_sample_seed(prompt, i))
             for i in range(n)]
    assert len(calls) == n
    cand_block = "\n".join(f"[{i + 1}] {c.text.strip()[:800]}"
                            for i, c in enumerate(calls))
    judge = client.complete(
        _messages(JUDGE_TEMPLATE.format(prompt=prompt, candidates=cand_block)),
        max_tokens=16, temperature=0.0)
    m = re.findall(r"BEST:\s*(\d+)", judge.text)
    pick = int(m[-1]) - 1 if m and 0 < int(m[-1]) <= n else 0
    return StrategyRun(answer=extract(calls[pick].text), calls=calls + [judge],
                       meta={"n": n, "picked": pick, "judge_parsed": bool(m)})


def s3b_best_of_n_entropy(client: ChatClient, prompt: str, extract,
                          max_tokens: int, n: int = 5) -> StrategyRun:
    calls = [client.complete(_messages(prompt), max_tokens=max_tokens,
                             temperature=0.7, signal=True,
                             seed=_sample_seed(prompt, i)) for i in range(n)]
    assert len(calls) == n
    assert all(c.token_entropies for c in calls), "signal missing — endpoint lacks token_entropies"
    assert len({c.text for c in calls}) > 1 or n == 1, (
        "all samples identical — sampling determinism regression (A3)")
    pick = min(range(n), key=lambda i: calls[i].mean_entropy)
    return StrategyRun(answer=extract(calls[pick].text), calls=calls,
                       meta={"n": n, "picked": pick,
                             "entropies": [round(c.mean_entropy, 5) for c in calls]})


# -- S6: self-refine. Prompts FROZEN here. Greedy throughout. On Qwen3.5's
# hybrid cache every round pays re-prefill — the meter reports that real
# cost via wall_s; no discounting.

S6_CRITIQUE_TEMPLATE = (
    "Below is a problem and a draft solution. Critique the draft: identify "
    "any errors in reasoning or the final answer. Be specific. If you find "
    "no errors, reply with exactly: NO ISSUES\n\n"
    "## Problem\n{problem}\n\n## Draft solution\n{draft}"
)
S6_REVISE_TEMPLATE = (
    "Below is a problem, a draft solution, and a critique of that draft. "
    "Write a corrected, complete solution. Keep the same answer format the "
    "problem asks for.\n\n## Problem\n{problem}\n\n## Draft solution\n"
    "{draft}\n\n## Critique\n{critique}"
)


def s6_self_refine(client: ChatClient, prompt: str, extract,
                   max_tokens: int, rounds: int = 2) -> StrategyRun:
    calls = []
    draft_c = client.complete(_messages(prompt), max_tokens=max_tokens,
                              temperature=0.0)
    calls.append(draft_c)
    draft = draft_c.text
    rounds_run = 0
    stopped_clean = False
    for _ in range(rounds):
        crit_c = client.complete(
            _messages(S6_CRITIQUE_TEMPLATE.format(problem=prompt, draft=draft)),
            max_tokens=max_tokens, temperature=0.0)
        calls.append(crit_c)
        rounds_run += 1
        if "NO ISSUES" in crit_c.text.strip()[:200]:
            stopped_clean = True
            break
        rev_c = client.complete(
            _messages(S6_REVISE_TEMPLATE.format(problem=prompt, draft=draft,
                                                critique=crit_c.text)),
            max_tokens=max_tokens, temperature=0.0)
        calls.append(rev_c)
        draft = rev_c.text
    assert rounds_run >= 1, "refine loop never ran (mechanism check)"
    return StrategyRun(answer=extract(draft), calls=calls,
                       meta={"rounds": rounds_run,
                             "stopped_clean": stopped_clean,
                             "revised": len(calls) > 2})


# -- S5: tool-augmented single pass. One generation; if it contains ```python blocks and no final
# answer yet, the LAST block is executed and its output returned for ONE
# continuation (max 2 executions). Greedy. Execution: subprocess, 6 s.

S5_TOOL_PROMPT = (
    "You may use Python to compute intermediate results. Write code in a "
    "```python block that prints what you need; after the block, STOP — "
    "the code's output will be provided to you. When you have the final "
    "answer, state it in the format the problem asks for.\n\n{problem}"
)
S5_RESULT_TEMPLATE = "{sofar}\n\n[python output]\n{output}\n\nContinue."


def _run_python(code: str, timeout_s: float = 6.0) -> str:
    import subprocess
    import sys

    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             timeout=timeout_s)
        return (out.stdout + out.stderr).strip()[:2000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "(execution timed out)"


def s5_tool_exec(client: ChatClient, prompt: str, extract,
                 max_tokens: int, max_execs: int = 2) -> StrategyRun:
    import re

    fence = re.compile(r"```python\n(.*?)```", re.S)
    calls, execs = [], 0
    msg = S5_TOOL_PROMPT.format(problem=prompt)
    c = client.complete(_messages(msg), max_tokens=max_tokens,
                        temperature=0.0)
    calls.append(c)
    text = c.text
    while execs < max_execs:
        blocks = fence.findall(text)
        # The frozen prompt instructs: write a block, then STOP. So the
        # execution trigger is "the response ends at a code block" — an
        # extraction-based check misfires because task extractors have
        # aggressive fallbacks that find "answers" inside code.
        tail = text.rsplit("```", 1)[-1] if blocks else ""
        wants_exec = bool(blocks) and len(tail.strip()) < 40
        if not wants_exec:
            break
        result = _run_python(blocks[-1])
        execs += 1
        msg = S5_RESULT_TEMPLATE.format(sofar=msg + "\n\n" + text,
                                        output=result)
        c = client.complete(_messages(msg), max_tokens=max_tokens,
                            temperature=0.0)
        calls.append(c)
        text = c.text
    return StrategyRun(answer=extract(text), calls=calls,
                       meta={"tool_execs": execs,
                             "used_tool": execs > 0})


def s_effort(client: ChatClient, prompt: str, extract, max_tokens: int,
             effort: str | None) -> StrategyRun:
    """Thinking at a template-defined reasoning effort (Qwen3.8: low /
    medium / xhigh via chat_template_kwargs), or thinking off when effort is
    None. Records the full response text and the per-token entropy trace
    in meta — the instrumentation the measurement harness's Phase 1 cells lacked (the mechanism research
    probe 1 could not locate where a runaway chain first reached its
    answer). Greedy, signal on."""
    kwargs = {"reasoning_effort": effort} if effort else None
    c = client.complete(_messages(prompt), max_tokens=max_tokens,
                        temperature=0.0, enable_thinking=effort is not None,
                        signal=True, template_kwargs=kwargs)
    thought = "</think>" in c.text
    think_tokens = None
    if thought:
        pre = c.text.split("</think>")[0]
        think_tokens = round(c.completion_tokens * len(pre) / max(1, len(c.text)))
    ents = c.token_entropies or []
    return StrategyRun(answer=extract(c.text), calls=[c], meta={
        "effort": effort, "thought": thought, "think_tokens_est": think_tokens,
        "mean_entropy": round(c.mean_entropy, 5) if ents else None,
        "max_entropy": round(max(ents), 5) if ents else None,
        "text": c.text,
        "token_entropies": [round(e, 3) for e in ents],
    })
