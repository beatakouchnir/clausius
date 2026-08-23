# SPDX-License-Identifier: Apache-2.0
"""The frozen extraction rules, table-driven. These tests ARE the freeze."""

import pytest

from clausius.measure.scoring import (
    after_think,
    extract_boxed,
    extract_gsm8k,
    normalize_math,
    score_gsm8k,
    score_math,
)


@pytest.mark.parametrize("text,gold,ok", [
    ("... so the total is 42.\n#### 42", "42", True),
    ("#### 1,234", "1234", True),
    ("#### $18", "18", True),
    ("#### 3.0", "3", True),
    ("the answer is 7", "7", True),            # fallback: last number
    ("prices were 5 then 9", "5", False),      # last number rule is strict
    ("no numbers here", "4", False),
])
def test_gsm8k_rules(text, gold, ok):
    assert score_gsm8k(text, gold)[0] is ok


@pytest.mark.parametrize("text,gold,ok", [
    ("thus \\boxed{42}", "42", True),
    ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}", True),
    ("\\boxed{2\\sqrt{3}}", "2\\sqrt{3}", True),
    ("nested \\boxed{\\text{west}}", "west", True),
    ("first \\boxed{1} then \\boxed{2}", "2", True),   # last boxed wins
    ("unboxed answer 17", "17", True),                 # fallback
    ("\\boxed{0.5}", "\\frac{1}{2}", False),           # no CAS: strings differ, recorded
])
def test_math_rules(text, gold, ok):
    assert score_math(text, gold)[0] is ok


def test_think_block_is_ignored():
    text = "I think 99 is wrong.</think>\n#### 12"
    assert score_gsm8k(text, "12")[0]
    assert extract_gsm8k(text) == "12"
    assert after_think("a</think>b") == "b"
    assert after_think("no think") == "no think"


def test_boxed_brace_matching():
    assert extract_boxed("\\boxed{{a}+{b}}") == "{a}+{b}"


def test_normalize_math_forms():
    assert normalize_math("\\frac{3}{4}") == "3/4"
    assert normalize_math("$ 42 $") == "42"
    assert normalize_math("\\text{west}") == "west"


@pytest.mark.parametrize("text,gold,ok", [
    ("reasoning...\nANSWER: B", "B", True),
    ("ANSWER: (c)", "C", True),
    ("ANSWER: A\nwait, ANSWER: D", "D", True),      # last wins
    ("I believe the answer is B", "B", True),        # tail fallback
    ("options A and B are wrong so C", "C", True),   # last standalone letter
    ("no letters here at all", "A", False),
])
def test_mcq_rules(text, gold, ok):
    from clausius.measure.scoring import score_mcq
    assert score_mcq(text, gold)[0] is ok
