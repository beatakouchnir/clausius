# SPDX-License-Identifier: Apache-2.0
"""Answer extraction and scoring. FROZEN before any model runs.

Rules are deliberately dumb and deterministic; every change after the first
run is a dated amendment. The F-series lesson lives here: a scorer that
quietly accepts more aliases mid-program can flip a finding's sign.
"""

from __future__ import annotations

import re

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GSM_HASH_RE = re.compile(r"####\s*(-?\$?[\d,]*\.?\d+)")


def after_think(text: str) -> str:
    """Thinking models emit reasoning then </think>; score what follows."""
    return text.split("</think>")[-1]


def normalize_number(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").rstrip("%").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.strip().lower()


def extract_gsm8k(text: str) -> str | None:
    text = after_think(text)
    m = _GSM_HASH_RE.findall(text)
    if m:
        return normalize_number(m[-1])
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} with brace matching; falls back to last number."""
    text = after_think(text)
    idx = text.rfind("\\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len("\\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return normalize_math(text[start:i])
                depth -= 1
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def normalize_math(s: str) -> str:
    """Light LaTeX normalization: enough for MATH-style golds, no CAS."""
    s = s.strip().strip("$ ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    return normalize_number(s) if _NUM_RE.fullmatch(s.replace("/", "")) else s.lower()


def score_gsm8k(pred_text: str, gold: str) -> tuple[bool, str | None]:
    extracted = extract_gsm8k(pred_text)
    return (extracted is not None and extracted == normalize_number(gold), extracted)


def score_math(pred_text: str, gold: str) -> tuple[bool, str | None]:
    extracted = extract_boxed(pred_text)
    return (extracted is not None and extracted == normalize_math(gold), extracted)


_ANSWER_LINE_RE = re.compile(r"ANSWER:\s*\(?([A-D])\)?", re.I)


def extract_letter(text: str) -> str | None:
    """MCQ rule (frozen): last ANSWER: <letter> line; fallback = last
    standalone A-D within the final 80 characters."""
    text = after_think(text)
    m = _ANSWER_LINE_RE.findall(text)
    if m:
        return m[-1].upper()
    tail = text.strip()[-80:].upper()
    letters = re.findall(r"\b([A-D])\b", tail)
    return letters[-1] if letters else None


def score_mcq(pred_text: str, gold: str) -> tuple[bool, str | None]:
    extracted = extract_letter(pred_text)
    return (extracted == gold.upper(), extracted)
