# SPDX-License-Identifier: Apache-2.0
"""The measurement engine: run a strategy over items against an OpenAI-compatible
endpoint, record one JSONL line per item (crash-safe resume), and reduce to card
rows with paired statistics. Ported from the research harness behind
docs/measurements.md (2026-08-23); the statistical rules were pre-registered
there and do not change without a dated note."""
from .cards import row_from_manifest, rows_from_dir  # noqa: F401
from .client import ChatClient, Completion  # noqa: F401
from .runner import Item, run_cell, wilson_ci  # noqa: F401
from .stats import load_manifest, mcnemar_exact, paired_comparison, synthesize_gate  # noqa: F401
