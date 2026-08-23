"""Manifest -> card row: the reduction `clausius plan` trusts."""
import json

from clausius.measure.cards import row_from_manifest, rows_from_dir


def _manifest(path, n, correct_n, truncated_n):
    rows = [{"id": f"i{k}", "correct": k < correct_n, "truncated": k < truncated_n, "wall_s": 10.0 + k, "tokens_out": 100 + k} for k in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_row_reduces_the_manifest(tmp_path):
    m = tmp_path / "s0sig-math.jsonl"; _manifest(m, 20, 15, 4)
    r = row_from_manifest(m, model="m", task="math500-l5", strategy="greedy", cap=2048, program="t")
    assert r["n"] == 20 and r["accuracy"] == 0.75 and r["truncation"] == 0.2
    lo, hi = r["ci95"]; assert lo < 0.75 < hi
    assert r["wall_s"] == 19.5 and r["tokens_out"] == 110 and r["source"]["cell"] == "s0sig-math"


def test_partial_cells_are_not_rows(tmp_path):
    m = tmp_path / "c.jsonl"; _manifest(m, 5, 3, 0)
    assert row_from_manifest(m, model="m", task="t", strategy="greedy", cap=1, expected_n=10) is None
    (tmp_path / "empty.jsonl").write_text("")
    assert row_from_manifest(tmp_path / "empty.jsonl", model="m", task="t", strategy="greedy", cap=1) is None


def test_rows_from_dir_uses_the_decoder(tmp_path):
    _manifest(tmp_path / "s0-a.jsonl", 4, 2, 0); _manifest(tmp_path / "junk.jsonl", 4, 2, 0)
    dec = lambda name: ("task-a", "greedy", False, None, None, 512) if name == "s0-a" else None  # noqa: E731
    rows = rows_from_dir(tmp_path, dec, model="m", program="t")
    assert [r["source"]["cell"] for r in rows] == ["s0-a"] and rows[0]["cap"] == 512
