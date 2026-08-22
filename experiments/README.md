# `experiments/` — the run history

These are the chain scripts that actually produced [`records/`](../records), kept as provenance rather than as an interface. If you want to know how a number in [FINDINGS.md](../FINDINGS.md) was generated, the script that generated it is here.

They are **not** a supported entry point. They assume local checkpoints, a external artifact store, and a Mac with enough unified memory, and several of them are supersets or repairs of earlier ones: `run_frontier5_topk_rerun.sh` re-runs an arm that `run_frontier3_topk.sh` got wrong. The numbering is chronological, not a hierarchy.

```bash
export QV=/path/to/python-with-mlx-lm   # defaults to `python`
export CLAUSIUS_ARTIFACTS=../artifacts
sh experiments/run_frontier1_exactness.sh
```

Two conventions in here are deliberate and worth stealing:

**Bounded waits, never `pgrep`.** An early overnight chain blocked on `while pgrep -f "stage_a --task gpqa"` — and the waiter's own command line matched the pattern, so it would have spun until morning without running anything. Every chain since waits on a marker in the previous chain's log with a hard iteration cap, so a chain that never finishes cannot hang the next one indefinitely.

**Every step is isolated.** A failing arm logs and the chain continues, because losing one measurement overnight is much cheaper than losing all of them.
