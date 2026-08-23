# clausius — measure the models you really run

[![ci](https://github.com/beatakouchnir/clausius/actions/workflows/test.yml/badge.svg)](https://github.com/beatakouchnir/clausius/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/clausius)](https://pypi.org/project/clausius/)
[![license](https://img.shields.io/pypi/l/clausius)](LICENSE)

You changed something — the quantization, a config flag, a fine-tune — and you want to know whether it broke the model. `clausius` answers that on your own prompts, with no labels, in three commands. It exits non-zero on a regression, so it drops into CI without glue.

## Install

```bash
pip install "clausius[mlx]"        # capture + compare (Apple Silicon)
pip install clausius               # compare and analysis only — pure numpy, runs anywhere
```

## Run it

Take 60 prompts of your own — production traffic is ideal, no labels needed — or the 60 in [`examples/prompts.jsonl`](https://github.com/beatakouchnir/clausius/blob/main/examples/prompts.jsonl). Capture the configuration you trust, capture the one you changed, compare:

```bash
clausius capture --model ./gemma-26b-a4b-4bit --prompts examples/prompts.jsonl --out ref.json  --max-tokens 1536
clausius capture --model ./gemma-26b-a4b-2bit --prompts examples/prompts.jsonl --out cand.json --max-tokens 1536
clausius compare ref.json cand.json --show 3
```

```
REGRESSION  (max d_z = +5.922, threshold 0.3, one-sided)
  compared 20 paired items, dropped 5 truncated
  all signals: max +5.92  p90 +11.64  mean +10.43  mean_top10 +9.00  first +2.80  gen_len +10.55
```

![clausius compare on real captures: the 2-bit config is flagged REGRESSION with the ref's coherent reasoning shown against the 2-bit model's garbled output](https://raw.githubusercontent.com/beatakouchnir/clausius/main/docs/compare.gif)

That is a real run on 25 unlabeled prompts. The two checkpoints differ only in quantization; the 2-bit one independently measures 73 points lower on instruction adherence. `--show 3` prints the items whose entropy moved most, with the text both configurations produced — the verdict says something broke, this says what.

**First time?** [Try it in 30 minutes](https://github.com/beatakouchnir/clausius/blob/main/USAGE.md#try-it-in-30-minutes): three public checkpoints, five commands, measured output included. **Then read [USAGE.md](https://github.com/beatakouchnir/clausius/blob/main/USAGE.md)**: choosing prompts, setting the token cap, reading `d_z` and its interval, calibrating your own null, running it as a CI gate.

## What the verdict means

- The signal is the change in the model's predictive entropy over its own answers, paired prompt by prompt between the two captures (`d_z`).
- The defaults are measured, not chosen: the 0.3 threshold comes from 13 configurations known to be harmless; the one-sided test from a construction that fools a two-sided one; the truncation filter from an effect that doubles once applied. `src/clausius/core.py` states each one and [the findings record](https://github.com/beatakouchnir/clausius/blob/main/the findings record) has the evidence.
- It is more sensitive than labels on the damage it was validated against: a −2.2pp quantization regression was flagged on 60 unlabeled prompts where a paired test on gold labels needed n=878 (F14).
- It is a regression check, not a score: it needs a reference capture and cannot rate a configuration in isolation.

## Limits

- Sensitivity is the weaker half. Specificity is 13/13; a −5.7pp configuration is missed at any threshold that preserves that record.
- Confidence-increasing damage would be invisible to a one-sided detector. Three mechanisms were built to produce it and none did, but it is not excluded.
- `d_z` is ordinal, not proportional: 3-bit loses 25× more accuracy than 4-bit and reads 2× the `d_z`. "Something moved, and roughly how hard" is supportable; "you lost k points" is not.
- The threshold is calibrated on one stack (MLX, Apple Silicon). On another framework, device, or quantizer, measure your own null first — [USAGE.md](https://github.com/beatakouchnir/clausius/blob/main/USAGE.md#calibrating-your-own-null) has the recipe.
- Local and self-hosted models only: hosted APIs expose no logprobs, or truncated ones, which is a different quantity.

Capture targets Apple Silicon via MLX in this release. An experimental PyTorch backend exists and is deliberately unshipped: it has been measured on mps and cpu, never calibrated on cuda or the CUDA-native quantizers, and shipping an uncalibrated threshold would contradict what this package claims about its defaults. It is archived at the tag `archive/torch-backend` rather than kept as a live branch, and will be revisited if a user needs it or a contribution calls for it; the bar it must clear is in [the experiment log](https://github.com/beatakouchnir/clausius/blob/main/the experiment log).

## Receipts

The numbers behind the defaults, measured on consumer hardware (M5 Max, 128 GB) across five model families, with negatives kept: eight controlled failures and seven corrections to the record. Headline rows:

| | result | where |
|---|---|---|
| Label-free detection works | validated against five unrelated damage mechanisms whose true damage was measured independently; benign controls stay clean (a 3.3× memory reduction changes ~25% of generations textually and moves no signal) | F8 |
| More sensitive than labels | the −2.2pp regression: 60 unlabeled prompts, where labels needed n=878 | F14 |
| Short benchmarks understate damage ~14× | factual QA loses 1.5pp where structured generation loses 18–21pp under the same quantization | F11 |
| Offload beats downsizing | an offloaded 35B at 3.40 GB scores 0.9447 on gsm8k against a natively-fitting 4B at 3.91 GB scoring 0.8426; the whole cost is latency | F3–F6 |

Everything else — the quantization ladder and frontier chart, the claim taxonomy, what did not work, the seven corrections, prior art — is on the [receipts page](https://github.com/beatakouchnir/clausius/blob/main/docs/receipts.md); the full record is [the findings record](https://github.com/beatakouchnir/clausius/blob/main/the findings record). The corpus is committed, so every table rebuilds on a laptop with no model and no accelerator.

## Where this is going

v0.2 adds the other half of "measure the models you really run": report cards — the benchmarks official model cards report, re-measured across the quantizations and configurations people actually deploy, with paired statistics and truncation rates — and `clausius plan`, per-model advice on thinking, reasoning effort, and output caps, generated from measured cells. The plan and its sequencing are in [docs/v02_plan.md](https://github.com/beatakouchnir/clausius/blob/main/docs/v02_plan.md).

**Sibling.** [boyle](https://github.com/beatakouchnir/boyle) runs the model you want at the memory pressure you specify — budgeted MoE inference with speed forecasts before you download. Its `predict` cites this repository's measured accuracy.

*Named for Rudolf Clausius, who coined the word entropy in 1865. Entropy is the signal this tool reads.*

## Repository layout

| path | what | needs |
|---|---|---|
| `src/clausius/` | the tool — capture, compare, CLI | numpy; mlx-lm only to capture |
| `tests/` | 36 tests, none load a model; CI installs the built wheel | numpy |
| `records/` | the measurement corpus behind the findings record, ~10 MB | — |
| `knowledge/` | the research package that produced the findings; not packaged | local checkpoints, `CLAUSIUS_ARTIFACTS` |
| `USAGE.md` | the operating manual | — |
| `the findings record` · `the experiment log` · `docs/receipts.md` | the record: results, designs, what was deliberately not built | — |
