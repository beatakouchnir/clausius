# clausius — did the thing you changed break your model?

Two questions, both answered on consumer hardware (M5 Max, 128 GB), across five
model families:

1. **If a model does not fit in your memory, what should you actually run?**
2. **If you change anything — config, quantization, fine-tune — how would you
   know it broke the model, without a labeled eval set?**

Full evidence, positives *and* negatives, in **[FINDINGS.md](https://github.com/beatakouchnir/clausius/blob/main/FINDINGS.md)** —
which opens with a prior-art accounting stating which parts independently
re-derive published work and which appear to be new.

## What's new here

- **Label-free regression detection, checked against independently measured
  damage.** Predictive entropy is well studied; using it to detect that *a
  deployment change broke the model* — validated against five unrelated damage
  mechanisms whose true cost was measured separately — appears not to be. It
  flagged a −2.2pp quantization regression on **60 unlabeled prompts**, where
  paired McNemar on gold labels needed **n=878** to reach p<0.05 (F14).
- **Accuracy measured across MoE offload configurations at all** — four surveyed
  implementations report none. It inverts the default advice: at matched memory,
  aggressive quantization costs **73 points** of instruction adherence where
  exact expert offload costs **1.0** (§1, §4).
- **The corpus is committed, so the tables below rebuild on a laptop** — no
  model, no accelerator, two commands. Negatives included: eight controlled
  failures where the interesting signal lost to a simpler one, and seven places
  where a result did not survive its own check and the record says so.

**Scope, up front.** Capturing needs Apple Silicon (`mlx-lm`). Everything else —
`compare`, the analysis, and re-deriving every table below from the committed
corpus — is pure numpy and runs anywhere, with no model downloads:

```bash
git clone https://github.com/beatakouchnir/clausius && cd clausius
pip install ".[dev]" && pytest -q                  # 36 tests, no accelerator
python -m knowledge.quantladder analyze            # rebuilds F14's ladder from records
python -m knowledge.frontier report                # rebuilds the 3-axis Pareto frontier
```

That is the fastest way to check whether the numbers here are real.

---

*Named for Rudolf Clausius, who coined the word **entropy** in 1865 — from the
Greek `trope`, transformation, shaped to echo "energy" so the two would sound
like the related quantities they are. Entropy is the signal this tool reads.*

## Quickstart

```bash
pip install "clausius[mlx] @ git+https://github.com/beatakouchnir/clausius"
```

The `mlx` extra is needed only to **capture**, and only runs on Apple Silicon.
`compare` and every analysis path are pure numpy and run anywhere.

Take 60 prompts of your own — production traffic is ideal, and **no labels are
needed** — or start with the 60 in [`examples/prompts.jsonl`](https://github.com/beatakouchnir/clausius/blob/main/examples/prompts.jsonl).
Capture the configuration you trust, capture the one you changed, compare:

```bash
clausius capture --model ./gemma-26b-a4b-4bit --prompts examples/prompts.jsonl \
    --out ref.json  --max-tokens 1536
clausius capture --model ./gemma-26b-a4b-2bit --prompts examples/prompts.jsonl \
    --out cand.json --max-tokens 1536
clausius compare ref.json cand.json --show 3
```

```
REGRESSION  (max d_z = +5.922, threshold 0.3, one-sided)
  compared 20 paired items, dropped 5 truncated
  all signals: max +5.92  p90 +11.64  mean +10.43  mean_top10 +9.00  first +2.80  gen_len +10.55
```

That is a real run. Those two checkpoints differ only in quantization, and the
2-bit one independently measures **73 points lower** on instruction adherence.
Twenty-five unlabeled prompts were enough to catch it. `compare` exits non-zero
on a regression, so it drops into CI without glue.

**→ [USAGE.md](https://github.com/beatakouchnir/clausius/blob/main/USAGE.md) is the operating manual**: choosing prompts, setting
the token cap, reading `d_z` and its interval, calibrating your own null, and
running it as a CI gate. Start there after your first run.

**Every default is a measured result, not a preference** — the 0.3 threshold
comes from 13 configurations known to be harmless, the one-sided test from a
construction that fools a two-sided one, the truncation filter from an effect
that doubles once you apply it. `src/clausius/core.py` states each one and
[FINDINGS.md](https://github.com/beatakouchnir/clausius/blob/main/FINDINGS.md) has the evidence.

> **Status.** `clausius` is installable and tested (36 tests, no accelerator
> required). The `knowledge/` research package that produced the findings is
> *not* packaged — it needs local model checkpoints and an external artifact
> store, and is kept for reproducibility rather than reuse.

---

## What was measured

Full evidence, positives and negatives, in [FINDINGS.md](https://github.com/beatakouchnir/clausius/blob/main/FINDINGS.md). The
headline results:

| | result | where |
|---|---|---|
| **Offload beats downsizing** | an offloaded 35B at **3.40 GB** scores **0.9447** on gsm8k, against a natively-fitting 4B at 3.91 GB scoring 0.8426. Same ordering on popqa (+13.9pp) and mmlu_pro (+20.9pp). The whole cost is latency. | F3–F6 |
| **The lossy shortcut is dominated** | *dropping* non-resident experts instead of fetching them costs **91% of gsm8k accuracy** at 50% residency, while being slower. Dominated on every axis, everywhere. | F6 |
| **Label-free detection works** | validated against five unrelated damage mechanisms whose true damage was measured independently — quantization, expert zeroing, top-k, expert substitution, LoRA fine-tuning. Benign controls stay clean: a 3.3× memory reduction changes ~25% of generations textually and moves no signal. | F8 |
| **It is more sensitive than labels** | a −2.2pp quantization regression flagged on **60 unlabeled prompts**; paired McNemar on gold answers needed **n=878** to reach p<0.05. | F14 |
| **Forgetting is detectable too** | move the reference and it becomes a training monitor: LoRA checkpoints on a held-out domain read d_z +0.74 to +0.84 at −16 to −26pp accuracy. | F9 |
| **Short benchmarks understate damage ~14×** | at matched memory, aggressive quantization costs **73 points** of instruction adherence where exact offload costs **1.0** — a difference invisible on factual QA, which loses 1.5pp where structured generation loses 18–21pp. | F11 |
| **Routing carries a fact-level address** | ablate the experts a fact routes to and that fact degrades far more than a paraphrase, a same-relation fact, or a random control. Replicated on two architectures. A **mechanism** result, not a product. | R9–R10 |

### What it does not do

- **Sensitivity is the weaker half.** Specificity is 13/13; a −5.7pp config is
  missed at any threshold that preserves that record.
- **Confidence-increasing damage would be invisible** to a one-sided detector.
  Three mechanisms were built to produce it and none did, but it is not excluded.
- **d_z is ordinal, not proportional.** 3-bit loses 25× more accuracy than 4-bit
  and reads 2× the d_z. "Something moved, and roughly how hard" is supportable;
  "you lost k accuracy points" is not.
- Reports that something moved, **not how much accuracy was lost**.
- Needs logits and a reference config; it cannot score a config in isolation.
- **The threshold is calibrated on one stack.** On a different framework,
  device or quantizer,
  measure your own null first — [USAGE.md](https://github.com/beatakouchnir/clausius/blob/main/USAGE.md#calibrating-your-own-null)
  has the recipe.

**Local and self-hosted only, deliberately.** Anthropic exposes no logprobs at
all, Gemini's are missing on current frontier models, and OpenAI caps
`top_logprobs` at 20 — *truncated* entropy, a different quantity. Hosted-API
support and cascade routing are recorded as **don't-build** decisions in
[EXPERIMENT.md](https://github.com/beatakouchnir/clausius/blob/main/EXPERIMENT.md), with the conditions that would reopen them.

**Apple Silicon to capture, anywhere to analyze.** `capture` needs `mlx-lm`;
`compare` and every analysis path are pure numpy.

Nothing about the *method* needs Apple Silicon — it wants greedy generation and
one teacher-forced pass yielding full-vocabulary logits, which PyTorch provides
too. A working PyTorch **framework backend** exists on the `feat/torch-backend`
branch and is **deliberately not shipped**.

Two independent things are often conflated here, so to be exact:

| | | |
|---|---|---|
| **framework** | mlx, PyTorch | which library runs the forward pass |
| **device** | cuda, mps, cpu | which hardware PyTorch dispatches to |

**Capture targets a GPU.** Apple Silicon via MLX in this release; CUDA is the
next one. CPU is not a supported capture target — measured on this machine,
decode runs at 6.7 tok/s for a 7B against 22 tok/s on the same hardware's GPU,
and the 26B MoE used for the findings below is impractical there. CPU remains
fine for everything that does not load a model: `compare`, the analysis paths,
and re-reading the corpus all run anywhere.

PyTorch does **not** require CUDA. The unshipped backend was measured on **mps**
and **cpu** with no CUDA present (F15, F15c), so the framework path is exercised
— what has never been calibrated is the **cuda device**, and separately the
CUDA-native quantizers (bitsandbytes, GPTQ, AWQ) that damage weights differently
from the ones measured here. Shipping a runtime whose threshold has not been
calibrated on the device most of its users would run would contradict the claim
this package makes about its defaults. The decision and the bar it must clear
are in [EXPERIMENT.md](https://github.com/beatakouchnir/clausius/blob/main/EXPERIMENT.md).

## Claim taxonomy — keep these separate

Two halves making **different kinds of claim**. Conflating them would overstate
one and undersell the other.

| half | signal | claim type | validated by | breadth |
|---|---|---|---|---|
| **online** | predictive entropy | *"this answer is likely wrong"*, *"this change broke the model"* — a **correctness** claim | Stages A–D, F8–F9 | 5 models, 6 task configs, 5 damage mechanisms |
| **offline** | routing ablation | *"this came from stored fact X"* — a **mechanism** claim | R3, R9, R9c, R10 | 2 architectures, probe-style |

**Entropy is mechanism-blind.** It reports that the model was uncertain and says
nothing about *how* the answer was produced.

Defensible: *"flag the answers the model was least sure of"*; *"this change
degraded the model"*; *"this claim causally depends on this stored fact"* (R9/R10
only).

**Not** defensible: *"our telemetry tells you how the answer was generated"* —
if the telemetry is entropy.

**Neither half claims to improve accuracy.**

## What did not work

Eight controlled negatives where routing lost to a simpler signal — usually
reading the prompt text, or predictive entropy. The structural reason: routing
is downstream of the residual stream *and* downstream of the prompt, so it is
bounded by both. Part II of [FINDINGS.md](https://github.com/beatakouchnir/clausius/blob/main/FINDINGS.md) records them so they are
not re-run.

## Seven corrections worth reading

The record keeps its own failures, because they were load-bearing:

- **A planned 16-50 GPU-hour sweep was canceled after reading the
  implementation.** `policy='exact'` fetches missing experts from disk before
  computing, so it would have measured a guaranteed flat line.
- **A confident mechanistic hypothesis died with its own bug fix.** A large
  gemma-vs-qwen split in top-k sensitivity, and a tidy shared-expert explanation
  for it, evaporated once a ranking error was corrected — `argpartition`
  guarantees membership, not order.
- **An instrumentation bug meant a whole axis was never measured.**
  `mx.get_peak_memory()` is a high-water mark that captured the model load
  before the cache freed it, so every configuration reported identical memory.
- **A published-looking result turned out to be a scoring artifact.** The RAG
  application's conditional claim read −0.431 in the predicted direction until
  the scorer was corrected to accept any gold alias rather than the first; it
  then read +0.066, the wrong direction. Recorded as open, not as a finding.
- **A rationale in the shipped code was wrong for three runs running.** The
  truncation filter was justified by "a damaged model rambles into the token
  cap". At cap 512 a *healthy* 4-bit checkpoint truncated 47/60 items where a
  destroyed 2-bit one truncated 3, and two LoRA fine-tunes truncated 22 and 33
  of 50 where their own base truncated none. Rambling tracks how far
  off-distribution a prompt is, not how damaged the model is. The filter was
  right; the reason was not (F14b).
- **A calibration claim was falsified by the next measurement.** A note here
  attributed a widened null (+0.172) to the reference being unquantized. The
  same pair against the same bf16 reference reads −0.062 on gsm8k: it was the
  prompt set, not the reference. Superseded rather than overwritten (F14c).
- **An assumption made the exact error the finding warns against.** F14 assumed
  3-bit "sits between two measured points in damage". Correct about the
  ordering, badly wrong about the distance — 3-bit costs −56.6pp, a broken
  deployment rather than a degraded one, while reading only 2x 4-bit's d_z. Kept
  visible, because it is a worked example of why d_z is ordinal and not
  proportional (F14d).

Two method rules came out of them, applied throughout:

> **Read a knob's implementation before scoping a sweep over it.**
> **A wiring control must be able to fail on the assumption it protects.**

The second has teeth: a top-k no-op control passed 16/16 while the ranking
assumption beneath it was wrong, because keeping every position is
order-independent.

## Prior art

Applied, not ours: MoE expert offloading with an LRU cache
([Mixtral-offloading](https://arxiv.org/pdf/2312.17238)), speculative expert
prefetch ([HOBBIT](https://arxiv.org/pdf/2411.01433), ExpertFlow, CommitMoE),
predictive entropy, McNemar, Belady, Pareto dominance. Several MIT-licensed
runtimes implement the offload mechanism and are downloadable today — this repo
does not try to compete with them.

What appears new: **accuracy measured across MoE offload configurations at
all** (four surveyed implementations report none); exact offload being
**capacity-non-reproducible but fixed-capacity deterministic**, which
invalidates output-diffing as a validation method; **prefill admitting *optimal*
caching**, since computing a layer's routing for the whole prompt yields the
access sequence before any fetch; and **label-free config-regression detection
validated against independently-known damage**.

---

## Reproducing the findings

Analysis is **stdlib/numpy only and loads no model** — that is deliberate, since
MLX uses unified memory and pinning to `mx.cpu` isolates nothing. The axis that
matters is loads-a-model vs doesn't.

```bash
python3 -m knowledge.quantladder analyze     # rebuilds F14's ladder from records
python3 -m knowledge.frontier report         # rebuilds the 3-axis Pareto frontier
python3 -m knowledge.probes --stats
```

Capture paths need `mlx-lm`, local checkpoints, and an external artifact store
(`CLAUSIUS_ARTIFACTS`) holding traces too large to publish. The outputs derived
from them are committed under `records/`, which is what the analysis paths read.
[`knowledge/README.md`](https://github.com/beatakouchnir/clausius/blob/main/knowledge/README.md) documents every module and how
to obtain the datasets; ten hard-won operational cautions are in
[EXPERIMENT.md](https://github.com/beatakouchnir/clausius/blob/main/EXPERIMENT.md).

## Repository layout

| path | what | needs |
|---|---|---|
| `src/clausius/` | **the tool** — capture, compare, CLI | numpy; mlx-lm only to capture |
| `tests/` | 36 tests, none load a model; CI installs the built wheel, not the source | numpy |
| `records/frontier`, `records/regress`, `records/context` | **the measurement corpus** — every number in FINDINGS.md, ~5 MB | — |
| `knowledge/` | the research package that produced the findings | local checkpoints, external artifact store |
| `USAGE.md` | **the operating manual** — prompts, caps, reading the output, CI | — |
| `FINDINGS.md` | the full experimental record, positives and negatives | — |
| `EXPERIMENT.md` | designs, scope decisions, and what was deliberately not built | — |

The split is deliberate. `clausius` is small, dependency-light and testable
because the detector **needs no labels and no benchmark harness to be used** —
scoring exists only to *validate* it. That validation is what `knowledge/`
does, and it is why that package is coupled to a specific local setup while the
tool is not.

The measurement corpus is committed. Every table in FINDINGS.md can be
recomputed from it with numpy alone, on a laptop, with no model downloads.
