# external — measuring what compression does to a model

Two questions, both answered on consumer hardware (M5 Max, 128 GB), across five
model families:

1. **If a model does not fit in your memory, what should you actually run?**
2. **If you change anything — config, quantization, fine-tune — how would you
   know it broke the model, without a labelled eval set?**

Full evidence, positives *and* negatives, in **[FINDINGS.md](FINDINGS.md)** —
which opens with a prior-art accounting stating which parts independently
re-derive published work and which appear to be new.

---

*Named for Rudolf Clausius, who coined the word **entropy** in 1865 — from the
Greek `trope`, transformation, shaped to echo "energy" so the two would sound
like the related quantities they are. Entropy is the signal this tool reads.*

## Quickstart

```bash
pip install "clausius[mlx] @ git+https://github.com/wintergreen22/clausius"
```

The `mlx` extra is needed only to **capture**, and only runs on Apple Silicon.
`compare` and every analysis path are pure numpy and run anywhere — `pip install
clausius` without the extra is enough to re-analyse captures or the published
corpus on any machine.

Take 60 prompts of your own — production traffic is ideal, and **no labels are
needed** — or start with the 60 in [`examples/prompts.jsonl`](examples/prompts.jsonl).
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

That is a real run, on 25 prompts. Those two checkpoints differ only in
quantization, and the 2-bit one independently measures **73 points lower** on
instruction adherence (§4). Twenty-five unlabelled prompts were enough to catch
it. `compare` exits non-zero on a regression, so it drops into CI without glue.

Start at 60 rather than 25, though: `compare` refuses to run on fewer than 20
paired items, and truncated items are dropped before that count is taken. The
run above landed on exactly 20 — one more truncation and it would have raised
instead of reporting. 60 keeps the floor out of reach on a first attempt.

**`capture` uses every prompt in the file.** `--limit N` takes only the first
N, and exists for prompt sets large enough that a badly chosen `--max-tokens`
is expensive to discover. Reach for it rarely: a full capture at a generous cap
*is* the reference you need, whereas a sampled probe is thrown away, so on a
60-prompt set sampling costs a run rather than saving one.

**Set `--max-tokens` for your traffic, and set it high.** The 512 default suits
short-answer work; a mixed instruction-following set will blow through it, and
items that hit the cap are dropped at compare time. A capture can be re-analysed
at any *tighter* cap, because every item's true length is known — but never at a
looser one, since truncated items no longer carry theirs. Capture generously
once rather than twice.

**`--show N` prints the items whose entropy moved most**, with the text both
configs produced. The verdict says something broke; this says what. It reads
recorded data, so it costs nothing and needs no model.

**What it costs.** Measured on `gemma-4-26b-a4b-it` (26B MoE) on an M5 Max:

| configuration | per prompt | 60 prompts | comparison (two captures) |
|---|---|---|---|
| 4-bit, ~13 GB | ~8 s | ~8 min | ~16 min |
| 8-bit, ~28 GB | ~11 s | ~11 min | ~22 min |
| bf16, ~52 GB | ~18 s | ~18 min | ~36 min |

Generation dominates, so cost scales with what your prompts elicit rather than
with `--max-tokens` directly: tripling the cap from 512 to 1536 cost 2.2x, not
3x, because items that finish early are unaffected.

**Captures are deterministic.** Greedy decoding over fixed weights, so the same
configuration on the same prompts reproduces exactly — across processes, not
merely within one. Measured: a capture at cap 1536 predicted 47/60 truncations
at cap 512, and a separate run actually at 512 truncated exactly 47. That is the
property the CI gate rests on; a detector that drifted between runs could not
gate anything. It also means comparing a config against *itself* yields exactly
zero rather than a noise floor — the ±0.10 null comes from benign *configuration*
changes, not from measurement noise.

**Managing the reference.** Capture the config you trust once, commit
`ref.json` as a build artifact, and compare every candidate against it.
Re-baseline deliberately — when you accept a new configuration, not when a run
goes red. `compare` **rejects** captures made on different prompt sets rather
than silently comparing them, so changing your prompts forces a new reference,
which is the intended behaviour and not an obstacle to route around.

The same three commands cover a LoRA checkpoint (`--adapter`), an inference
backend, or an offload setting — clausius does not manage configurations, it
compares whatever two runs you hand it.

```python
from clausius import capture, compare

ref  = capture("./model", prompts, tag="v1")
cand = capture("./model", prompts, tag="v2", adapter="./my-lora")
print(compare(ref, cand))
```

**Every default is a measured result, not a preference** — the 0.3 threshold
comes from 13 configurations known to be harmless, the one-sided test from a
construction that fools a two-sided one, the truncation filter from an effect
that doubles once you apply it. `src/clausius/core.py` states each one and
[FINDINGS.md](FINDINGS.md) has the evidence.

> **Status.** `clausius` is installable and tested (14 tests, no accelerator
> required). The `knowledge/` research package that produced the findings below
> is *not* packaged — it needs local model checkpoints and a sibling
> `quantize` checkout, and is kept for reproducibility rather than reuse.

---

## 1. Offloading beats downsizing, and it is not close

For MoE models, streaming experts from SSD is **accuracy-preserving down to 12%
expert residency**. At that point the offloaded 35B model uses **less memory
than a 4B model that fits natively**, and scores higher:

| gsm8k | memory | accuracy | s/item |
|---|---|---|---|
| qwen-35b, exact offload, 12% resident | **3.40 GB** | **0.9447** | 10.21 |
| gemma-26b, exact offload, 19% resident | 3.76 GB | 0.9091 | 7.04 |
| gemma-e4b, fully resident | 3.91 GB | 0.8426 | 2.74 |

Same ordering on popqa (**+13.9pp**) and mmlu_pro (**+20.9pp**). Accuracy is
flat across a 4-8x capacity range — on qwen/popqa it is *identical* (0.2900) at
every rung from 100% down to 12%.

**The entire cost is latency.** After an initial cliff, throughput falls as
roughly **memory^0.5** — 4x less memory for ~2x slower. Time-to-first-token is
the real constraint: it degrades 2-3x faster than throughput, because decode's
working set is 8 experts per layer while prefill routes every token
independently. **Your TTFT budget, not accuracy, limits how far you can
compress.**

**And there is a trap.** The *lossy* alternative — dropping non-resident experts
instead of fetching them, which looks like an ordinary memory setting — costs
**91% of gsm8k accuracy at 50% residency**. At matched memory it scores 0.0250
against the small model's 0.8426, while being *slower*. It is dominated on every
axis, everywhere.

## 2. You can detect that *anything you changed* broke the model — without labels

Compare the model's own per-token entropy against a reference, paired on the
same **unlabelled** prompts. Validated against **five unrelated damage
mechanisms** whose true damage was measured independently:

| mechanism | example | true damage | entropy d_z | flagged |
|---|---|---|---|---|
| *(benign control)* | offload cap-64, 3.3x less memory | +0.005, n.s. | **−0.05** | no ✓ |
| *(benign control)* | quantization `mixed_4_6` | +0.005 | **−0.03** | no ✓ |
| quantization | 2-bit | −0.231 | **+2.82** | yes ✓ |
| quantization | 3-bit | −0.031 | +1.02 | yes ✓ |
| expert zeroing | static cap-64 | −0.925 | +0.64 | yes ✓ |
| top-k reduction | top-k=4 | **−0.016, p=0.004** | +0.56 | yes ✓ |
| expert substitution | non-resident → resident expert | −0.230 | +1.06 | yes ✓ |
| LoRA fine-tuning | `adapter-qa`, held-out domain | −0.260 | +0.76 | yes ✓ |

All figures are `p90` on the full item set. `max` is the better statistic when
*ranking* configs on long-output tasks — see below — but **every verdict above is
identical under both**, which is the property that matters.

Quantization matters most in that list: expert-zeroing and top-k are both
*routing* interventions, so a detector that also works on quantization and on
fine-tuning is not tracking one mechanism's signature.

**The quantization ladder is monotone** on popqa — `mixed_4_6` +0.005 → −0.03,
`mixed_3_6` −0.015 → +0.66, 3-bit −0.031 → +1.02, 2-bit −0.231 → +2.82 — and it
resolves damage as small as **1.5 percentage points**.

**The benign arms are the load-bearing control.** A 3-3.3x memory reduction
changes ~25% of generations *textually* and moves no signal. It detects damage —
not change, not perturbation.

**It is more sensitive than the labelled alternative.** Top-k=4's −2.3pp
regression is flagged at **n=200**, where paired McNemar on labels reads p=0.227
and needs **n=800** to reach significance. Four times the eval budget for the
same answer, if you even have labels.

**Calibration.** Thirteen benign configurations across two models and two tasks
put the false-alarm null at **|d_z| ≤ 0.10**, which sets the threshold at 0.3.
An earlier eyeballed 0.5 was ~5x the null and was costing real detections.

> **The null width varies with the prompt set.** An 8-bit checkpoint of
> `gemma-4-26b-a4b-it` is benign by labels (gsm8k 0.8500 vs 0.8450 on the same
> 200 items, and +1.3pp at n=300, both n.s.). Against a bf16 reference it reads
> **d_z +0.172** on a mixed instruction-following set of 60 prompts, and
> **−0.062** on 200 gsm8k items. Same pair, same reference, opposite sides of
> the ±0.10 null. The 0.3 threshold keeps margin in both, but ±0.10 is a
> property of the corpus it was measured on — single-task runs at per-task caps
> — and a heterogeneous prompt set is entitled to a wider null.

**Adversarial testing.** The dangerous failure would be damage that makes a model
*more* confident — a silent false negative. Two constructions were built to
produce it: **expert substitution** (non-resident experts routed to a real
resident expert, so the arithmetic is well-formed and only the weights are
wrong) and **top-k = 1**. Both are destructive and both were detected. Across
every damaged configuration measured, d_z is **positive**; the only negative
belongs to logit sharpening, whose greedy output is bit-identical by
construction and whose accuracy delta is exactly 0.0000. So a one-sided detector
is correct on every arm tested, including the ones built to defeat it. That is
*"three mechanisms attempted, none evaded"* — not *"ruled out"*.

### What it does not do

- **Sensitivity is the weaker half.** Specificity is 13/13; a −5.7pp config is
  missed at any threshold that preserves that record.
- **Confidence-increasing damage would be invisible** to a one-sided detector.
  None could be constructed here, but the possibility is not excluded, and the
  two-sided alternative has a demonstrated false positive (logit sharpening).
  This is a calibration choice the tool must expose.
- **d_z magnitude is mechanism-dependent.** Quantization at −1.5pp reads +0.66;
  expert-zeroing at −5.7pp reads +0.26. "Something moved, and roughly how hard"
  is supportable; "you lost k accuracy points" is not.
- **No aggregation dominates; the verdict is what's robust.** `p90` is a
  percentile over generated tokens, so hundreds of confident rambling tokens
  dilute it — on the quantization ladder it *inverts* on both long-output tasks
  while `max` stays monotone. But across the full arm set `max` wins on some
  configs and `p90` on others. At threshold 0.3 **every damaged arm flags and
  every benign arm stays clean under both**, so report several and prefer `max`
  only when ordering matters.
- Reports that something moved, **not how much accuracy was lost**.
- Needs logits and a reference config; it cannot score a config in isolation.

**Local and self-hosted only, deliberately.** Anthropic exposes no logprobs at
all, Gemini's are missing on current frontier models, and OpenAI caps
`top_logprobs` at 20 — which is *truncated* entropy, a different quantity from
the full-distribution measurements here. Hosted-API support and entropy-gated
cascade routing are both recorded as **don't-build** decisions in
[EXPERIMENT.md](EXPERIMENT.md), with the conditions that would reopen them.

**Two backends, one validated.** `--backend mlx` is what every number in
[FINDINGS.md](FINDINGS.md) was measured on. `--backend transformers` runs the
same measurement on torch — CUDA, CPU or MPS — because nothing about the method
needs Apple Silicon: it wants greedy generation and one teacher-forced pass
yielding full-vocabulary logits. **It is experimental.** It is validated on CPU
and MPS against measured gsm8k accuracy (F15); it is **untested on CUDA**, on
multi-GPU sharding, and against CUDA quantizers (bitsandbytes, GPTQ, AWQ).
Comparing a capture from one backend against the other warns, because the
threshold was calibrated within a runtime.

### Calibrating your own null

The 0.3 threshold comes from 13 benign configurations measured on **one stack**,
and F14c showed the null is not a universal constant — the same benign 8-bit
checkpoint against the same bf16 reference reads **+0.172** on a mixed
instruction set and **−0.062** on gsm8k. On a new backend, a new quantizer or a
markedly different prompt set, measure your own floor before trusting the
default:

```bash
# two configurations you have independent reason to believe are equivalent —
# a benign precision change, a runtime version bump, a cache setting
clausius capture --model ./cfg-a --prompts prompts.jsonl --out null_a.json --max-tokens 1536
clausius capture --model ./cfg-b --prompts prompts.jsonl --out null_b.json --max-tokens 1536
clausius compare null_a.json null_b.json
```

Whatever |d_z| that produces is **your** false-alarm floor. If it exceeds 0.10,
scale the threshold with it — the published 0.3 is ~3x a null of 0.10, so a
floor of 0.2 argues for `--threshold 0.6`. Repeat with two or three benign pairs
rather than one; a single pair gives you a point, not a floor. The reported
interval matters here too: a null estimated on 25 items with a CI half a unit
wide has not established anything.

This is the procedure that produced the 0.3 default, and it is the honest
answer to "does the threshold transfer to my setup" on any stack this project
has not measured.

## 3. The same machinery detects catastrophic forgetting

Move the reference and it becomes a training monitor: reference = the
adapter-free base, candidate = a LoRA checkpoint, items = **a domain the adapter
was never trained on**.

| checkpoint | popqa accuracy | Δacc | entropy d_z |
|---|---|---|---|
| base | 0.2900 | — | — |
| `adapter-attn` | 0.1300 | −0.160 | **+0.84** |
| `adapter-router` | 0.1250 | −0.165 | **+0.74** |
| `adapter-qa` | 0.0300 | −0.260 | **+0.76** |

Forgetting is normally detected by accuracy on held-out tasks, **which needs
labels**; the existing entropy work uses entropy as a *training mechanism* or
over *attention* distributions. A label-free output-entropy monitor was not found
published. The value is timing — forgetting is usually discovered after the
compute is spent.

*(A third application — measuring whether retrieved RAG context actually helped
— is **partially validated**: whether retrieval supplied the answer is
detectable without labels on three models (paired entropy shift d_z −1.06 to
−1.86). Ranking *per-query* usefulness is out of scope for the task tested,
because when the answer is verbatim present retrieval is binary rather than
graded — even a 4B model uses it at 99.7%. See F10 in [FINDINGS.md](FINDINGS.md).)*

## 4. Short factual benchmarks understate compression damage ~14x

Compression does not degrade capabilities evenly, and the benchmark you validate
on decides what you see. Same model, same five-point quantization ladder:

| config | **instruction adherence** | short factual recall | multi-step math |
|---|---|---|---|
| `mixed_4_6` | −0.006 | +0.005 | +0.015 |
| `mixed_3_6` | **−0.215** | −0.015 | −0.180 |
| 3-bit | **−0.392** | −0.031 | −0.473 |
| 2-bit | **−0.734** | −0.231 | −0.895 |

**Short factual recall is the anomaly** — it loses 1.5pp where the others lose
18–21pp. It answers in one entity token; the others generate long structured
output where errors compound *within a single generation*.

**And on that same sensitive task, the two ways of saving memory separate by an
order of magnitude:**

| configuration | memory | instruction adherence |
|---|---|---|
| 4-bit, fully resident | 13.23 GB | 0.8492 |
| **2-bit, fully resident** | **7.35 GB** | **0.1150** |
| **4-bit + exact offload, 50% resident** | **7.49 GB** | **0.8687** |
| 4-bit + exact offload, 19% resident | 3.76 GB | 0.8636 |

For the same ~44% memory reduction, aggressive quantization costs **73 points**
of instruction adherence and exact offload costs **1.0 point** — and offload
then reaches 3.76 GB, half what 2-bit needs, for 1.5 points. **When a MoE model
doesn't fit, stream the experts; don't crush the weights.**

This matters for agents, which are the structured-generation workload and whose
errors compound *again* across steps. Instruction adherence 0.849 → 0.634 over a
10-step task takes end-to-end success from **19.5% to 1.1%** — a 95% relative
collapse from a setting that looks like a rounding error on factual QA.

## 5. Routing carries a causal, fact-level address

The original question, and it holds. Ablate the experts a fact routes to and
that fact degrades far more than a paraphrase, a same-relation fact, or a random
control — replicated on two architectures, localised to the late third of
layers, distributed rather than point-like, surviving into the model's own
generated text, and validated by injecting facts the model provably could not
have known.

| condition | qwen ΔNLL | gemma-base ΔNLL |
|---|---|---|
| `own` — this fact's experts | 3.022 | 3.539 |
| `para` — same fact, different wording | 3.134 | 3.460 |
| `samerel` — same entity, different relation | 1.943 | 2.314 |

It is a **mechanism** result, not a product. See the claim taxonomy below.

---

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
bounded by both. Part II of [FINDINGS.md](FINDINGS.md) records them so they are
not re-run.

## Three corrections worth reading

The record keeps its own failures, because they were load-bearing:

- **A planned 16-50 GPU-hour sweep was cancelled after reading the
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

## Running it

Analysis is **stdlib/numpy only and loads no model**. That is deliberate: MLX
uses unified memory, so pinning to `mx.cpu` isolates nothing — same RAM pool,
same bus. The axis that matters is loads-a-model vs doesn't.

```
# no model needed
python3 -m knowledge.probes --stats
python3 -m knowledge.routing --model qwen --kind expert
python3 -m knowledge.frontier hot            # hot-expert lists from saved traces
python3 -m knowledge.frontier report         # 3-axis Pareto frontier

# needs mlx-lm (use quantize's venv)
QV=../quantize/.venv/bin/python
$QV -m knowledge.seam                        # resolve both router spellings
$QV -m knowledge.capture --selftest          # stub-model pipeline check

# the frontier: exactness, speed/memory, accuracy per config
$QV -m knowledge.frontier gen   --model qwen --capacity 256 --tag full
$QV -m knowledge.frontier compare --model qwen --a full --b c64
$QV -m knowledge.frontier speed --model qwen --capacity 64
$QV -m knowledge.frontier acc   --model qwen --capacity 64 --policy static --task popqa

# the label-free detector
$QV -m knowledge.regress capture --model qwen --capacity 256 --policy exact --tag ref
$QV -m knowledge.regress capture --model qwen --capacity 64 --policy static --tag static_c64
$QV -m knowledge.regress analyse --model qwen --task gsm8k
```

Reads `../quantize/records` (override with `QUANTIZE_REPO`). `quantize` and
`ghostlight` are **read-only** to this project; nothing here writes to them.

### Operational cautions, learned the hard way

- **Memory.** A first fine-tune peaked at **128.3 GB on a 128 GB machine**.
  Every GPU entry point now sets an explicit `mx.set_memory_limit`.
- **Background jobs.** An overnight chain once blocked on
  `while pgrep -f "stage_a --task gpqa"` whose own command line matched the
  pattern. Chains here use bounded waits, never `pgrep` waiters.
- **Token caps.** Truncation manufactures errors — one benchmark moved
  0.345 → 0.820 purely by raising the cap. Caps are set per task and never
  shortened to save time.

---

## Repository layout

| path | what | needs |
|---|---|---|
| `src/clausius/` | **the tool** — capture, compare, CLI | numpy; mlx-lm only to capture |
| `tests/` | 14 tests, none load a model; CI installs the built wheel, not the source | numpy |
| `records/frontier`, `records/regress`, `records/context` | **the measurement corpus** — every number in FINDINGS.md, ~5 MB | — |
| `knowledge/` | the research package that produced the findings | local checkpoints, sibling `quantize` |
| `FINDINGS.md` | the full experimental record, positives and negatives | — |
| `EXPERIMENT.md` | designs, scope decisions, and what was deliberately not built | — |

The split is deliberate. `clausius` is small, dependency-light and testable
because the detector **needs no labels and no benchmark harness to be used** —
scoring exists only to *validate* it. That validation is what `knowledge/`
does, and it is why that package is coupled to a specific local setup while the
tool is not.

The measurement corpus is committed. Every table in FINDINGS.md can be
recomputed from it with numpy alone, on a laptop, with no model downloads.
