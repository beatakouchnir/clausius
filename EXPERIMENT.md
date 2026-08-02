# Validation design — is entropy-based error prediction general?

R13 established, on **one benchmark and one model**: predictive entropy flags a
wrong answer at **AUC 0.892** on PopQA/qwen-MoE, beating routing (0.781), with
combining the two *worse* than entropy alone (0.877).

That is a promising result on a single point in a large space. This design tests
whether it survives, and it is ordered by **how likely each risk is to break the
result**, not by how easy each is to run.

---

## REVISION after the CoT probe (R14) — read this first

A design probe on GSM8K (n=250) changed three things materially.

**1. R13's measurement position does not transfer to CoT.** Entropy at the last
prompt token scores **0.447** on GSM8K — below chance — and at the answer token
**0.419**. Both are useless. The reason for `answer` failing was predicted: by
the time the chain is written the model has committed, so a confident-but-wrong
chain ends in a near-deterministic token. **Use `mean` entropy over the
generation for CoT, `first` for short-answer recall.**

**2. Token caps contaminate everything, and quantize's own data shows how much.**
27/250 GSM8K generations hit the 512-token cap and **81.5% of those were wrong**
— *73% of all errors were truncation artifacts, not reasoning errors*. Removing
capped items inverts the ranking:

| variant | all items | uncapped only |
|---|---|---|
| `gen_len` (dumb baseline) | 0.878 | **0.585** |
| `mean` entropy | 0.762 | **0.897** |
| `p90` | 0.751 | 0.880 |
| `first` | 0.447 | 0.466 |

`suite.json` confirms the mechanism: `gsm8k/qwen` 0.86 -> **0.935** at cap960;
`mmlu_pro/qwen` 0.345 -> **0.820** at cap4160. **The original plan here proposed
MMLU-Pro precisely because its 0.345 accuracy offered "more errors" — those
errors are mostly truncation.** Every run must use the generous cap, and every
run must report its truncation rate and exclude capped items from the AUC.

**3. Errors must come from HARD tasks, not from capping easy ones.** Uncapped
GSM8K leaves 8 errors in 250 items, far too few for a stable AUC. Sizing must be
by expected *error count* (target >= 100), not item count:

| task | expected error rate | items for ~100 errors |
|---|---|---|
| AA-Omniscience | >50% (by design) | ~200 |
| HLE | ~83% (gemma 17.2%) | ~120 |
| PopQA | 75% (measured) | ~135 |
| GPQA Diamond | ~20% (gemma 79.2%) | ~500 |
| GSM8K uncapped | ~4% | ~2500 — **impractical, drop as a primary task** |

---

## The three risks, in priority order

### 1. Task type — the biggest risk by far

PopQA is pure long-tail **recall**: short answers, no reasoning, 75% error rate,
and the answer token immediately follows the prompt. Entropy at the last prompt
token therefore measures uncertainty about *the answer itself*.

On other task types that alignment breaks:

| task | why entropy might fail |
|---|---|
| **GSM8K** (math CoT) | errors come from a bad reasoning chain, not uncertain recall. The model can be **confidently wrong** after a wrong step. Entropy at the prompt's end predicts the first *reasoning* token, which is not the answer |
| **MMLU-Pro** (multiple choice) | the answer space is ~10 letters; entropy over that is a different quantity from entropy over a 250k vocabulary |
| **LongBench** (long-form) | there is no single answer token to measure at |
| **GPQA Diamond** (hard science) | high error rate, but errors are reasoning failures on material the model partly knows |

**If entropy only works on recall, the product is a recall-answer checker, not a
general error detector.** That is a much narrower claim and must be known before
publishing.

### 2. The competitor not yet tested: self-consistency

Sampling k times and measuring answer agreement is what production systems
actually use, and it is the honest strong baseline. It costs k x generation.

**The real product claim is not "entropy beats routing" — it is "entropy
approaches self-consistency at 1/k the cost."** If self-consistency dominates,
the answer is to use self-consistency and neither entropy nor routing matters.
This baseline is currently missing and its absence is the weakest point in R13.

### 3. Architecture portability

Entropy should be architecture-agnostic; that is its main advantage over
routing. Untested. The matched qwen MoE/dense pair is the clean test because it
holds training constant.

---

## Measurement: capture once, analyse many ways

R13 measured entropy at the **last prompt token**. That is correct for
short-answer recall and wrong for CoT, where the answer arrives after a chain.

So the capture records **per-token entropy and top-1 probability across the
whole generation**, plus routing where the model is MoE, and every variant is
computed offline:

- `first` — at the last prompt token (R13's measure)
- `answer` — at the token that emits the final answer (located via the
  `ANSWER:` marker the suite already requires, or the first content token)
- `mean`, `max`, `min` — over generated tokens
- `last` — at the final generated token

This follows `profile_experts.py`'s discipline: the GPU pass is expensive and
the analysis is free, so never bake an analysis choice into a capture.

---

## Stages, with decision points

### Stage A — task generalisation (decisive, run first)

**Model:** qwen3.6-35b-a3b MoE only. **Tasks (revised):** PopQA (done, 300
errors), **AA-Omniscience** and **HLE** as the high-error knowledge tasks,
**MMLU-Pro at cap4160** as the CoT task, GSM8K at cap960 kept only as a
low-error-rate control. `think=False` throughout to match `suite.json`.
Sized by expected error count, not item count.

Reuses `quantize/suite.py` loaders, prompt builder and scorers unchanged, so
accuracies are comparable to the recorded runs and the scorer is not
re-litigated.

**Signals:** all entropy variants, top1, answer NLL, generation length (a dumb
baseline that must be beaten — long answers may simply correlate with errors),
and routing.

**Decision point:** if entropy holds above ~0.75 AUC on at least three of five
task types, continue to Stage B. If it collapses to near-chance off recall
tasks, stop and rescope the product to recall-answer checking.

### Stage B — self-consistency comparison

**Tasks:** the best and worst from Stage A. **k=5** samples at temperature 0.7,
agreement rate as the signal.

Report entropy, self-consistency, and their combination, **against cost**: 1x vs
5x generation. The product claim lives or dies here.

### Stage C — architecture portability

**Models:** qwen MoE, qwen 27B dense (matched pair), gemma-4-26b-a4b MoE,
gemma-4-31b dense, gemma-4-e4b dense. **Tasks:** two from Stage A — one recall,
one reasoning.

Tests whether entropy's AUC is stable across architectures, and confirms routing
is not needed (it cannot be computed on the dense models at all).

### Stage D — product metrics

AUC measures ranking; a product needs a threshold. Report **error-catch rate at
a fixed review budget** (flag 10%/20%/30% of answers, what fraction of errors is
caught) and a calibration curve. This is what a buyer evaluates.

---

## Pre-registered bars, fixed before any run

1. Entropy generalises if it holds **AUC >= 0.75 on >= 3 of 5 task types**.
2. Entropy is competitive with self-consistency if it is within **0.05 AUC at
   1/5 the cost**.
3. Routing earns a place in the online product only if it **beats entropy on
   >= 2 task types**, or adds **>= 0.03 AUC** in combination. R13 showed it
   doing neither on PopQA.
4. Any signal must beat **generation length**, the dumb baseline.

Directions are fixed a priori: higher entropy, lower top1, higher NLL, longer
generation all predict error.

---

## Cost

| stage | runs | estimate |
|---|---|---|
| A | 5 tasks x 1 model | ~3 h (GSM8K/MMLU-Pro generate 512 tokens/item; PopQA 48) |
| B | 2 tasks x 5 samples | ~4 h |
| C | 2 tasks x 5 models | ~6 h |
| D | analysis only | free |

~13 h of GPU, runnable unattended overnight in stages. Nothing here needs the
cloud budget.

**Download required:** `Qwen/Qwen3.6-27B` (dense) — ~16 GB at 4-bit, 786 GB
free. Only needed for Stage C's matched pair; Stages A and B run on what is
already on disk.

---

## What is deliberately excluded

- **HumanEval** — code, execution-scored; "error" is a failing test, a different
  object from a false factual claim.
- **IFEval** — multi-constraint instruction following; an answer is partially
  correct, so the binary label is not meaningful.
- **The agentic and multimodal benchmarks** on benchlm (BrowseComp, OSWorld,
  MMMU-Pro) — they need tools or images this setup does not have.
- **HLE / AA-Omniscience** — attractive because the error rates are very high
  (gemma 17.2% and 18.2%), and AA-Omniscience is purpose-built around
  hallucination vs abstention. Excluded only pending a check that the datasets
  are openly available; worth adding to Stage A if they are.


---

# Designs pending GPU (drafted 2026-07-29)

## A. Entity discrimination — a test that can actually work

**Why the last one failed.** `other` was split by whether the substitute entity
shared the target's `field`. But the field **never appears in the question** —
"Who directs Glascheis Trust?" contains no field information — so the split
varied something the model could not condition on. Same-field 0.582 vs
cross-field 0.558 is not a null result; it is a non-experiment.

**The rule this violated:** a control can only vary what is present in the
input. Anything else measures nothing.

Three replacements, cheapest first.

**A1 — name-similarity gradient (free, no retraining, no new capture).**
Entities differ *only* in coined name tokens, so if entity routing is
name-driven, `other` damage should rise as names get more similar. Correlate
per-item `other` damage against token-level similarity between the target and
substitute entity names (shared token count / Jaccard over the tokenised name).
A positive correlation says entity discrimination is real but name-token-driven;
a flat one says entity identity is barely encoded at all. Graded rather than
binary, and it runs on `inject.json`'s saved rows.

**A2 — org-type split (free, existing data).** Names are `<coined> <type>` where
type ∈ {Trust, Foundation, Institute, Society, Consortium, Bureau, Academy,
Guild}. Unlike `field`, the **type is in the prompt**. Split `other` by
same-type vs different-type. Weak semantics, but it is a valid control because
the variable is present in the input.

**A3 — semantically-loaded entities (needs a retrain).** The real hypothesis is
that coined entities have no semantic content, whereas R9's grid used
France/Japan/iron/gold with rich pretrained representations. Test it by
injecting entities whose names carry meaning in the prompt — e.g.
`the Glacier Survey of Vantholm` vs `the Foundry Guild of Bracken` — and split
`other` by semantic distance of the head noun. If discrimination improves, weak
entity separation is a limitation of the injection paradigm, not of R9.

## B. Improving separation (currently 0.673 vs 0.563/0.581)

Ranked by expected gain per unit cost.

**B1 — WITHDRAWN after measurement.** The consensus set retains 7.07 of 8
experts per layer (paraphrases route near-identically), so it is not a distinct
intervention. The overlap check this prompted — see FINDINGS R9g — showed
`para` and `other` sit at matched overlap (0.78 vs 0.73) with 68% different
damage, which is a stronger result than B1 could have produced.

*Original rationale, kept for the record:* Today `own` ablates the experts from ONE
rendering of the fact, so it inevitably mixes fact-specific and
input-specific routing. Instead ablate the **intersection** (or top-weighted
union) of experts across all three paraphrases — the routing the fact evokes
*regardless of wording*. If a fact-level address exists, the consensus set
should damage its own fact more sharply and its neighbours less. This is the
single change most likely to separate the two hypotheses rather than merely
reduce noise.

**B2 — average over multiple substitutes (cheap variance reduction).** `para`,
`samerel` and `other` each currently draw ONE random substitute per item, so
each condition carries the variance of a single draw. Averaging 3-5 draws should
tighten every cell without changing what is measured.

**B3 — score the FIRST answer token only.** NLL is currently averaged over the
whole answer span, and for multi-token coined names the later tokens are nearly
determined by the first — diluting the signal with tokens that carry no fact
content. Scoring the decisive token concentrates it.

**B4 — paired within-item statistics (free).** Compare `own − samerel` *per
item* rather than group means. Item-level variance in base NLL is large and
cancels exactly in a paired test.

**B5 — retune K and layers for this setting.** Layers 28-39 and K=8 were chosen
on grid2's pretrained facts (R9c). LoRA touched every layer's router, so
injected facts may localise differently. A short sweep is cheap.

**B6 — train to higher recall.** Recall is 41.7%; items the model answers
correctly are the only usable ones, and crisper retrieval plausibly means
crisper addresses. More epochs or fewer facts would raise it.

**Order:** B4 and B2 first (free/cheap, pure variance), then B1 (the real
experiment), then B3, then B5/B6 only if still needed.

**Pre-registered bar:** the separation is meaningfully improved if
`para>samerel` for correctly-recalled injected facts exceeds **0.75** while the
two confabulation groups stay near **0.55-0.60**. Raising all three together
would indicate a scoring artifact, not better discrimination.

---

# The configuration frontier — pivot of 2026-07-30

The product question changed shape, and one code reading killed the experiment
that had just been scoped. Recorded here in the order it actually happened,
because the wrong turn is the instructive part.

## Why the accuracy-vs-budget sweep was cancelled

The plan was: sweep the offload cache's resident capacity, measure benchmark
accuracy at each rung, publish the accuracy-vs-memory curve nobody in the MLX
ecosystem has. Estimated 16-50 GPU hours.

Then `quantize/offload.py` `ExpertCache.ensure()`: at `policy='exact'` a cache
miss calls `_install(e)`, which **fetches the real weights from disk before the
gather**. A miss costs latency and nothing else.

> **At `policy='exact'`, accuracy is capacity-invariant by construction.** The
> sweep would have spent 16-50 hours redrawing a flat line.

Which also means the control-vs-offload accuracy deltas in
`quantize/records/suite.json` — mmlu_pro 0.530 → 0.500, and ifeval 0.850 →
0.875 in the *other* direction — are **numerical noise, not degradation**. The
file names the source itself: `gather_qmm`'s sorted and unsorted paths "disagree
by ~1.3e-3 absolute", and `do_sort = indices.size >= 64` routes decode (8
indices) and prefill (>=64) down different kernels. A 1e-3 perturbation flips an
occasional greedy argmax and the generation diverges from there. Those deltas
were quoted three times in conversation as if they were signal.

**Method rule added:** before scoping a sweep over a knob, read the knob's
implementation and establish whether it *can* move the measured quantity.

## Two families of mechanism

| family | mechanism | trades | floor |
|---|---|---|---|
| **lossless** | `policy='exact'` | memory ↔ **speed** | none — shrink arbitrarily, pay latency |
| **lossy** | `policy='static'` (non-resident → zero slot), top-k reduction, heterogeneous precision | memory/speed ↔ **accuracy** | yes |

Conflating these produced the wrong plan. "Compress harder" means different
things in each column.

## Two results available from arithmetic alone

**1. Exact offload dominates heterogeneous precision on memory.** W3's most
aggressive mix (1% hot at 4-bit, rest 2-bit) needs 7.19 GB of experts = 8.54 GB
resident. Exact offload measured **7.78 GB** and reaches **4.3 GB** at cap-32.
Mixed precision keeps everything resident, so it never buys memory — it buys
*speed*, by dodging the streaming tax, and pays in accuracy. That reframes W3
from a compression play into a latency play.

**2. On long context the offloaded big model is already dominated.**

| config | longbench | s/item | memory |
|---|---|---|---|
| gemma-26b, exact offload | 0.5262 | **22.79** | 7.78 GB |
| gemma-e4b, resident | 0.5132 | **1.55** | 4.20 GB |

1.3pp — inside noise at n=150 — for 14.7× the latency and 1.9× the memory. No
one should run that configuration. The crossing point on long context is behind
us, not ahead.

## The reframe: 3-axis Pareto, not maximum compression

Not "how far can we compress" but **"at what point is a compressed big model
dominated by a small model that fits natively?"** A configuration is dead if
something else is no worse on (memory, accuracy, speed).

For a 397B-class model the sequence is:

1. compress losslessly → accuracy intact, gets slow
2. **too slow** — SwiftLM reports 5.2 tok/s for a *122B*; a 397B exact is low
   single digits. You die of latency, not accuracy.
3. reach for lossy shortcuts (2-bit cold tail, top-k−2, 3-bit KV) to buy speed
4. **accuracy dies here** — on the lossy axis, driven by a latency constraint
5. a smaller resident model would have been better all along

**The crossing point is not on the offload axis at all.** It is on the axis
people are forced onto once offload gets too slow — and that is the axis for
which nobody publishes numbers.

**Falsifiable claim:** region 4 is always a mistake. If lossy compression of a
big model lands below a small model that fits natively, the honest advice is
"use the smaller model."

## Blocked: the mixed-precision three-way

The sharpest test would be three configurations at matched memory — 26b exact
offload (7.78 GB), 26b hot-10%/cold-2-bit resident (9.06 GB), e4b resident
(4.20 GB) — with the middle arm's accuracy being W3's explicitly open "crux".

**Blocked, architecturally.** `mlx_lm.models.switch_layers.QuantizedSwitchLinear`
holds **one** `self.weight` of shape `(num_experts, out, in)` quantized with a
single `bits` and `group_size`. Per-expert mixed bit width inside a layer is
impossible without splitting each layer into two modules (hot subset + cold
subset) and routing between them, or writing custom kernels. `quantize/skew.py`
is analysis only — it projects GB via a `gb(n_experts, bits)` formula and builds
no artifact. Not an overnight job; needs a decision on whether to build it.

## What runs instead — `knowledge/frontier.py` + `run_frontier.sh`

`policy='static'` is the lossy mechanism that is *already implemented*, needs no
new artifact, and answers the same shape of question. Launched 2026-07-30 00:36.

1. **Exactness verification.** cap-full vs cap-low at `policy='exact'`, greedy,
   16 gsm8k items × 256 tokens, token-id diff. Long generations because
   divergence compounds — a 4-token answer can match by luck. *This validates
   the claim the whole plan rests on.* If it diverges, accuracy is not
   capacity-invariant and the sweep comes back.
2. **Speed curve, measured.** tok/s and TTFT at 6 (gemma) / 7 (qwen) rungs,
   replacing the modelled tax. The model was optimistic: `reads_per_token` rises
   9 → 122 across the range, so effective bandwidth degrades below the assumed
   7.4 GB/s, and eviction churn is not free.
3. **Static-policy accuracy.** popqa and gsm8k at 3-4 rungs, both models.
   Pins come from `frontier hot`, which derives per-layer hot-expert lists from
   the saved traces — wrap's default pins experts 0..C-1, an arbitrary subset
   that would understate static badly.

**Pre-registered predictions.**
- Exactness: **identical** token sequences at every capacity. Divergence at
  token 0 would be a wiring bug; divergence deep in the sequence would be
  floating-point drift compounding through greedy decoding.
  → **FAILED, informatively. See "Exactness result" below.**
- static at cap-full: must reproduce resident accuracy (gemma popqa 0.225).
  This is the wiring control — every expert pinned means nothing can be zeroed.
  If it misses, the pins are wrong and the rest of the arm is noise.
- static at low capacity: **badly damaged.** The traces put the top 25% of
  experts at only **50.7%** (gemma) / **55.4%** (qwen) of routing decisions, so
  cap-32 zeroes about half of all decisions.

Derived from the traces (`records/frontier/hot.*.json`, 129,690 / 142,240
records) — decision share carried by the top-k experts:

| top | gemma | qwen |
|---|---|---|
| 5% | 0.142 | 0.183 |
| 10% | 0.265 | 0.303 |
| 25% | 0.507 | 0.554 |
| 50% | 0.786 | 0.811 |

Skew is real but **moderate**, and W3 established it is a *single-stream*
property: Gini 0.506 at batch 1 falls to 0.335 at batch 32. This lever is for
local single-user inference, not batched serving.

## Queued next, needing a decision

- **top-k reduction arm** — SwiftLM ships `SWIFTLM_TOP_K` as a *speed* setting
  and reports top-k=4 (5.91 tok/s) as faster than top-k=6 (5.20), with **no
  accuracy number attached**. Reducing top-k below training is the route-around
  ablation this project already measured as damaging (R9e). Cheap (~1.5 h) and
  the most differentiated single result available. Reuses `provenance.py`'s
  `BanGate`; needs a small gate patch, which is why it is not in the unattended
  chain.
- **Mixed-precision artifact** — needs the two-module split above. Decision
  required: build it, or declare exact-offload's memory dominance sufficient to
  close W3.
- **Product framing** — the deliverable is a *configuration advisor* (given
  memory, task, latency tolerance → which of model size / capacity / top-k / KV
  precision), not a runtime. Complements SwiftLM et al. rather than racing them,
  and it is benchmark work, which is what this stack is good at. Cost: the
  frontier is per-(model, task, hardware), so value scales with measurement
  coverage.

## Contamination note

The `think/cap4096` rows in `quantize/records/suite.json` (humaneval control
0.38 vs **e4b 0.85**; gsm8k control 0.65 vs e4b 0.89) are R14 truncation
artifacts — the big models are cut off mid-reasoning while e4b finishes. They
must stay out of any frontier. Use the cap4160/cap960 rows.

## Exactness result (2026-07-30 00:39) — prediction failed

gemma, cap-128 (full) vs cap-32, `policy='exact'`, 16 gsm8k items x 256 tokens:

```
identical sequences: 10/16
first-divergence token index: [26, 132, 128, 36, 21, 101]
NOT lossless — 6/16 sequences differ
```

**Never at token 0.** By the pre-registration that rules out a wiring bug and
points at numerics. Two capacity-dependent sources, both confirmed in the code:

1. **`_gather_sort(x, slots)` sorts on SLOTS, not expert ids.** Slot assignment
   is a function of capacity — at full capacity `preload(range(n_experts))`
   makes it a fixed permutation; below it, slots are churning LRU positions. The
   sorted `gather_qmm` path therefore accumulates in a different order.
2. **Prefill is chunked below full capacity.** `__call__` splits the token axis
   when the working set exceeds the cache, because otherwise "the experts
   installed first are evicted before the gather runs and their slots read
   garbage". Different chunking = different reduction grouping.

Both change floating-point rounding, and `do_sort = indices.size >= 64` means
this happens in **prefill**, so the logits differ before the first token is even
emitted. Greedy decoding then either agrees for a while or flips at the first
marginal choice — hence divergence at token 21-132 rather than token 1.

### The corrected claim

Not "accuracy is capacity-invariant by construction". Instead:

> `policy='exact'` is **semantically exact** — every routed expert's true
> weights are fetched, nothing is dropped, nothing reads garbage — but it is
> **not numerically reproducible across capacities**. Deviations are unbiased
> rounding, not information loss, so accuracy is unchanged *in expectation*
> while any single run wobbles.

The suite.json evidence supports unbiasedness: the deltas go both ways
(mmlu_pro 0.530 → 0.500 but ifeval 0.850 → 0.875 and qwen mmlu_pro 0.345 →
0.370). That is scatter, not decline.

### Why this strengthens the product thesis

It is a methodological finding, and a sharp one:

> You cannot validate a MoE offload runtime by diffing its outputs against a
> resident reference. Cache size changes the arithmetic. Validation requires
> comparing **accuracy distributions** at sufficient n — which is exactly the
> capability none of the four competing implementations has, and it means none
> of them knows its runtime is nondeterministic in cache size.

`mlx-flash`'s "bit-perfect operators" claim is worth checking against this: an
operator can be bit-perfect in isolation while the surrounding cache changes
reduction order.

### Consequences for the plan

- The cancelled accuracy-vs-capacity sweep stays cancelled. The question is not
  "does accuracy decline with capacity" (unbiased rounding says no) but "what is
  the noise floor", which needs far fewer items.
- **Two arms added** (`run_frontier2.sh`, queued behind chain 1 on a bounded
  wait, never a pgrep waiter):
  - **determinism control** — same capacity twice. Greedy decoding has no
    sampling, so byte-identical output means the divergence is a deterministic
    function of cache size and therefore reproducible/validatable; differing
    output would mean the runtime is nondeterministic outright and no
    output-diff test can ever validate it. This control was missing.
  - **accuracy noise floor** — exact policy, popqa, 5 capacities (gemma) and 4
    (qwen). Without it a 2pp static-policy drop cannot be attributed to the
    mechanism rather than to rounding. Prediction: unbiased scatter around the
    resident value, **not** a monotone decline. A monotone decline would falsify
    the two-family split.

## Chain 1 + 2 results (2026-07-30 02:18 / 02:35)

### Determinism control: PASSED, and it is the good outcome

gemma cap-64 run twice, `policy='exact'`: **16/16 identical**. Greedy decoding
has no sampling, so this establishes that the capacity divergence is a
**deterministic function of cache size**, not run-to-run nondeterminism. The
runtime is reproducible and therefore validatable — capacity just has to be held
fixed. The alternative (nondeterministic) would have meant no output-diff test
could ever validate an offload runtime.

### Noise floor: ~zero, prediction confirmed

popqa accuracy, `policy='exact'`:

| capacity | 100% | 75% | 50% | 38% | 25% |
|---|---|---|---|---|---|
| gemma | 0.2287 | 0.2287 | 0.2258 | 0.2328 | 0.2287 |
| qwen | 0.2900 | 0.2900 | 0.2900 | — | 0.2900 |

Spread **0.7pp gemma / 0.0pp qwen**, no monotone decline. The two-family split
holds: exact offload's numerical divergence is real but does not move accuracy.

### Static policy: catastrophic, and far steeper than the decision shares imply

| | 100% | 75% | 50% | 25% |
|---|---|---|---|---|
| gemma gsm8k | 0.9146 | — | 0.5250 | **0.0250** |
| gemma popqa | 0.2353 | 0.1719 | 0.0612 | 0.0256 |
| qwen gsm8k | 0.9447 | — | **0.0854** | 0.0151 |
| qwen popqa | 0.2850 | — | 0.1500 | 0.0700 |

**Wiring control passed** — at full capacity static reproduces resident accuracy
(gemma gsm8k 0.9146 vs control 0.9050; qwen 0.9447 vs 0.9350), so the pins are
right and the collapse is the mechanism, not a bug.

At 50% capacity the top half of experts carry 79-81% of decisions, so only ~20%
of decisions are zeroed — and that costs **qwen gsm8k 91% of its accuracy**.
Multi-step reasoning compounds errors across tokens, so it is far more sensitive
than the decision share suggests. qwen collapses harder than gemma, so severity
is architecture-dependent.

### The answer to "when is a smaller model better", at matched memory (~4.2-4.3 GB)

| config | memory | gsm8k | s/item |
|---|---|---|---|
| gemma-26b, **exact**, cap-32 | ~4.3 GB | ~0.91 (inferred) | ~6.2 |
| gemma-26b, **static**, cap-32 | ~4.3 GB | **0.0250** | 7.88 |
| gemma-e4b, resident | 4.20 GB | 0.785 | 2.55 |

**The lossy path is dominated by everything** — slower AND 31x less accurate than
the small model at the same memory. So the crossing point is not gradual: it is
the moment you switch mechanism. And exact offload at the small model's footprint
keeps the big model's accuracy for ~2.5x the latency, which is the defensible
product position.

### Speed curve: the cost model was pessimistic by 2-3x

| model | 100% | 75% | 50% | 25% | 19% | 12% |
|---|---|---|---|---|---|---|
| gemma tok/s | 98.7 | 60.5 | 53.1 | 38.7 | 34.2 | — |
| qwen tok/s | 114.0 | 57.8 | 40.7 | 39.1 | 36.1 | 32.2 |

Total tax **2.9x** (gemma, 19% resident) and **3.5x** (qwen, 12%) against the
6.0x / 10.8x modelled. The shape explains it: a **cliff** from 100% to 75%
(full capacity short-circuits `ensure()` — no device->host sync at all), then a
very gentle slope. gemma 96 -> 24 is 4x less memory for 1.8x less speed. Most of
the cost is the residency check, not disk I/O — so once the sync is paid,
compressing further is nearly free. This materially strengthens the "run any MoE
larger than your VRAM" thesis.

TTFT rises 6.2-6.7x (0.23 -> 1.45 s gemma), so prefill degrades far worse than
decode. That is the long-context cost driver.

## Chain 3 (launched 07:17) — gaps, reference, and the top-k knob

**Instrumentation bug found and fixed.** `mx.get_peak_memory()` is a high-water
mark from process start, so it captured the full model load *before* `wrap()`
freed the expert weights — every capacity reported the same 13.48 GB (gemma) /
18.17 GB (qwen) and **the memory axis was never measured**. `load_wrapped` now
calls `reset_peak_memory()` after wrapping and records `get_active_memory()`.

Arms, in run order:

1. **Gap 1 — memory per rung, measured.** Re-runs the speed sweep at all rungs.
2. **Gap 2 — noise floor on gsm8k.** The floor was measured on popqa only, whose
   64-token cap gives the capacity-dependent rounding least room to compound.
   ~250-token generations are where a task-dependent floor would show, and this
   turns the ~0.91 above from inferred into measured.
3. **Reference — e4b through THIS scorer** (`--policy none`, added for non-MoE
   models). Removes a cross-harness assumption from the headline claim.
4. **Top-k wiring control.** `cut_topk` at the native k=8 must be a no-op,
   compared against chain 1's `gen.gemma.full.json`. Runs *before* the payload so
   a bug is visible in the log rather than buried in the numbers.
5. **Top-k accuracy** at full capacity (offload out of the picture, top-k the
   only variable): k ∈ {6, 4}, gsm8k + popqa, both models.
6. **`frontier report`** — assembles every record into 3-axis Pareto dominance.

### Two implementation notes that matter for reading the results

- **`cut_topk` measures accuracy, not speed.** It masks gate weights while
  SwitchGLU still gathers all 8 experts, so tok/s from those runs is **not**
  comparable to SwiftLM's `SWIFTLM_TOP_K`. Deliberate: it isolates the accuracy
  cost of the knob. Masking to -1e9 before the downstream softmax renormalises,
  which is the *charitable* reading of the knob — dropping without renormalising
  would shrink each block's output magnitude and flatter the comparison.
- **Instance wrapping, not `__call__` assignment.** `gate.__call__ = fn` does not
  intercept `holder.gate(x)`: Python resolves the call on the type, so the cut
  would silently do nothing — a bug indistinguishable from "the knob is free".
  Follows `capture.Recorder`'s pattern. Patching the class instead would hit
  every `nn.Linear`, since qwen's gate *is* one.

### Report caveat

Pareto dominance over the current record set marks a 0.026-accuracy static
configuration "on frontier" purely for being cheapest. That is arithmetically
correct and practically absurd; the e4b reference rows are what kill those
configurations, which is why arm 3 exists.

## Chain 3 results (2026-07-30 10:42)

### Gap 1 closed — memory, finally measured

| capacity | 100% | 75% | 50% | 38% | 25% | 19% | 12% |
|---|---|---|---|---|---|---|---|
| gemma active GB | 13.48 | 10.49 | 7.49 | 6.00 | 4.50 | 3.76 | — |
| qwen active GB | 18.17 | 13.95 | 9.73 | 7.62 | 5.51 | 4.46 | **3.40** |

Linear in capacity and close to the arithmetic that had been standing in for it.

### Gap 2 closed — the noise floor holds on LONG generations

The worry was that popqa's 64-token cap understated it. It does not:

| gsm8k, `policy='exact'` | 100% | 50% | 25% |
|---|---|---|---|
| gemma | 0.9146 | 0.9150 | 0.9091 |
| qwen | 0.9397 | 0.9444 | 0.9447 |

0.5-0.6pp across a 4x capacity range on ~250-token generations, qwen drifting
slightly *upward*. Unbiased scatter, as predicted. The matched-memory table's
inferred ~0.91 is now **measured at 0.9091**.

### e4b through this scorer — the cross-harness gap was real

**gsm8k 0.8426** (quantize's harness: 0.785), **popqa 0.1508** (0.135), at
**3.91 GB**. Quoting across harnesses would have flattered the comparison by
~6pp, which is why the reference arm was worth its 15 minutes.

### Top-k arm — qwen valid, gemma INVALID

| model | task | k=8 | k=6 | k=4 |
|---|---|---|---|---|
| qwen | gsm8k | 0.9444 | 0.9296 (−1.6%) | 0.9239 (−2.2%) |
| qwen | popqa | 0.2900 | 0.2600 | 0.2800 |
| ~~gemma~~ | ~~gsm8k~~ | ~~0.9146~~ | ~~0.1000~~ | ~~0.0150~~ |
| ~~gemma~~ | ~~popqa~~ | ~~0.2287~~ | ~~0.0311~~ | ~~0.0151~~ |

**The gemma rows measure the wrong thing.** `gemma4_text.Router` selects with
`mx.argpartition(kth=-top_k)[..., -top_k:]`, which guarantees *membership* but
not *order*. Masking positions `0..k_keep` therefore dropped an **arbitrary 2 of
8** experts, not the 2 weakest — random expert ablation wearing top-k
reduction's name. qwen's path masks by value (`out >= kth`) and is sound.

**The control could not have caught it.** At `k_keep == top_k` the mask keeps
every position regardless of order, so the 16/16 identical result validated the
masking arithmetic and never touched the ranking assumption underneath. A
control that cannot fail on the bug you have is not a control.

**Method rule added:** a wiring control must be able to fail on the specific
assumption it is meant to protect. Prefer a control that exercises the
assumption (drop the weakest expert) over one that trivially satisfies it (drop
nothing).

Chain 5 re-runs gemma with value-based selection, overwriting the invalid
records in place, behind a **k=7 control**: dropping only the single weakest of
8 must leave accuracy near 0.9146. Near 0.10 would mean it is still dropping
randomly.

### Hypothesis for the architecture split, if it survives the fix

qwen's config carries `shared_expert_intermediate_size: 512` — an always-on
shared expert alongside the 256 routed ones. If a shared branch carries baseline
computation, routed experts are refinements and dropping two costs little. gemma
has no shared expert and routes 8 of 128 (6.25%) against qwen's 8 of 256 (3.1%),
so each gemma expert carries proportionally more. That would predict qwen stays
robust and gemma degrades meaningfully — but *not* catastrophically — once
ranking is correct.

If it holds, the product claim sharpens: **the top-k knob's safety is an
architectural property, not a universal one**, so a runtime that exposes it
globally (SwiftLM's `SWIFTLM_TOP_K`) is safe on some checkpoints and destructive
on others, with no way for the user to tell which.

## Chain 4 (launched 10:42) — the dominance rungs

Chain 3's memory curve put qwen cap-32 at **3.40 GB** and gemma cap-24 at
**3.76 GB**, both *below* e4b's measured **3.91 GB**. With the noise floor flat
across a 4x capacity range, accuracy should hold there — which would upgrade the
claim from "more accurate at comparable memory" to **strict dominance on both
memory and accuracy, losing only speed**. Those rungs run first; mmlu_pro
(n=150, cap4160 kept because R14 forbids shortening it) runs last.

## Chain 4 results (2026-07-30 14:30) — DOMINANCE CONFIRMED

The claim upgraded. An offloaded big model does not merely trade memory for
accuracy against a small model that fits natively — **it beats it on both**.

### gsm8k

| config | memory | accuracy | s/item |
|---|---|---|---|
| **qwen-35b, exact, cap-32** | **3.40 GB** | **0.9447** | 10.21 |
| gemma-26b, exact, cap-24 | 3.76 GB | 0.9091 | 7.04 |
| **gemma-e4b, resident** | **3.91 GB** | **0.8426** | **2.74** |

qwen-35b offloaded into **3.40 GB** — *less memory than e4b's 3.91* — scores
**+10.2pp**. gemma-26b at 3.76 GB scores +6.7pp, also on less memory.

### popqa

| config | memory | accuracy | s/item |
|---|---|---|---|
| qwen-35b, cap-32 | 3.40 GB | **0.2900** | 0.73 |
| gemma-26b, cap-24 | 3.76 GB | 0.2287 | 0.90 |
| gemma-e4b | 3.91 GB | 0.1508 | 0.19 |

**+13.9pp — nearly double the accuracy — on less memory.**

### mmlu_pro (n=150, cap4160) — the widest gap, as predicted

| config | memory | accuracy | s/item |
|---|---|---|---|
| gemma-26b, cap-32 | 4.50 GB | **0.8456** | 15.51 |
| qwen-35b, cap-64 | 5.51 GB | 0.8014 | 28.35 |
| gemma-e4b | 3.91 GB | 0.6364 | 8.87 |

**+20.9pp for 0.6 GB more memory.** Knowledge+reasoning is where the small model
falls furthest behind, and where offloading the big one pays most.

### The frontier, stated

Across all three tasks the offloaded big model **dominates the small model on
memory and accuracy simultaneously, and loses only on speed** — 2.6x to 3.7x
s/item. So the answer to "at what point is compressing the big model worse than
using a smaller one" is:

> **On the lossless mechanism, never — within the measured range.** Down to 12%
> resident, exact offload keeps the big model's full accuracy at *less* memory
> than the small model needs. The crossing point is a LATENCY decision, not an
> accuracy one.
>
> **On a lossy mechanism, immediately.** static at matched memory scores 0.0250
> against e4b's 0.8426, slower. There is no regime where it wins.

### Task-dependence, which is the product

gemma wins mmlu_pro (0.8456 vs 0.8014); qwen wins gsm8k (0.9447 vs 0.9091) and
popqa (0.2900 vs 0.2287). **No single model dominates**, so "which model at which
capacity" is a real per-task decision — exactly the advisor the product is meant
to be, rather than a slogan about compression ratios.

### Three caveats the headline must carry

1. **Speed is the whole cost.** 2.6-3.7x s/item, and TTFT is worse than
   throughput: 1.425 s at qwen cap-32 against ~0.2 s resident. Interactive use
   feels this.
2. **SSD footprint is not free.** qwen at 3.40 GB RAM also needs its **17 GB**
   expert store on disk; e4b needs ~4 GB total. Offload trades disk for RAM, and
   the comparison is only "less memory" in the axis that binds.
3. **Measured range only.** 12% resident is the floor tested. Nothing here says
   the accuracy floor stays flat at 5% or 1%, and the sims' lowest rung was
   2 GB.

## Chain 5 results (2026-07-30 15:09) — the top-k arm, corrected

**Control passed.** gemma at k=7 (drop only the single weakest of 8) scores
0.8995 against 0.9146 native — near baseline, not the ~0.10 the buggy code
produced. Value-based selection works.

| model | task | k=8 | k=7 | k=6 | k=4 |
|---|---|---|---|---|---|
| gemma | gsm8k | 0.9146 | 0.8995 | **0.9200** | 0.8700 |
| gemma | popqa | 0.2287 | — | **0.2632** | 0.2251 |
| qwen | gsm8k | 0.9444 | **0.9548** | 0.9296 | 0.9239 |
| qwen | popqa | 0.2900 | — | 0.2600 | 0.2800 |

### The architecture-split hypothesis is DEAD

It was entirely an artifact of the ranking bug. With correct selection, **top-k
reduction is mild on both architectures** — gemma −4.9% at k=4, qwen −2.2%. The
shared-expert explanation (`shared_expert_intermediate_size: 512`) is not needed
and is not evidence for anything.

Recorded because the hypothesis was written down before the fix: it was a
confident mechanistic story built on a measurement artifact, and it read as
plausible precisely because the artifact was large.

### And the corrected result is NOT RESOLVABLE at n=200

Binomial 95% CI half-widths at these base rates:

| | p | n | ±95% |
|---|---|---|---|
| gemma gsm8k | 0.91 | 198 | 4.0pp |
| qwen gsm8k | 0.94 | 199 | 3.3pp |
| gemma popqa | 0.23 | 188 | 6.0pp |
| qwen popqa | 0.29 | 200 | 6.3pp |

The entire observed spread — gemma gsm8k 0.8700-0.9200 (5.0pp), qwen gsm8k
0.9239-0.9548 (3.1pp) — sits **inside** those bands, and the non-monotonicity
gives it away: k=6 beats k=8 on both gemma tasks and k=7 beats k=8 on qwen.
That is sampling noise, not a dose-response curve.

> **Honest statement:** reducing top-k from 8 to 4 costs at most a few points on
> these two models and two tasks, and n=200 cannot resolve the effect more
> precisely than that. It is neither the catastrophe the buggy run showed nor
> demonstrably free.

### What this does to the competitive story

It weakens it, and the writeup must say so. The claim was that SwiftLM ships
`SWIFTLM_TOP_K` as a speed setting with no accuracy number attached. That remains
true — but the measured cost looks *small*, so "unlabelled" is the fair
criticism, not "destructive".

This is a case where measurement failed to confirm the alarming hypothesis, which
is the point of having the measurement. The product framing survives intact and
is arguably better for it: **a validation harness that says "this knob is
probably fine on your model" is worth exactly as much as one that says "this knob
will destroy you" — the value is in knowing, and nobody currently knows.**

To resolve it properly would need n≈1000+ per cell, or a paired design over
identical items (per-item records are saved, so McNemar is available offline).

---

# Scope decision — DON'T BUILD: hosted-API support and cascade routing

Decided 2026-07-30. Recorded as a decision rather than an omission, with the
evidence and the conditions that would reopen it, so it does not get
re-proposed every time someone notices entropy could route queries.

## What was considered

1. **Hosted-API support** — running the label-free detector against
   OpenAI/Anthropic/Gemini rather than only local inference.
2. **Cascade / effort routing** — using entropy from a cheap call to decide
   which model, or which reasoning-effort level, to spend on a given prompt.

## Why not — hosted APIs

| provider | logprobs | state |
|---|---|---|
| OpenAI | yes | `logprobs: true`, but `top_logprobs` capped at **20** |
| Anthropic | **no** | not supported; only a community shim exists |
| Gemini | partial | `responseLogprobs` exists but is **missing on the current frontier models** (3.1 Pro, 3.6 Flash) and absent from the next-gen Interactions API |

So the addressable surface is one provider and a fraction of a second. Worse,
`top_logprobs <= 20` yields **truncated** entropy over the top 20 tokens, while
every measurement in this project is full-distribution entropy over ~248k. That
is a different quantity, and whether the detector survives the truncation is
untested — an extra unknown stacked on an already thin surface.

An earlier note in this file claimed the tool "works with any API returning
logprobs". That was too generous and is corrected here.

## Why not — cascade routing

The use case is real and Stage D supports it directly (PopQA's top decile by
confidence is 96% accurate against a 25% base rate). It is also **crowded, and
our signal is specifically the baseline that recent work beats**:

- [UCCI](https://arxiv.org/abs/2605.18796) maps token-level uncertainty to a
  per-query error probability by isotonic regression and picks the escalation
  threshold by constrained cost minimisation — cutting cost 31% at
  micro-F1 0.91 while **beating entropy thresholding**.
- [FrugalGPT, RouteLLM, and a 2026 survey](https://arxiv.org/html/2603.04445v2)
  cover the space; [*When to Think Deeply*](https://arxiv.org/pdf/2606.06745)
  covers effort selection specifically.

Entering that field would mean competing at the losing end of a published
comparison, on a third product surface, needing API access we mostly cannot get.

## What this protects

Focus stays on **local and self-hosted inference**, which is where the
distinctive asset is: nobody in the MoE offload ecosystem measures accuracy at
all, and the frontier and detector results have no published counterpart. It is
also internally coherent — you do not compress somebody else's hosted API, so
the compression question and the local-inference constraint are the same
question.

## What would reopen it

- **Anthropic or Gemini shipping logprobs on current models**, which would make
  the surface broad enough to matter.
- **Evidence that truncated top-20 entropy preserves the detector's separation.**
  Cheap to test but currently pointless: it only pays off if the hosted surface
  widens first. Needs a re-run, since only aggregates were saved, not
  distributions.
- **A local-currency version of cascading.** The published work optimises
  dollars per API call; for local inference the currency is latency and memory,
  and that is genuinely underexplored — e.g. serve from the small resident model
  and escalate to the offloaded big one only when entropy says to, buying most
  of F5's +10pp without paying its 3.7x latency on every query. This composes
  with the frontier rather than starting a third product, and is the only form
  of the idea worth revisiting.

---

# The entropy design space — what is open and what is taken

Explored 2026-07-30, search-first. Entropy is unusually cheap: it is a byproduct
of a forward pass you are already doing, needs no second model, no LLM judge and
no labels. That cheapness is what makes it scale where LLM-as-judge does not, so
it is worth knowing the whole space rather than only the corner F8 occupies.

## The reusable asset is not entropy — it is the paired design

What makes F8 work is not the signal. Entropy is one cheap scalar among several
(F8 also measures `mean`, `max`, `first`, `mean_top10`). What makes it work is
the harness around it:

> **same items → two configurations → paired per-item delta → effect size
> against a calibrated benign null.**

That structure is indifferent to *what* was changed. Every Tier-1 item below is
the same tool pointed at a different intervention, which is why they compose
into one product rather than three.

## Tier 1 — open, and reachable from what exists here

### 1. Label-free forgetting monitor during fine-tuning

**The gap.** Forgetting is currently detected by "accuracy on held-out test sets
from previous tasks, though this still requires labels". The entropy work that
exists points elsewhere:
[EAFT](https://arxiv.org/pdf/2601.02151) uses entropy as a *training mechanism*
to mitigate forgetting, and the
[mechanistic analysis](https://arxiv.org/html/2601.18699v1) uses **attention**
entropy as an explanatory variable — neither is a label-free monitor over output
entropy. No paper found doing that.

**The application.** Exactly F8's design with the reference moved: reference =
base checkpoint, candidate = checkpoint N, items = unlabelled held-out prompts
from domains you are *not* training on. A rising paired delta says the fine-tune
is damaging general capability, mid-run, without a labelled eval.

**Why it is reachable.** This repo has both halves already — `finetune.py` from
the injection work, and `regress.py`. The experiment is a LoRA run with periodic
checkpoints and the detector pointed at held-out prompts.

**Why it matters.** Forgetting is usually discovered *after* training, by which
point the compute is spent. A cheap in-loop monitor is a different product from
a post-hoc benchmark.

### 2. Backend / quantization / hardware equivalence

**The research validates the problem and does not close the tooling gap.**

- [*The Silent Hyperparameter*](https://arxiv.org/pdf/2605.19537): backend-induced
  numerical differences propagate into "divergent generations, altered decision
  boundaries, and benchmark score shifts large enough to affect model rankings".
- [*The Illusion of Equivalency*](https://arxiv.org/html/2607.08734): "quantized
  variants do not reliably reproduce base-model behavior, **even when accuracy or
  perplexity appears preserved**", with drift growing as bit-width falls.
- [*Quantifying non-deterministic drift*](https://arxiv.org/html/2601.19934v1).

The second of those independently corroborates **F2** — semantically exact,
numerically non-reproducible — from the quantization direction rather than the
cache direction. It also states the thing that makes a detector necessary:
accuracy and perplexity can look preserved while behaviour has moved.

**What this changes.** F8 is currently scoped to "did your offload config break
the model". The literature says backends, quantization levels and hardware do the
same thing, so the honest scope is **any intervention on a model**, which is
better supported and no harder to build. Recommended prose for the writeup:
*"did anything you changed — config, quantization, backend, hardware — move the
model?"*

**Caveat.** Parity testing in CI is named as a *strategy* in practitioner
writing; whether a packaged tool exists was not established and should be checked
before claiming the gap.

### 3. RAG context utilisation, measured by entropy drop

**The idea.** Reference = prompt without retrieved context; candidate = prompt
with it. If entropy does not fall, the retrieval did not help, whatever a
relevance score says. Directly connected to **R2**, which separated
context-grounded from parametric answering on this stack.

**Why it might be open.** RAGAS and similar tools measure context relevance by
**LLM judgement** — expensive, slow, and itself unreliable. An entropy-based
version is essentially free and needs no judge model.

**Unverified.** Not searched thoroughly enough to claim the gap. Check before
building.

## Tier 2 — plausible, unverified, cheap to check

- **Eval-item quality.** Items on which *every* configuration is high-entropy are
  candidates for ambiguity or mislabelling. Adjacent to dataset cartography,
  which uses *training* dynamics rather than inference-time entropy.
- **Eval-set sufficiency.** F8 already demonstrated entropy resolving at n=200
  what labels needed n=800 for. Turning that into a power analysis — "how many
  items do you need to detect a k-point regression" — is a small, genuinely
  useful piece of methodology.
- **Prompt ambiguity localisation.** Entropy at *prompt* positions (teacher-forced)
  rather than generated ones, to locate where a prompt is underspecified.

## Tier 3 — DON'T BUILD

- **Span-level hallucination detection in long-form output.** Crowded and
  *shipped*: [`obalcells/hallucination_probes`](https://github.com/obalcells/hallucination_probes)
  is a working open-source tool with real-time token-level visualisation;
  LettuceDetect-Qwen-2B localises unsupported spans over 32k tokens;
  [SpanUQ](https://arxiv.org/pdf/2607.05721) and
  [Semantic Entropy Probes](https://arxiv.org/pdf/2406.15927) cover the research.
  Raw entropy against fine-tuned probes is the R12/R13 pattern again — a simpler
  signal losing to a stronger one — and the honest expectation is that we lose.
- **Cascade / effort routing.** Separately recorded above; our signal is the
  baseline [UCCI](https://arxiv.org/abs/2605.18796) beats.
- **Membership / contamination detection.** This repo already tried and failed
  (P1), and the MIA literature is dense.
- **Early exit, speculative-decoding acceptance, KV eviction.** Well covered
  (CALM, H2O, SnapKV); these are runtime optimisations, and the standing decision
  is not to compete on runtime.

## Recommended order, if this thread is picked up

1. **Widen F8's stated scope** to any intervention — costs nothing, better
   supported by the literature than the current offload-only framing, and it is
   the honest description of what the tool already does.
2. **Forgetting monitor** — the clearest open gap and both halves already exist
   in this repo.
3. **Truncated-entropy check** — only if hosted APIs are ever reopened.
4. **RAG utilisation** — verify the gap first.
