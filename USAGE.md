# Using clausius

How to run the detector and read what it tells you. For *why* each default is
what it is, see [FINDINGS.md](FINDINGS.md); this file is the operating manual.

- [Choosing prompts](#choosing-prompts)
- [Setting the token cap](#setting-the-token-cap)
- [Reading the output](#reading-the-output)
- [Seeing what changed](#seeing-what-changed)
- [Calibrating your own null](#calibrating-your-own-null)
- [Running it in CI](#running-it-in-ci)
- [What it costs](#what-it-costs)
- [The Python API](#the-python-api)
- [Every flag](#every-flag)

---

## Choosing prompts

**No labels are needed.** A sample of real production traffic is ideal, because
the detector's job is to notice a change in behaviour on the distribution you
actually serve. Sixty is a good starting number, and
[`examples/prompts.jsonl`](examples/prompts.jsonl) ships sixty you can use
immediately.

`compare` refuses to run on fewer than **20 surviving paired items**, and
truncated items are dropped before that count is taken — so 60 keeps the floor
out of reach on a first attempt while 25 sits right on it.

**Composition changes the answer.** Compression does not degrade capabilities
evenly: short factual recall loses 1.5pp where structured generation loses
18–21pp at the same bit width, a ~14x difference in what the same setting
appears to cost (FINDINGS §4). A prompt set of only short answers will
understate damage. The shipped example set is six blocks of ten — factual
recall, multi-step arithmetic, instruction following, technical explanation,
code, business writing — for that reason.

One prompt per line, either JSONL with a `prompt`, `text` or `input` field, or
plain text:

```jsonl
{"prompt": "What is the capital of Australia?"}
{"prompt": "Summarise the following in exactly three bullet points: ..."}
```

---

## Setting the token cap

**`--max-tokens` is the setting most likely to make a first run unusable.** The
512 default suits short-answer work. A mixed instruction-following set will blow
straight through it — the shipped example set truncates 47 of 60 items at 512 on
a healthy model.

Items that hit the cap are **dropped at compare time**, so a cap that is too
tight silently shrinks your sample until `compare` refuses to run.

**Capture generously once rather than twice.** Every capture prints the
truncation it would have had at every *tighter* cap:

```
→ ref.json
truncation at this and every tighter cap:
    cap   512:  47 truncated,  13 survive  below compare's floor of 20
    cap  1024:  26 truncated,  34 survive
    cap  1536:  12 truncated,  48 survive  <- this run
    12 item(s) reached the 1536 cap; their true lengths are unknown, so this
    table cannot be extended above it
```

That table is exact downward and silent upward, and the asymmetry is the point:
a capture can be re-analysed at any tighter cap because every finished item's
true length is recorded, but never at a looser one, because a truncated item
never revealed how long it wanted to be. **So capture high.** If the reference
alone already falls under the floor, `capture` exits non-zero and says so rather
than letting you pay for a second capture that cannot help.

---

## Reading the output

```
REGRESSION  (max d_z = +0.654 [95% CI +0.31, +1.02], threshold 0.3, one-sided)
  compared 40 paired items, dropped 20 truncated
  all signals: max +0.65  p90 +0.67  mean +0.79  mean_top10 +0.96  first +0.06  gen_len +0.18
```

**The verdict.** `REGRESSION` or `clean`, decided by one thing only: whether the
chosen signal exceeds the threshold. `compare` exits **1** on a regression and
**0** on a clean pair, so it drops into CI without glue.

**`d_z`** is a standardised effect size — the mean of the paired per-item
entropy differences, divided by their standard deviation. Positive means the
candidate is *less* certain than the reference. It has no units and does not
convert to accuracy.

> **`d_z` is ordinal, not proportional.** On a matched ladder, 3-bit lost **25×
> more accuracy** than 4-bit (−56.6pp against −2.2pp) and read only **2× the
> d_z** (+1.70 against +0.84). Trust the ordering; do not read a magnitude as a
> quantity of damage. "Something moved, and roughly how hard" is supportable;
> "you lost k accuracy points" is not.

**The interval** is a 95% bootstrap over resampling your *prompts*. It answers
"would a different sample of prompts have given a different number?" — which is
how you tell a real marginal result from noise. It is seeded, so repeated runs
on identical inputs give identical output.

It does **not** cover benign-configuration variation, which is what the ±0.10
null measures. A tight interval around +0.15 means *this prompt set reliably
reads +0.15*, not that the change is safe.

**The six signals** are aggregations of the same per-token entropy. No single
one dominates — `max` stays monotone where `p90` inverts on long-output tasks,
and each wins on some arms. They are all reported so the verdict can be seen to
be robust rather than an artifact of one choice. If they disagree sharply,
treat the result as unresolved rather than picking the one you like.

Watch `gen_len` in particular: if it is the only signal that moved, the
"difference" may be generation length rather than uncertainty.

**`dropped N truncated`** tells you which prompts were measured, not just how
many. The filter removes long-output items, and those are the family most
sensitive to compression — so a large drop count means the surviving sample is
biased toward the *least* sensitive prompts. Raise the cap rather than accept it.

### What to do with a verdict

| result | what it means | next step |
|---|---|---|
| `clean`, tight interval | no detectable change on this traffic | ship |
| `clean`, wide interval | underpowered, not established | more prompts |
| `REGRESSION`, interval clear of the threshold | something changed | `--show` to see what |
| `REGRESSION`, interval straddling the threshold | marginal; may be a small effect or noise | more prompts, and calibrate your null |
| signals disagree | unresolved | inspect items, do not average them |

---

## Seeing what changed

The verdict says something broke. `--show N` says *what*:

```bash
clausius compare ref.json cand.json --show 3
```

```
  item 132  Δmax +1.77  (1.73 -> 3.50)
    prompt: Shiloh is 44 years old today. In 7 years, he will be three times ...
    ref   : <|channel>thought * Shiloh's current age: 44 years old. * Timeframe ...
    cand  : <|channel>thought * Shiloh's current age: 44 years old. * Timeframe ...
```

It reads recorded data, so it costs nothing and needs no model. Because `d_z` is
ordinal, this per-item view is how you form your own judgement of severity.

---

## Calibrating your own null

The 0.3 threshold comes from 13 benign configurations measured on **one stack**,
and the null is not a universal constant — the same benign 8-bit checkpoint
against the same bf16 reference reads **+0.172** on a mixed instruction set and
**−0.062** on gsm8k (FINDINGS F14c). On a new backend, a new quantizer or a
markedly different prompt set, measure your own floor first:

```bash
# two configurations you have independent reason to believe are equivalent —
# a benign precision change, a runtime version bump, a cache setting
clausius capture --model ./cfg-a --prompts prompts.jsonl --out null_a.json --max-tokens 1536
clausius capture --model ./cfg-b --prompts prompts.jsonl --out null_b.json --max-tokens 1536
clausius compare null_a.json null_b.json
```

Whatever |d_z| that produces is **your** false-alarm floor. If it exceeds 0.10,
scale the threshold with it — the published 0.3 is ~3× a null of 0.10, so a
floor of 0.2 argues for `--threshold 0.6`. Use two or three benign pairs, not
one: a single pair gives you a point, not a floor. The interval matters here
too — a null estimated on 25 items with a CI half a unit wide has established
nothing.

This is the procedure that produced the 0.3 default, and it is the honest answer
to "does the threshold transfer to my setup" on any stack this project has not
measured.

---

## Running it in CI

`compare` exits non-zero on a regression, so the gate is the command itself:

```bash
clausius capture --model "$MODEL" --prompts ci/prompts.jsonl \
    --out "$ARTIFACTS/cand.json" --max-tokens 1536
clausius compare artifacts/golden-ref.json "$ARTIFACTS/cand.json" --json | tee report.json
```

**Managing the reference.** Capture the configuration you trust once, commit
`ref.json` as a build artifact, and compare every candidate against it.
Re-baseline **deliberately** — when you accept a new configuration, not when a
run goes red.

`compare` **rejects** captures made on different prompt sets rather than
silently comparing them, so changing your prompts forces a new reference. That
is intended behaviour, not an obstacle to route around: pairing two different
prompt sets would compare datasets, not configurations.

**Captures are deterministic**, which is what makes the gate meaningful. Greedy
decoding over fixed weights reproduces exactly, across processes and not merely
within one — a capture at cap 1536 predicted 47/60 truncations at cap 512, and a
separate run actually at 512 truncated exactly 47. It also means comparing a
config against *itself* yields exactly zero rather than a noise floor: the ±0.10
null comes from benign *configuration* changes, not from measurement noise.

---

## What it costs

Measured on `gemma-4-26b-a4b-it` (26B MoE) on an M5 Max:

| configuration | per prompt | 60 prompts | comparison (two captures) |
|---|---|---|---|
| 4-bit, ~13 GB | ~8 s | ~8 min | ~16 min |
| 8-bit, ~28 GB | ~11 s | ~11 min | ~22 min |
| bf16, ~52 GB | ~18 s | ~18 min | ~36 min |

Generation dominates, so cost scales with what your prompts elicit rather than
with `--max-tokens` directly: tripling the cap from 512 to 1536 cost 2.2×, not
3×, because items that finish early are unaffected.

---

## The Python API

The same three steps, and the extension point for configurations the CLI cannot
construct — a patched runtime, a custom cache, an offload wrapper:

```python
from clausius import capture, compare, top_movers

ref  = capture("./model", prompts, tag="v1")
cand = capture("./model", prompts, tag="v2", adapter="./my-lora")
print(compare(ref, cand))

for m in top_movers(ref, cand, n=3):
    print(m['i'], m['delta'], m['cand_text'][:120])
```

Pass `model_obj`/`tokenizer` to measure an already-loaded model:

```python
model, tok = my_patched_loader("./model")
cand = capture(None, prompts, tag="patched", model_obj=model, tokenizer=tok)
```

clausius does not manage configurations. It compares whatever two runs you hand
it, which is why a LoRA checkpoint, an inference backend and an offload setting
all work through the same three commands.

---

## Every flag

### `clausius capture`

| flag | default | notes |
|---|---|---|
| `--model` | required | path or HF id |
| `--prompts` | required | JSONL or plain text, one per line |
| `--out` | required | where to write the capture |
| `--max-tokens` | 512 | **set this for your traffic** — see above |
| `--adapter` | — | LoRA adapter path |
| `--limit N` | all prompts | takes only the first N. Rarely worth it: a full capture at a generous cap *is* the reference, whereas a sampled probe is thrown away |
| `--raw` | off | skip the chat template. For base models; on an instruct model this causes runaway generation |
| `--tag` | filename stem | label recorded on the capture |

### `clausius compare`

| flag | default | notes |
|---|---|---|
| `--signal` | `max` | one of `max`, `p90`, `mean`, `mean_top10`, `first`, `gen_len` |
| `--threshold` | 0.3 | ~3× the measured null; raise it if your own null is wider |
| `--show N` | 0 | print the N items whose entropy moved most |
| `--two-sided` | off | also flag entropy *decreases*. Known false positive: a pure confidence change with zero accuracy impact trips it |
| `--keep-truncated` | off | do not drop capped items. Dilutes the effect and lets generation length leak in — use when an arm cannot survive the filter, and say so when reporting |
| `--json` | off | machine-readable output including the interval |

Exit codes: **0** clean, **1** regression, **2** (on `capture`) the capture
cannot support a comparison.
