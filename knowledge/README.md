# `knowledge/` — the research package

This is the code that produced [FINDINGS.md](../FINDINGS.md). It is **not the
product** — that is [`src/clausius/`](../src/clausius), which is installable,
tested, and depends on none of this.

It is kept for one reason: the findings make claims, and the claims should be
checkable. It is coupled to local model checkpoints and a sibling `quantize`
checkout, so it is not expected to run unmodified on another machine.

```bash
export QUANTIZE_REPO=/path/to/quantize   # defaults to ../quantize
python3 -m knowledge.frontier report     # no model needed
```

Modules marked **superseded** produced negative results. They are here because
[FINDINGS.md Part II](../FINDINGS.md) records eight controlled negatives, and a
record that kept only the wins would invite re-running the dead ends. The code
is the evidence for those, not a suggestion to build on them.

## Infrastructure

| module | what |
|---|---|
| `traces` | read quantize's captured routing traces — stdlib only, no GPU |
| `seam` | locate a model's expert modules and router without knowing the family |
| `probes` | matched recall/derivation probe suites, built to separate mechanism from topic |
| `capture` | capture per-token routing over a probe suite |
| `cot` | where to measure entropy when the answer follows a reasoning chain (R14) |
| `_gl` | thin re-export of the sibling `ghostlight` repo's calibration metrics — read-only, by path |
| `finetune` | run an mlx-lm LoRA arm under a hard memory ceiling |
| `corpus` | build a membership benchmark with ground truth by construction |

## Deployment configuration and the detector — Part III (F1–F13)

| module | what |
|---|---|
| `frontier` | the configuration frontier: exactness, speed, memory, accuracy per config |
| `regress` | the label-free detector, research version — `clausius` is the shipped one |
| `context` | does retrieved context help, measured without labels (**F10: open, does not yet validate**) |

## Correctness signal — Stages A–D (R13–R17)

| module | what |
|---|---|
| `stage_a` | does entropy-based error prediction generalise across task types |
| `stage_b` | entropy vs self-consistency — the honest competitor |
| `stage_d` | product metrics: catch-at-budget, risk–coverage, calibration |
| `popqa` | error prediction on a benchmark with a real error rate |

## Mechanism — the causal routing address (R1, R3, R9, R10)

| module | what |
|---|---|
| `routing` | does expert selection separate recall from derivation, against a real null |
| `provenance` | causal provenance: are a fact's own experts the ones that carry it |
| `generated` | the same question over the model's own generated text |
| `inject` | ground-truth validation using facts the model provably could not have known |
| `identity` | does an individual fact have an address, or just a topic |

## Superseded — negative results, kept as evidence

| module | what it tried | outcome |
|---|---|---|
| `meter` | routing as a classifier for where an answer came from | lost to reading the prompt text |
| `transfer` | does the classifier generalise across suites | it does not — each number was a template |
| `position` | where in the sequence fact identity lives | resolved a caveat; not a usable signal |
| `fabrication` | does routing distinguish retrieval from fabrication | no |
| `errors` | can any signal predict a stated FALSE fact | entropy won; routing was not needed |
| `readout` | read-only fact identification in generated text | ties with reading the text |
| `annotate` | the mechanism meter applied to ordinary text | superseded by the above |
| `prefetch` | early-layer routing to prefetch late-layer experts | signal is real and far too small |
| `entity_disc` | is entity identity encoded, and by what | three designs failed; unexplained |
| `membership`, `detect`, `learned` | was a document in the training data | no — see FINDINGS P1 |
