# `knowledge/` — the research package

This is the code that produced [FINDINGS.md](../FINDINGS.md). It is **not the product** — that is [`src/clausius/`](../src/clausius), which is installable, tested, and depends on none of this.

It is kept for one reason: the findings make claims, and the claims should be checkable. It is coupled to local model checkpoints and an external artifact store, so it is not expected to run unmodified on another machine.

Code it needs from the author's other projects is **vendored** under `_vendor/`, not imported, so this repository is self-contained: clone it and every import resolves.

```bash
export CLAUSIUS_ARTIFACTS=/path/to/artifacts   # defaults to ../artifacts
python3 -m knowledge.frontier report     # no model needed
```

Modules marked **superseded** produced negative results. They are here because [FINDINGS.md Part II](../FINDINGS.md) records eight controlled negatives, and a record that kept only the wins would invite re-running the dead ends. The code is the evidence for those, not a suggestion to build on them.

## Datasets — none are bundled

Every task fetches its data from the Hugging Face Hub at run time, so `records/` holds only derived measurements. Reproducing an arm means obtaining the dataset from its original source under that source's terms; see NOTICE for the full list and licenses.

Most load with no setup. **Two are gated** and need a one-off approval, which is granted automatically once you accept the terms on the dataset page:

```bash
huggingface-cli login
# then accept the terms at:
#   https://huggingface.co/datasets/Idavidrein/gpqa     (GPQA — CC BY 4.0)
#   https://huggingface.co/datasets/cais/hle            (HLE  — MIT)
```

Without that, `load_dataset` raises a gated-repo error naming the dataset; every other task is unaffected.

**IFEval needs its scorer fetched separately.** The dataset itself is open (`google/IFEval`, Apache-2.0), but scoring instruction adherence requires Google Research's `instruction_following_eval` registry, which is Apache-2.0, not on PyPI, and not bundled here. Fetch it once as a package named `_ifeval_official`, importable from wherever you run:

```bash
mkdir -p _ifeval_official && touch _ifeval_official/__init__.py
B=https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval
for f in instructions.py instructions_registry.py instructions_util.py; do
    curl -sSo _ifeval_official/$f $B/$f
done
```

`load_items('ifeval', ...)` checks for it **before** the run starts and exits with these instructions if it is missing — deliberately, because without it every item scores as no-verdict, and a no-verdict read as a boolean is `False`. An unscoreable arm would otherwise report near-zero accuracy that looks exactly like catastrophic damage.

**GPQA carries one extra condition.** Its authors ask that the dataset not be posted in plain text online, to keep it out of future training corpora. This repository honors that: no GPQA question, option or gold answer appears here, and the model's generated answer text has been stripped from `records/stage_a.gpqa*.json`, leaving correctness, entropy and the domain label. That costs nothing for reproduction — the questions were never in `records/` to begin with, since the records are outputs, not inputs.

## Infrastructure

| module | what |
|---|---|
| `traces` | read captured routing traces from the artifact store — stdlib only, no GPU |
| `seam` | locate a model's expert modules and router without knowing the family |
| `probes` | matched recall/derivation probe suites, built to separate mechanism from topic |
| `capture` | capture per-token routing over a probe suite |
| `cot` | where to measure entropy when the answer follows a reasoning chain (R14) |
| `_gl` | flat re-export of the vendored calibration metrics (`_vendor/calibration.py`) |
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
| `stage_a` | does entropy-based error prediction generalize across task types |
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
| `transfer` | does the classifier generalize across suites | it does not — each number was a template |
| `position` | where in the sequence fact identity lives | resolved a caveat; not a usable signal |
| `fabrication` | does routing distinguish retrieval from fabrication | no |
| `errors` | can any signal predict a stated FALSE fact | entropy won; routing was not needed |
| `readout` | read-only fact identification in generated text | ties with reading the text |
| `annotate` | the mechanism meter applied to ordinary text | superseded by the above |
| `prefetch` | early-layer routing to prefetch late-layer experts | signal is real and far too small |
| `entity_disc` | is entity identity encoded, and by what | three designs failed; unexplained |
| `membership`, `detect`, `learned` | was a document in the training data | no — see FINDINGS P1 |
