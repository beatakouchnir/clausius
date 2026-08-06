# FINDINGS — the experimental record

The full evidence behind [README.md](README.md), positives and negatives in the
order they were established. **Read the negatives**: two of them are load-bearing
for the positives. R8's failure is why R9 was reframed as a causal test rather
than a classification one; R6c is why membership detection was dropped. A record
that kept only the wins would invite re-running the dead ends.

Sections are labeled R1–R9/P1 because the code docstrings reference those
labels; Part III uses F-labels.

---

# Summary

Two threads, both on consumer hardware (M5 Max, 128 GB), across five model
families.

**Deployment configuration (Part III).** For Mixture-of-Experts models, expert
offloading to SSD is **accuracy-preserving down to 12% expert residency** — and
at that point the offloaded 35B model uses *less memory* than a 4B model that
fits natively while scoring substantially higher:

| gsm8k | memory | accuracy | s/item |
|---|---|---|---|
| qwen-35b, exact offload, 12% resident | **3.40 GB** | **0.9447** | 10.21 |
| gemma-e4b, fully resident | 3.91 GB | 0.8426 | 2.74 |

Same ordering on popqa (+13.9pp) and mmlu_pro (+20.9pp). The entire cost is
latency: throughput falls as roughly **memory^0.5** after an initial cliff, and
time-to-first-token degrades 2-3x faster than throughput. Meanwhile the *lossy*
alternative — dropping non-resident experts, which superficially looks like a
memory setting — costs **91% of gsm8k accuracy** at 50% residency and is
dominated on every axis by simply using a smaller model.

**Label-free regression detection (F8).** Whether a deployment config silently
degraded a model can be detected **without any labeled eval set**, by comparing
the model's own per-token entropy distribution against a reference config on the
same unlabeled prompts. Six configurations of independently known damage, two
architectures, three failure mechanisms:

| config | true damage | entropy d_z | flagged | correct |
|---|---|---|---|---|
| qwen exact cap-64 | +0.005, n.s. | −0.05 | no | ✓ |
| gemma exact cap-32 | −0.006, n.s. | −0.08 | no | ✓ |
| qwen top-k=4 | **−0.023, p=0.004** | +0.56 | yes | ✓ |
| qwen static cap-128 | −0.854 | +1.20 | yes | ✓ |
| gemma static cap-32 | −0.890 | +1.72 | yes | ✓ |
| qwen static cap-64 | −0.925 | +2.54 | yes | ✓ |

6/6 on gsm8k, no false positives, effect sizes monotone in true damage — and a
second task (popqa, 64-token cap) extends the benign null to **13 configurations
within |d_z| ≤ 0.10**, which calibrates the threshold at 0.3 rather than the 0.5
first used. Sensitivity is the weaker half: on gemma/popqa a −16.8pp config
scores +0.49. The benign arms are the load-bearing control: a 3-3.3x memory reduction changes ~25% of
generations *textually* and moves no signal — so it detects damage, not change.
And it is more sensitive than the labeled alternative: top-k=4's −2.3pp
regression is flagged at **n=200**, where paired McNemar on labels needs
**n=800** to reach significance.

**Mechanism (Part I).** Expert routing carries a **causal, fact-level address**:
ablating the experts a fact selects damages that fact far more than a paraphrase,
a same-relation fact, or a random control (R9), replicated on two architectures,
localised to the late third of layers, distributed and redundant rather than
point-like, surviving into the model's own generated text, and validated against
ground truth by injecting facts the model provably could not have known.

**What did not work (Part II).** Eight controlled negatives in which routing lost
to a simpler signal — usually reading the prompt text, or predictive entropy.
The structural reason: routing is downstream of the residual stream and
downstream of the prompt, so it is bounded by both.

## How to read this record

Every claim here is scoped to what was measured, and the failures are kept in
deliberately. Three specific corrections are load-bearing:

- **A planned 16-50 GPU-hour sweep was canceled after reading the
  implementation** — `policy='exact'` fetches missing experts from disk before
  computing, so the sweep would have measured a guaranteed flat line (F1).
- **A confident mechanistic hypothesis was killed by its own bug fix.** A large
  gemma-vs-qwen split in top-k sensitivity, and a plausible shared-expert
  explanation for it, evaporated once a ranking error was corrected (F7).
- **An instrumentation bug meant a whole axis was never measured.**
  `mx.get_peak_memory()` is a high-water mark that captured the model load
  before the cache freed it, so every configuration reported identical memory
  (F4).

Two method rules came out of those, and both are applied throughout:
*read a knob's implementation before scoping a sweep over it*, and *a wiring
control must be able to fail on the assumption it is meant to protect.*

---

# Tooling landscape — what exists to build on

Searched 2026-07. Four layers of stack, and the gap is not where the research is.

| layer | state | implication |
|---|---|---|
| **application** | grounding/hallucination detection IS productized — Guardrails AI, LangKit, RAGAS, HaluGate, Tonic Validate, ConSens | but the SOTA production method is *delegating judgment to another LLM*: expensive, slow |
| **research** | internal-state probes work well on **dense** models — SAPLMA, INSIDE, MIND, HaloScope, HalluShift, MultiHaluDet (98.55% AUROC on HaluEval/TriviaQA) | none use routing; a routing detector would compete against a stronger signal on a smaller market |
| **interp libraries** | TransformerLens / nnsight / nnterp list MoE router logits as **future work** | no research tooling for routing either |
| **serving** | vLLM: *"does not natively support returning router selections (router logits or expert assignments) for each layer during inference"*. Workarounds need deep model-code edits and there is **no standardized interface across MoE architectures**. SGLang's routing work is load-balancing, not observability | **the real gap** |

MoE-specific research tools exist but are research-grade: `MoE_analysis`
(expert-level interpretation scripts), MixtureKit (Streamlit routing
visualiser).

## Consequence for what to build

**Not a detector.** Hidden-state probes are already at 98.55% AUROC on dense
models, and this project's own R6c and R11 both show routing losing to simpler
signals whenever the task is scoring. Competing there is competing on the weak
axis against a larger market.

**The gap is plumbing.** Nobody can get routing out of a production MoE server
portably. `seam.py` already does exactly that — resolving routers across
families by class, covering qwen's `mlp.gate`, gemma's `layer.router`, and
gemma's internal top-k via `Router.proj`. vLLM's stated blocker ("no
standardized interface across different MoE architectures") is the thing it
solves.

**And routing is small in a way hidden states are not.** 40 layers x 8 experts
of int16 is ~640 bytes/token; 40 x 2048 float hidden states is ~320 KB — roughly
**500x less**. You can log routing for every token of every request. You cannot
log hidden states. That is the property that makes a product possible, and it
fits what this project actually found: routing is informative about *what the
model did* and weak as a *predictor*.

So the deliverable shape is **provenance telemetry cheap enough to leave on**,
not a better detector.

---

# Prior art — what is ours and what is not

Checked after the fact, which was the wrong order. The literature search that
preceded the membership work (and saved a month) was not repeated before R9,
R9c or R9d. Doing it now:

| result here | prior work | overlap |
|---|---|---|
| **R9** expert-level causal ablation for factual recall in MoE | *Expert-Aware Causal Tracing of Factual Recall in Sparse MoE LMs* (arXiv 2606.03780), on Qwen3-30B-A3B-Base (48L, 128E, top-8) and Mixtral | **high** — same setting, near-identical model family |
| **R9d** distributed, redundant, no single layer necessary | Hochman, Shapira & Goldberg, *Factual Retrieval in LLMs Is a Redundant, Distributed and Non-Contiguous Process* (arXiv 2606.21345) | **high** on the conclusion; they use dense models and iterative patching, so ours is a replication in MoE at expert level |
| **R3** retrieval vs computation | *Disentangling Recall and Reasoning through Layer-wise Attention and Activation Analysis* (arXiv 2510.03366) | substantial |
| **R2** context-grounded vs parametric | ReDeEP; LLM-Check | substantial |
| **P1** membership benchmarks are broken | Das et al., *Blind Baselines Beat MIA*; Duan et al. | known in advance — it is why the certified corpus exists |
| framework | ROME / MEMIT causal tracing | foundational |

The Expert-Aware paper independently reports the same architectural split we
saw: factual recovery "can either concentrate in a recurrent expert or remain
distributed across a routed expert coalition."

## What appears to be genuinely new here

1. **The `para` / `samerel` / `other` control triad.** Prior MoE work controls
   against *other experts selected for the same prompt* — within-prompt
   attribution, "which of the eight matters?". Contrasting a fact's experts
   against a **paraphrase of the same fact**, **same entity / different
   relation**, and **different entity / same relation** asks a different
   question: is the address fact-level, entity-level, or merely input-specific?
   No prior work found running it.
2. **R10's within-passage crossover on model-generated text.** Prior work uses
   probes throughout. Ablating fact i's experts and measuring damage at i versus
   j inside the same generated passage — which admits no text baseline — was not
   found elsewhere.
3. **The R8b argument** that provenance-as-classification is unwinnable when the
   prompt determines the label, so provenance must be causal. Possibly folklore,
   but not found stated.

## Consequence for the plan

**The editing step is the one the literature specifically predicts will fail.**
Hase et al., *Does Localization Inform Editing?* (arXiv 2301.04213), found the
correlation between causal-tracing localisation and edit success is near zero
across ROME, MEMIT and finetuning. Hochman et al. attribute this to exactly the
redundancy R9d measured. Proposing "use the address to edit" without engaging
that result would be proposing a known dead end.

---

# Part I — What holds

## R9 — routing carries a causal, fact-level address

→ `knowledge/provenance.py`, `records/provenance.*.json`

Every earlier attempt at provenance was a **classification** task, and
classification of a prompt-determined label is unwinnable in principle: the
prompt is an oracle for its own label, which is why bag-of-words beat routing in
four of five conditions (R8b). Provenance is a **causal** question — which
components, if removed, change *this* answer — and a causal test has no text
baseline, because bag-of-words cannot nominate experts to ablate.

Five conditions at **matched ablation size** (K=8 experts banned per layer,
every layer; gate scores set to −inf *before* top-k, so the router must route
around them). Only *which* K differs:

| condition | what it bans |
|---|---|
| `own` | the experts this fact actually routed to |
| `para` | same fact, **different wording** |
| `samerel` | **same entity, different relation** |
| `other` | different entity, same relation |
| `random` | K random experts per layer |

**qwen3.6-35b-a3b (instruct), ΔNLL on the fact's own answer:**

| domain | n | own | para | samerel | other | random | para>samerel |
|---|---|---|---|---|---|---|---|
| country | 135 | 3.703 | 3.859 | 2.426 | 1.492 | 0.022 | **0.793** |
| composer | 52 | 3.118 | 3.219 | 2.053 | 2.621 | 0.030 | **0.711** |
| author | 72 | 2.233 | 2.116 | 1.365 | 1.937 | 0.070 | **0.689** |
| element | 81 | 2.527 | 2.551 | 1.434 | 2.363 | −0.020 | **0.687** |
| **ALL** | 340 | **3.022** | **3.134** | 1.943 | 1.966 | 0.024 | **0.738** |

**The ordering is `own ≈ para > samerel ≈ other ≫ random`, and each step means
something distinct:**

- **`para ≈ own` (3.13 vs 3.02).** A different *wording* of the same fact damages
  it as much as its own wording. This kills the obvious confound — that "own"
  experts, being top-scoring *for this input*, are maximally disruptive to
  remove. `para` is a different input; had that been the mechanism it would have
  behaved like `other`. It does not.
- **`samerel ≈ other` (1.94 vs 1.97).** Changing *either* the entity *or* the
  relation costs the same. The address is specific to the **conjunction**, not
  to the entity alone and not to the question form alone. That is what makes it
  fact-level rather than entity-level.
- **`random ≈ 0` (0.024).** Banning 8 arbitrary experts per layer across all 40
  layers costs essentially nothing, while banning the 8 the model chose costs
  3.02. MoE computation is extraordinarily concentrated on its selected paths.

**Pre-registered bar, met:** own must exceed an equally-sized other-fact ablation
on a majority of facts in ≥3 of 4 domains. All four pass, and the sharper
fact-vs-entity test passes in all four too.

### R9b — replication on gemma-4-26b-a4b (base)

| domain | n | own | para | samerel | other | random | para>samerel | kept ok |
|---|---|---|---|---|---|---|---|---|
| element | 111 | 3.640 | 3.465 | 2.244 | 3.497 | 0.048 | **0.755** | 0.288 |
| country | 97 | 3.474 | 3.302 | 2.296 | 2.580 | 0.091 | **0.811** | 0.113 |
| composer | 27 | 3.678 | 4.822 | 3.067 | 3.623 | 0.258 | 0.889 | **0.000** |
| author | 12 | 2.815 | 2.875 | 3.249 | 2.530 | 0.122 | 0.500 | 0.083 |
| **ALL** | 247 | **3.539** | **3.460** | 2.314 | 3.104 | 0.092 | **0.783** | 0.178 |

The ordering replicates and para>samerel is 0.783 against qwen's 0.738. **But
the replication rests on two domains, not four**: composer has `kept ok` 0.000
and author has n=12, both uninterpretable by the criterion below. Element
(n=111) and country (n=97) carry it.

One real difference: on gemma `other` (3.104) costs more than `samerel` (2.314),
whereas on qwen they were equal. Changing the entity hurts more than changing
the relation there, so gemma's address looks more **entity-weighted**. The
fact-level test still passes; the two architectures are not identical in shape.

### R9c — the address lives in the late layers, and a lighter touch reads it better

The whole-stack ablation was heavy: ΔNLL ≈ 3 with only ~a third of the damage
fact-specific and the answer surviving under a third of the time. Restricting
the intervention to the **late layers** improves every axis at once.

qwen, K=8, per layer band (n=120 subset):

| layers | own | para | samerel | other | random | para>samerel | kept ok |
|---|---|---|---|---|---|---|---|
| 24–31 (8) | 0.046 | 0.050 | 0.037 | 0.041 | −0.005 | 0.558 | 0.983 |
| 20–35 (16) | 0.218 | 0.211 | 0.249 | 0.153 | −0.003 | 0.542 | 0.858 |
| 36–39 (4) | 1.752 | 1.794 | 1.257 | 0.512 | 0.042 | 0.683 | 0.425 |
| **28–39 (12)** | 2.788 | 2.753 | 1.704 | 0.753 | 0.037 | **0.708** | 0.342 |
| all (40) | 3.835 | 3.981 | 2.539 | 1.573 | 0.029 | 0.775 | 0.283 |

**Layers 24–31 carry almost nothing (own 0.046) while 36–39 — four layers, 10%
of the stack — carry half the total damage.** This is not "late layers matter
more" in general: random ablation is flat across bands (0.03–0.04), so it is the
fact-specific routing that is concentrated late. Note these are *not* the layers
R2's grounding signal occupied (24–31), so grounding and fact-identity live in
different places.

**Full run at layers 28–39 beats the whole-stack run on every axis:**

| domain | n | own | para | samerel | other | random | para>samerel | kept ok |
|---|---|---|---|---|---|---|---|---|
| country | 135 | 2.664 | 2.635 | 1.648 | 0.719 | 0.042 | 0.704 | 0.333 |
| element | 81 | 2.272 | 2.260 | 1.031 | 2.221 | 0.044 | 0.791 | 0.309 |
| composer | 52 | 1.955 | 1.861 | 0.937 | 1.738 | −0.020 | **0.895** | 0.327 |
| author | 72 | 1.529 | 1.453 | 0.595 | 1.124 | 0.006 | **0.852** | 0.403 |
| **ALL** | 340 | 2.222 | 2.212 | 1.207 | 1.319 | 0.025 | **0.777** | **0.341** |

| | whole stack | layers 28–39 |
|---|---|---|
| para>samerel | 0.738 | **0.777** |
| kept ok | 0.294 | **0.341** |
| fact-specific share (own−samerel)/own | 34% | **46%** |
| layers touched | 40 | 12 |

**This also resolves the thin-domain problem.** Composer and author were the two
domains R9b could not read (`kept ok` 0.192 and 0.333 on qwen, 0.000 and 0.083
on gemma). Under the lighter intervention they are the *strongest* domains —
para>samerel 0.895 and 0.852 with survival 0.327 and 0.403. The earlier weakness
was the intervention destroying the model, not an absence of address.

**The knob transfers**, checked on gemma before building on it:

| model | band | para>samerel | kept ok |
|---|---|---|---|
| qwen | all 40 | 0.738 | 0.294 |
| qwen | **28–39** (30%) | **0.777** | **0.341** |
| gemma | all 30 | 0.783 | 0.178 |
| gemma | **18–29** (40%) | **0.813** | **0.368** |

Late-layer restriction improves selectivity *and* survival on both
architectures. Use it as the default; the whole-stack numbers above are kept
because they were the pre-registered configuration.

### R9d — per-layer scan: the address is distributed and redundant, not localised

R9c swept *bands*, which cannot distinguish "twelve contributing layers" from
"one layer plus eleven passengers". Ablating **one layer at a time**, K=8, same
60-probe subset as the band runs:

| | joint ablation (own) | sum of the same layers ablated singly | ratio |
|---|---|---|---|
| layers 36–39 (4) | **1.130** | 0.366 | **3.1×** |
| layers 28–39 (12) | **2.303** | 0.652 | **3.5×** |
| all 40 | **3.398** | 1.174 | **2.9×** |

**Joint ablation does roughly three times what the individual ablations sum to.**
No single layer carries the address: the best is L36 at own 0.261 against 1.130
for the four-layer band containing it, and only three layers of forty exceed
own 0.10 (L21, L35, L36). Single-layer `kept ok` is 0.87–0.98 — one layer's
experts can be removed and the answer barely notices.

**So the address is a distributed, redundant code across the late layers.**
Remove one layer's experts and the remaining layers compensate; remove several
together and the compensation fails. This is W5.1b's redundancy finding one
level up — there, dropping one expert of 128 cost ≤8% because the router still
had seven; here, dropping one layer costs almost nothing because 39 others still
carry the fact.

**This corrects a natural misreading of R9c.** "Concentrated in the late layers"
is true at the level of *bands* — late bands work, early bands do not — but it
must not be read as "localised to a few layers you could point at". You cannot
name the layer a fact lives in. You can only say it lives in the late third of
the stack, spread across it, with no single layer necessary.

Incidental: L21 is the strongest single layer outside the late band (own 0.117).
It is also W5.3's top ablation-ranked knowledge layer and R3's best single layer
for the retrieval-vs-computation meter. Three instruments agreeing on one layer
is suggestive, but at this magnitude it is not evidence.

### R9g — the overlap confound, checked at last (and R9 survives it)

B1 proposed ablating the **consensus** expert set across a fact's paraphrases,
on the theory that a single rendering mixes fact-specific with input-specific
routing. Measuring the consensus first killed the experiment and produced
something better.

**Paraphrases route almost identically.** Strict intersection across three
paraphrases retains **7.07 of 8** experts per layer on the injected corpus
(7.95 at 2-of-3), and 6.3/8 on grid2. A consensus ablation is therefore not a
distinct intervention — it is the single-paraphrase set minus one expert.

That raised a much sharper worry: **if `para` shares ~78% of `own`'s experts,
is `para ≈ own` simply a near-tautology?** This is the confound that should have
been checked when R9 was designed. Measured on grid2, layers 28-39, K=8:

| condition | overlap with the `own` set | R9 damage |
|---|---|---|
| own | 1.000 | 2.222 |
| **para** | **0.779** | **2.212** |
| **other** | **0.731** | **1.319** |
| samerel | 0.559 | 1.207 |
| random | 0.030 | 0.025 |

**`para` and `other` sit at nearly the same overlap (0.78 vs 0.73) yet differ in
damage by 68% (2.212 vs 1.319).** The data contains a matched-overlap natural
experiment, and at matched overlap, ablating the *same fact* through a different
wording is far more damaging than ablating a *different fact*. Damage is
therefore not a function of set overlap, and R9's ordering is not an artifact of
paraphrases happening to route alike.

Note also that `para` has the highest damage-per-unit-overlap of any condition
(2.84 vs own's 2.22): the shared experts it removes are disproportionately the
ones that matter.

**Caveat.** This is a group-level argument from near-matched overlap, not a
per-item regression of damage on overlap with condition as a covariate. The
latter would be stronger and needs per-item overlap paired with per-item damage,
which the R9 records do not currently store together.

**B1 is withdrawn** — the consensus set is 88% identical to the single-paraphrase
set, so it cannot separate the hypotheses it was designed to separate. The
overlap check it prompted is worth more than the experiment would have been.

### R9f — ground-truth validation by injection (Option B)

R9 infers an address from differential damage, with no ground truth about what
the address should be — the gap *Do Localization Methods Actually Localize
Memorized Data in LLMs?* (arXiv 2311.09060) identifies. This is the injection
form of that test: teach the model facts it **provably could not know**, then
run R9's protocol on them.

250 entities invented from a syllable grammar, 8,250 Q->A pairs (3 phrasings per
fact), LoRA ~2 epochs at lr 5e-5. Recall on the trained set against a matched
never-trained set from the same generator:

| | recall |
|---|---|
| **injected facts** | **0.417** |
| never-trained facts (same generator, same format) | **0.011** |

Ablation on the injected facts, K=8, layers 28-39:

| own | para | samerel | other | random | para>samerel |
|---|---|---|---|---|---|
| 0.624 | 0.612 | **0.430** | 0.579 | **0.040** | **0.725** |

**`own ≈ para > samerel`, and `random ≈ 0`** — the same structure R9 found on
pretrained facts, at a comparable rate (0.725 here vs 0.777 there). **So the
fact-address pattern is not an artifact of how pretraining organizes
knowledge**: it appears for a fact installed by LoRA in two epochs, on an entity
that did not exist a day earlier.

**Two honest qualifications.**

`other` (0.558) sits much closer to `own` (0.624) than in R9 (1.319 vs 2.222),
so *entity* discrimination is weak here. **Three designs have now failed to
explain it, each for a different structural reason, and that is the finding.**

| test | result | why it could not work |
|---|---|---|
| split by `field` | same 0.582 / cross 0.558 | the field **never appears in the question**, so the split varied something the model could not condition on |
| A1 name-similarity gradient | Spearman **+0.006** | 408 of 450 substitute pairs share **exactly zero** character trigrams — the variable has no variance, so there is no gradient to correlate against |
| A2 org-type split | same 0.690 / diff 0.981, n=65 | direction **opposite** to prediction and underpowered; unreadable as anything but noise |

**The corpus is structurally unsuited to this question, and that is a
consequence of its original design.** It was built for membership inference,
where maximal homogeneity between members and non-members is exactly what keeps
the blind baseline at chance. That same homogeneity makes the entities mutually
orthogonal — no two are *similar* — so "does routing separate similar entities
less well than dissimilar ones?" has no items to measure on. Homogeneity was a
design goal for the null and is fatal here.

**What would work (A3, not run):** inject entities whose names carry meaning
present in the prompt — `the Glacier Survey of Vantholm` vs `the Foundry Guild
of Bracken` — deliberately constructing graded semantic similarity, then split
`other` by distance between head nouns. Until then, **weak entity discrimination
on injected facts is unexplained**, and the leading hypothesis (coined entities
lack the semantic representation that France or iron carry) remains untested.

**The null, run properly — and it sharpens the claim.** Requiring correct
answers left ~12 non-member items, too few to ablate. Scoring damage against the
model's OWN output instead makes the null runnable at full size: a non-injected
fact still produces a confabulation, and asking whether *that* has an address is
the better question anyway.

| group | n | own | para | samerel | other | random | para>samerel |
|---|---|---|---|---|---|---|---|
| **injected, recalled correctly** | 450 | **1.073** | 1.057 | 0.796 | 0.939 | 0.027 | **0.673** |
| injected, answered wrong | 630 | 0.553 | 0.560 | 0.457 | 0.461 | 0.015 | 0.563 |
| never injected | 1080 | 0.542 | 0.532 | 0.389 | 0.479 | 0.036 | 0.581 |

**Structure tracks RETRIEVAL SUCCESS, not injection.** Facts the model actually
recalls show the ordering at 0.673 with roughly double the damage magnitude.
Facts it was taught but gets *wrong* (0.563) are indistinguishable from facts it
was never taught (0.581) — both are confabulations, both show only weak
structure.

So the address is a property of **successfully retrieving a stored fact**, not of
the question, the entity, or mere presence in the training data. That is a
sharper claim than R9 could make, and the confabulation groups are the control
that establishes it.

**Caveat on magnitude.** The separation is real but modest — 0.673 against
0.563/0.581. And self-answer scoring makes `own` damage partly tautological (the
experts that produced output X are by construction those whose removal damages
X), so `para > samerel` is the non-circular statistic and the one quoted here.

### What it cost: four bugs, three of them recurrences

1. **Document training gives recitation, not recall.** The Phase 1 adapter
   completes a document's own wording 23.5% of the time (vs 0% for never-seen
   docs, so injection worked) but answers questions **0.000**. Reciting is not
   recalling; the QA-format retrain is what made facts queryable.
2. **Probed in the wrong format.** Chat-wrapping a fact installed as
   `Q:/A:` cost a third of the recall (0.408 -> 0.275). Same class as W5's
   base-vs-instruct trap: probe a fact in the form it was learned.
3. **The leading-space bug, third appearance.** `.strip()` on an answer whose
   training text is `"A: {val}"` scores every probe against a token the model
   never emits — 0.000 for members *and* non-members.
4. **A format artifact masquerading as recall.** Every `journal` value is
   "the X Review", so first-token matching scored **1.000 for members and
   non-members alike**, inflating the null from 0.011 to 0.210. Dropped.

Recurring lesson: in this project, **two conditions returning identical numbers
has been a reliable bug signal, never a finding.**

### R9e — ablation-mechanism robustness: partial, with an honest gap

*Transformer Circuit Faithfulness Metrics Are Not Robust* (arXiv 2407.08734)
reports that circuit conclusions flip on "seemingly insignificant changes in
the ablation methodology". R9 rests on **one** intervention, and this project
had already seen the sensitivity first-hand — gemma's zero-weight ablation gave
dNLL 17 where route-around gave 3.5. So the mechanism was made a parameter.

| model | mechanism | own | para | samerel | other | random | para>samerel |
|---|---|---|---|---|---|---|---|
| qwen | route-around | 2.222 | 2.212 | 1.207 | 1.319 | 0.025 | **0.777** |
| qwen | resample | 2.028 | 1.997 | 1.178 | 1.329 | 0.040 | **0.738** |
| gemma-base | route-around | 2.321 | 2.414 | 1.496 | 1.822 | 0.043 | **0.813** |
| gemma-base | zero-weight K=2 | −0.001 | −0.004 | 0.017 | 0.002 | 0.005 | 0.458 |
| gemma-base | zero-weight K=4 | 0.031 | −0.004 | 0.103 | 0.030 | 0.023 | 0.409 |

**The ordering `own ≈ para > samerel ≈ other >> random` survives every
intervention that actually bites** — route-around on two architectures, and
stochastic resampling (the causal-scrubbing form, where a banned expert is
given a score drawn from the layer's observed distribution and therefore
sometimes survives selection).

**Zero-weight has no readable window, so it neither supports nor refutes.** At
K<=4 the model barely notices (own 0.031 against random 0.023, survival 96-98%)
— W5.1b's redundancy at work: drop 4 of 8 contributions and the remaining 4
carry it. At K=8 it silences the whole routed branch, which R9b showed destroys
the model. Differential damage cannot be measured when total damage sits at
noise level. **This is a genuine gap, not a passed test.**

**`mean` is not a distinct mechanism, and reporting it as one would have been
inflation.** Replacing a banned expert's score with the full-vector mean
(−5.92, sd 1.01) puts it far below the top-8 cutoff, so it deselects exactly as
−inf does and reproduces `route` to three decimals. For a score-returning
router, *any* sub-threshold replacement is the same intervention. The honest
mechanism count is three, not four.

### Three harness bugs this exposed, all of the same family

Each would have produced a confident, meaningless robustness claim:

1. **Silent fallback.** `zero` on a score-returning router fell through to
   route-around and reported route's numbers under a different label — a
   fabricated second mechanism. Now raises.
2. **Measurement after an early return.** The baseline pass runs with no bans
   set, and the full-vector measurement sat below `if ban is None: return`, so
   it never executed — leaving `mean`/`resample` with no replacement value and
   silently falling back to route. Now measured before the return, with a guard
   that refuses to proceed without a replacement.
3. **Wrong baseline population.** The first replacement value came from the
   top-32 scores in the trace, whose mean sits far above the full distribution,
   so banned experts stayed competitive and often survived selection — `mean`
   ablated almost nothing (own 0.10 vs route's 1.07).

### Why K=8, and why `kept ok` gates everything

`kept ok` is how often the answer survives the ablation at all. If it is ~0 the
model is destroyed rather than selectively damaged, and the ΔNLL ordering is
noise. The K sweep on qwen:

| K | ΔNLL own | ΔNLL random | own>other | kept ok | readable? |
|---|---|---|---|---|---|
| **8** | 3.02 | 0.07 | 0.632 | 0.19–0.36 | **yes** |
| 16 | 4.71 | 0.11 | 0.656 | 0.015–0.086 | no |
| 32 | 6.98 | 0.18 | 0.582 | 0.000–0.012 | no |

Separation *persists* at K=16 and K=32 while the model answers correctly ~1.5%
and ~0.3% of the time. **Fixing K=16 without the sweep would have reported a
passing result on a broken model.**

### Two errors it took to get the gemma replication

The first gemma run was garbage — random ablation at ΔNLL **17.0** against
qwen's 0.02, `kept ok` 0.000, every condition destroyed. The random control is
what caught it: a random-ablation ΔNLL of 17 is impossible if the intervention
is selective, so it flagged a broken *setup* rather than a broken hypothesis.

1. **Wrong seam.** gemma's `Router` runs top-k internally and returns
   `(indices, weights)`, so wrapping it allowed only post-selection
   weight-zeroing — and with top_k=8, banning K=8 silences *every* selected
   expert, i.e. the entire routed branch. That is W5.0's whole-branch ablation,
   not a fact ablation. `Router.proj` is the Linear producing raw scores;
   wrapping *that* restores the same route-around mechanism qwen uses, and
   random ablation immediately fell from 17.0 to 0.066.
2. **Wrong checkpoint.** Even fixed, gemma-**instruct** has no measurement
   window — flat noise to K=4, catastrophe at K=8 — because it is saturated
   (entropy ≈0.001, the condition W5.2 flagged). Its answers sit at probability
   ~1.0, so NLL is unmoved until the branch breaks. W5's own prescription is to
   use the **base** checkpoint for factual recall, and on base the window
   reappears. qwen needed neither correction: it has no base checkpoint and its
   instruct tune is *not* saturated (baseline NLL 0.047, W5.3). That asymmetry
   was in the handoff and should have been applied before the first gemma run.

### Honest limits

- **The intervention is heavy.** ΔNLL ≈ 3 and the answer survives about a third
  of the time. Valid because all conditions are size-matched, but not a light
  touch.
- **The fact-specific component is a minority of the damage.** own (3.02) minus
  samerel (1.94) ≈ 1.1 of 3.0 — roughly a third is fact-specific; the rest is
  shared across any routed-expert ablation.
- Only facts the model answers correctly are included.

---

## R10 — provenance over the model's own generated text

→ `knowledge/generated.py`, `records/generated.json`, `records/generated.open.json`

R9's facts were all put in front of the model by a probe harness. The
application is the reverse: the model writes a paragraph and you ask which
stored facts it drew on. R8b established that this framing is *required* — when
the prompt names the fact, bag-of-words is an oracle and routing cannot win.

**The design is a within-passage crossover, which removes the need for any
baseline.** For two fact-bearing positions i and j in the same generated passage:

```
ban(experts routed at i)  ->  measure NLL at i and at j
ban(experts routed at j)  ->  measure NLL at i and at j
selectivity = (dNLL_ii - dNLL_ij) + (dNLL_jj - dNLL_ji)   > 0
```

Same passage, same tokens, same sequence positions, same number of experts
banned, same forward pass. The only thing that varies is *whose* experts were
removed. A confound would have to explain why banning position i's experts
specifically damages position i, in text the model wrote itself.

Target positions are located by matching the generation against the grid2
answer table, so there is ground truth without hand-annotation and without the
prompt ever stating the answer.

| condition | pairs | own | other | random | selectivity | frac own>other |
|---|---|---|---|---|---|---|
| **directed** (prompt names entity + relation) | 12 | 3.430 | 0.211 | 0.014 | **+3.219** | **0.875** |
| **open-ended** (model picks the facts) | 13 | 6.492 | 4.229 | −0.000 | **+2.263** | **0.846** |
| — same domain | 10 | | | | +2.014 | 0.800 |
| — cross domain | 3 | | | | +3.094 | 1.000 |

**R9's fact-level address survives into free generation.** In the directed
condition the separation is stark: ablating a fact's own experts costs 3.43
while ablating a neighboring fact's experts in the same sentence costs 0.21.

**The open-ended condition is the real test and it holds**, with two honest
differences. `other` damage is far higher (4.23 vs 0.21) because open prose is
more entangled — facts share context, so removing any fact's experts degrades
the whole passage. Selectivity is correspondingly lower but still clear, and
cross-domain pairs separate better (+3.09, 1.000) than same-domain (+2.01,
0.800), which is the expected direction if the address is fact-specific.

**Random ablation is ~0 throughout.** At K=64 — a *quarter* of all experts,
every layer — random ablation costs 0.0005 while the 8 the model chose cost
3.75. MoE computation is extraordinarily concentrated on its selected paths.

### Limits

- **Small.** 12 and 13 pairs; 24 and 26 directed tests.
- **Confident tokens.** Most directed targets have base NLL ≈ 0.0001, so damage
  is measured from a floor. Open-ended targets are more varied (0.004–0.96).
- **qwen only.** The gemma path is implemented (`Router.proj` seam) but not run.
- **Targets are facts the model got right**, located by string match against a
  fixed table — this does not find facts outside grid2.

### Two matching faults, both mine

1. **Accents and multi-token answers.** Prefix-matching decoded tokens lost 4 of
   12 tasks: the model writes `Brasília` and `złoty` where the table says
   `Brasilia`/`zloty`, and `Cu`/`Hg` tokenise as two pieces. Fixed by
   accent-folding and mapping a *character* offset to a token index.
2. **Substring matches with no fact behind them.** The open scan matched `sol`
   (Peru's currency) inside a word in a passage about Chopin. Fixed with word
   boundaries. Without it the crossover would have run on a target that was not
   a fact at all.

---

## R11 — prefetch: the signal is real and far too small

→ `knowledge/prefetch.py`, `records/prefetch.qwen.json`

The last path to the original run-large-models-locally goal. W4's offload
runtime pays for **miss rate**, and OPT beats LRU by +0.173 hit rate at 2 GB
because it sees the future. R9c suggests a cheap proxy: fact-specific routing is
concentrated in the *late* layers, so by the time a forward pass finishes the
early layers the fact may be determined — and a correct prediction there buys
the whole early-layer compute time as a fetch window. That is *within-token*
prefetch, a much easier target than predicting the next token.

Predicting the top-8 in layers 28–39 from the experts selected in layers 0–27,
fitted on half the tokens and evaluated on the other half:

| policy | recall@8 | |
|---|---|---|
| frequency (static resident set) | 0.1221 | |
| **previous token, same layer** | **0.4177** | what LRU exploits, free |
| previous token, same domain | 0.4177 | |
| early-layer co-occurrence | 0.1803 | **loses to recency by 0.24** |
| previous + random fill (2k) | 0.4366 | control |
| previous + popularity fill (2k) | 0.4747 | control |
| previous + early-layer fill (2k) | **0.5045** | **+0.0298 over best control** |

**Alone the signal is far worse than recency** — 0.18 against 0.42. Recency is
extremely strong here because consecutive tokens in a generation route alike,
which is exactly the locality LRU already captures for nothing.

**On top of recency it adds a real but small amount.** At double budget,
early-layer fill beats popularity fill by +0.030 and random fill by +0.068, so
the lift is signal rather than budget diversity — the random-fill control rules
that out.

**Verdict: not a lever.** Three reasons, and the third is decisive:

1. +0.030 is about 17% of the whole OPT−LRU headroom (+0.173 at 2 GB), and that
   comparison flatters it — recall@2k is not the same metric as cache hit rate.
2. It requires **doubling the prefetch budget**, which costs the memory the
   exercise is trying to save.
3. Recency at 0.4177 is already most of what is available, and it is free.

This closes the compression line. The fact-level address is real (R9/R9c) but it
does not convert into a runtime win, which is consistent with everything else
here: routing is informative about *what the model did* and uninformative as a
*predictor* competing against a cheaper signal.

---

## R12 — read-only fact identification: ties with reading the text

→ `knowledge/readout.py`, `records/readout.json`

**The experiment the product depended on.** R9/R10 establish the address by
*ablation* — N destructive forward passes per fact, impossible in production.
Telemetry must be read-only. R8 showed read-only fact identification losing to
bag-of-words, but that was in the probe setting where the prompt named the
entity and relation. The stated hypothesis was that in generated prose nothing
names the fact, so the baseline that beat us there would not exist.

90 generated passages, 158 fact tokens, 25 facts, **leave-one-passage-out**:

| baseline | accuracy | |
|---|---|---|
| chance | 0.0400 | |
| prior | 0.1600 | label imbalance |
| position | 0.1467 | "facts appear in a predictable order" |
| **bag-of-words on the preceding text** | **0.6067** | |
| **routing (read-only, one pass)** | **0.6000** | **lift −0.007** |

**Routing reads the fact at 15x chance and ties with reading the text.**

**The hypothesis was wrong, in the way the module docstring flagged to watch
for.** Generated prose *does* name its own facts — the model writes "The capital
of France is Paris", which contains both the entity and the relation. R8's
objection followed us into the generated-text setting.

Per domain: country routing 0.530 vs words 0.590; element routing 0.708 vs words
0.646. A wash, in different directions.

### The secondary result, which is real but is not a rescue

Routing and text are **semi-independent views**, and their agreement predicts
correctness sharply:

| | n | correct |
|---|---|---|
| routing and words **agree** | 78 | **0.936** |
| routing and words **differ** | 72 | routing 0.236, words 0.250 |

Combining them (0.75 routing + 0.25 words, normalized log-scores) gives 0.6467
against 0.6067 for words alone. So routing does carry information the text does
not — but as a *second opinion*, not a standalone signal. Caveat: ensemble
disagreement predicting error is a well-known effect, and whether routing is a
better second view than another text model is untested.

### Verdict for the product

**Routing provenance is an offline forensic tool, not telemetry.** The causal
version (R9/R10) is strong and needs ablation; the read-only version ties with
reading the output. A product that needs to know which facts an answer used can
read the text more cheaply than it can read the router.

What survives for a serving stack: **routing-vs-text agreement as a confidence
signal** — spans where the two views disagree are ~25% correct against ~94%
where they agree. That is a smaller claim than "provenance telemetry", and it
should be scoped as such.

### The methodological catch

A first run scored `position` at **1.0000**, which is impossible for a real
signal. Cause: decoding is greedy, so repeating one prompt three times per
entity produced **byte-identical** passages, and the held-out-passage split was
holding out a copy of a training item. Every baseline was scoring by
twin-matching. Fixed with four distinct prompt phrasings per entity, plus a
guard that fails loudly if >20% of facts have identical prefixes across
occurrences (now 0/33). The `position` baseline existed to catch "facts appear
in predictable order"; it caught a worse problem instead.

---

## R17 (Stage C) — entropy is architecture-independent

→ `records/stage_a.json`

Five models, two families, both architectures, same two tasks. `p90` of
per-token entropy, abstentions and truncations excluded:

| model | arch | popqa `p90` | omniscience `p90` | popqa abstain |
|---|---|---|---|---|
| qwen36-35b-a3b | **MoE** | **0.922** | **0.764** | 0.0% |
| qwen36-27b | dense | **0.917** | **0.735** | 0.0% |
| gemma-4-26b-a4b | **MoE** | 0.889 | 0.734 | 6.8% |
| gemma-4-31b | dense | 0.849 | 0.706 | 16.4% |
| gemma-4-e4b | dense | 0.884 | 0.763 | 11.2% |
| **spread** | | **0.073** | **0.058** | |

**The matched pair is the clean test** — qwen MoE vs qwen dense hold family,
training and 4-bit quantization constant and vary only the architecture.
Entropy moves **0.005** on popqa and **0.029** on omniscience, well inside the
bootstrap intervals from Stage A. The generation-length baseline tracks equally
closely (0.600 vs 0.590), which is the sanity check that this is genuine
invariance rather than two unrelated numbers landing near each other.

Across all five models the spread is 0.073 (popqa) and 0.058 (omniscience) —
smaller than the within-task CI width. **Entropy needs no router seam and works
on dense models**, which is precisely what routing cannot do, and it is why the
online half of the product is not MoE-specific.

### Abstention varies enormously by model, and that is a product finding

| model | omniscience abstain |
|---|---|
| gemma-4-e4b | **65.0%** |
| qwen36-27b dense | 31.0% |
| qwen36-35b-a3b | 21.3% |
| gemma-4-31b | 17.0% |
| gemma-4-26b-a4b | 15.0% |

The smallest model abstains on two thirds of AA-Omniscience — it *knows* it does
not know, which is the behavior that benchmark rewards. Any deployment that
scores abstention as an error (as an earlier suite here did) would rank e4b
catastrophically and wrongly. Abstention rate is a model property, it varies by
4x across this set, and it must be measured per model rather than assumed.

## R16 (Stage B) — entropy vs self-consistency: competitive at 1/4 the cost, 2 of 3

→ `knowledge/stage_b.py`, `records/stage_b.json`

The comparison that decides whether this is a product or a redundancy.
Self-consistency — sample k times, measure agreement — is what production
systems actually use. k=5 at temperature 0.7, vote signals from the vendored calibration suite's
`vote_signals` so the numbers stay comparable to its n=740 study. Wall-clock is
measured per arm, not assumed.

| task | errors | **p90 entropy** (1x) | best vote signal | gap | combined | measured cost |
|---|---|---|---|---|---|---|
| popqa | 142 | **0.923** | 0.896 (share) | **+0.027** | 0.930 | 4.0x |
| mmlu_pro | 17 | 0.780 | 0.808 (share) | −0.028 | **0.863** | 4.8x |
| omniscience | 131 | 0.733 | 0.802 (share) | **−0.069** | 0.805 | 3.7x |

**Pre-registered bar — within 0.05 AUC of the best vote signal at 1/k cost —
passes on 2 of 3.** On PopQA entropy *beats* 5-vote self-consistency outright
while costing a quarter as much. On MMLU-Pro it is within tolerance. On
AA-Omniscience it loses by 0.069.

**Where it loses, and why that is coherent.** AA-Omniscience runs at 14%
accuracy against PopQA's 25%. At that difficulty the model's samples disagree
wildly, which is highly informative for a vote, while entropy appears to
saturate. So the failure is on the *hardest* task, which is also where a
practitioner would most willingly pay 4x.

**Combination is task-dependent, and an earlier partial read here was wrong.**
On two tasks it adds almost nothing (+0.007 popqa, +0.003 omniscience) — but on
MMLU-Pro it adds **+0.055** over the best single signal (0.863 vs 0.808). So the
two signals are largely redundant on short-answer recall and genuinely
complementary on multiple choice. The MMLU-Pro cell rests on 17 errors and
should be held loosely.

**Product reading.** Entropy is the right default: one forward pass, no sampling,
competitive-or-better on most tasks. Self-consistency is the right escalation
for hard or high-stakes items — which fits the two-mode architecture rather than
undermining it, with entropy triaging and something more expensive spent only
where it is warranted.

## R15 (Stage A) — entropy generalizes across task types, but not as R13 measured it

→ `knowledge/stage_a.py`, `records/stage_a.json`

R13's 0.892 was one benchmark and one measurement position. Five task
configurations on qwen3.6-35b-a3b, generous token caps, abstentions and
truncated generations excluded, 95% CIs from 400 bootstrap resamples:

| task | n | errors | `first` (R13) | `mean` | **`p90`** | `len` (dumb) |
|---|---|---|---|---|---|---|
| popqa | 250 | 187 | 0.915 | 0.894 | **0.922** | 0.600 |
| gsm8k (cap960) | 197 | 10 | **0.444** | 0.905 | **0.895** | 0.684 |
| gpqa (cap8192) | 188 | 34 | 0.712 | 0.780 | **0.777** | 0.688 |
| omniscience | 236 | 203 | 0.695 | 0.720 | **0.764** | 0.537 |
| mmlu_pro (cap4160) | 744 | 119 | 0.624 | 0.754 | **0.754** | 0.594 |

**`p90` of per-token entropy clears the pre-registered 0.75 bar on 5/5 task
types** and beats the generation-length baseline everywhere by +0.09 to +0.32.

**R13's measurement position was a lucky fit to its task.** `first` (entropy at
the last prompt token) scores 0.915 on PopQA — the task R13 used — and **0.444
on GSM8K**, worse than chance. The headline this project has been quoting since
R13 was specific to short-answer recall, not a property of entropy. The signal
that generalizes is an **aggregate over the generation**, and one measure now
covers recall and reasoning without per-task configuration.

**Honest on the CIs.** Lower 95% bounds for `p90` are 0.88 (popqa), 0.77
(gsm8k), 0.71 (mmlu_pro), 0.70 (gpqa), 0.68 (omniscience) — so two of five dip
below 0.75 at 95% confidence. The point estimate clears the bar 5/5; the
interval does not. GSM8K's 0.895 rests on **10 errors** and its CI spans
0.77-0.97, so it should not be quoted alone.

### The token cap is a first-order confound, quantified

GPQA run at both caps, same 198 items:

| | truncated | usable | errors | error rate among survivors |
|---|---|---|---|---|
| cap4160 | **24.2%** | 146 | 16 | 11.0% |
| cap8192 | **2.5%** | 188 | **34** | 18.1% |

Doubling the cap more than doubled the usable errors. And the error rate among
*surviving* items rose 11% -> 18%, which measures the selection bias directly:
excluding truncated items at the tight cap was systematically removing the
harder questions, leaving an easier set. Any error-prediction result computed at
a tight cap is measuring truncation, not error.

This is the same effect recorded across the earlier suite (`mmlu_pro/qwen`
0.345 -> 0.820 at cap4160; `gsm8k/qwen` 0.86 -> 0.935 at cap960) and that R14
found on GSM8K, where 73% of "errors" were truncation artifacts.

## R13 — error prediction on PopQA: entropy wins, routing is not needed

→ `knowledge/popqa.py`, `records/popqa.json`

The triage question answered on a real error corpus. **400 PopQA items, 300
genuine model errors (75%)**, ground truth from the benchmark's curated alias
sets — not from a table written here. Dataset, prompt builder (`think=False`)
and scorer come from the vendored `_vendor/suite.py` unchanged, so the accuracy is
comparable to the run already recorded there: **0.250 here against 0.29
recorded**, on a different 400-item sample. That agreement is the check that
the harness is sound.

| signal | AUC vs error | cost |
|---|---|---|
| **entropy** | **0.8924** | one scalar, free, works on dense models |
| top1_prob | 0.8840 | one scalar, free |
| routing | 0.7805 | 40x8 ints/token, needs the router seam |
| routing + entropy | 0.8765 | *worse than entropy alone* |

**Entropy beats routing by 0.11 AUC, and combining them makes it worse** —
routing does not add an independent component, it dilutes.

**The prediction that motivated this experiment was wrong.** The argument was
that entropy measures uncertainty while the dangerous failure is the confident
error, so entropy should be blind in the confident half and routing should earn
its place there. It does not:

| | n | errors | entropy | routing |
|---|---|---|---|---|
| confident (entropy <= median) | 200 | 106 | **0.753** | 0.643 |
| uncertain | 200 | 194 | **0.885** | 0.760 |

Entropy still separates errors within the confident band. The confident-and-wrong
population exists, and entropy finds it anyway.

### Why the earlier attempts could not have shown this

Two prior captures failed to answer it, both instructively:

1. **Hand-authored questions produced zero errors.** qwen answered all 44
   factual questions correctly and **refused all 16 invented entities** rather
   than fabricating. Both apparent errors were bugs in the scoring here — a
   Polish `ł` destroyed by an ASCII filter, and a refusal phrase missing from a
   pattern list. Net: the measurement produced 100% of the apparent errors.
   Worth keeping as a finding in its own right — **R6's fictional-entity
   fabrication was an artefact of forced completion**; in a chat setting a
   modern instruct tune refuses.
2. **The R12 agreement result was circular.** "Error" meant the routing reader
   was wrong and "disagreement" meant routing differed from a text reader, so
   disagreement mechanically implied error, and the corpus contained no model
   mistakes at all.

Only a curated benchmark with a real error rate removes the author from the
ground-truth path.

### Verdict for the product

**Use entropy for online triage.** It is better (0.892 vs 0.781), free, needs no
router seam, and works on dense models — which makes the online half of the
product architecture-agnostic and addresses a far larger market than MoE alone.

**Routing keeps the offline half.** R9/R10 remain the only thing here that can
say *which stored fact* a claim causally depends on, and no scalar does that.

This is the seventh applied framing in which routing loses to a simpler signal
(entity familiarity, fabrication, membership, contamination, prefetch, read-only
provenance, error prediction). The pattern is entirely consistent: routing is
informative about **what the model did** and is not competitive as a
**predictor**.

---

## R3 — routing separates retrieval from computation

→ `knowledge/probes.py` (`build_computation`), `records/meter.computation.*.json`

Both classes emit the **same answer token**, from an **identical context**, with
the **same two numbers** present. Only the question differs:

```
shared      Note the values 3 and 1645.
retrieved   ... The Treaty of Westphalia was signed in the year   -> 1648
computed    ... Taking those two values together, their sum is    -> 1648
```

One answer must come from the weights; the other is arithmetic over operands
sitting in the prompt. 38 numeric facts, 3 paraphrases each, 228 probes.
Leave-one-fact-out, 200 label permutations.

| condition | n | bal. acc | null p95 | length-only | digits-only |
|---|---|---|---|---|---|
| **retrieved vs computed** | 226 | **0.982** | 0.561 | 0.616 | 0.504 |
| — year facts only | 168 | **1.000** | 0.583 | 0.655 | 0.506 |
| — count facts only | 58 | **1.000** | 0.525 | 0.696 | 0.500 |

p = 0.005 (the floor at 200 draws). Both known cues are dead: digit count at
chance, length reaching only 0.616 against routing's 0.982.

**Replicates exactly on gemma-4-26b-a4b:** 0.982 overall, 1.000 on year facts,
0.964 on count facts, with the same cues controlled (R7).

### The free-text meter

→ `knowledge/annotate.py`

R3 lives in a probe harness; `annotate.py` takes it to running text. A
naive-Bayes profile fitted on the R3 trace scores every token position.

**Survives the format shift.** Re-scoring the probe answers as *raw
continuations* rather than chat turns, **leave-one-fact-out** so the profile
cannot be recognizing a fact it trained on: **AUC 1.000**.

**Absolute thresholds do not transfer.** On free text the profile marked 39 of
41 tokens as retrieval and zero as computation — the highest "retrieval" scores
went to whitespace (178), ` in` (176), ` of` (157). The profile was fitted only
on *answer* positions, so determiners and punctuation are out of distribution
and a binary trained on two answer types has no "neither" class.

**Within-document ranking does transfer.** On a passage alternating facts and
arithmetic, the four lowest-ranked tokens out of 28 were exactly the four
arithmetic answer tokens; the two factual answers sat at ranks 25 and 27:

```
capital of NorwayR isR OsloR .  Nine times seven isc sixtyc -threec .
capitalR ofR PeruR isR LimaR .  Fif teen plusc eightc isc twentyc -threec .
```

So the meter is a **relative** instrument: it ranks tokens within a passage and
cannot say "this whole passage is retrieval".

### R4 — cross-suite transfer

| trained on | tested on | result |
|---|---|---|
| computation | mechanism | `recall`→retrieved 1.000, `derive`→computed 0.958 (bal. **0.979**) |
| computation | grounding | `parametric` 1.000, `distractor` 1.000, `contextual` 1.000 → all retrieved |
| grounding | computation | `computed`→contextual 1.000, but `retrieved`→parametric **0.491** |

**R3's profile generalizes on qwen** — trained on numeric arithmetic where both
classes emit numbers, it correctly labels word-answer recall in a
differently-worded suite at 0.979, which it cannot have learned from answer
form. **It is more precisely a computation detector**: every grounding class
reads as `retrieved`, so the confident pole is "this is arithmetic".

**R7 qualification: this does not replicate on gemma** (0.979 → 0.696). The
*effect* transfers across architectures; the learned *decision boundary* does
not — consistent with W5.2.

---

## R1 — expert selection separates recall from derivation, against a null

→ `knowledge/routing.py`, `records/routing_null.*.json`

**The inherited number needed a null.** W5.1b read top-16 Jaccard 0.03–0.14 as
near-disjoint routing. But two *independent random* 16-subsets of 128 experts
overlap by K²/n = 2 on average — Jaccard ≈ **0.067**. Three of the four reported
layers sit at or below chance. It was set arithmetic, not evidence.

**With a proper null the effect is real anyway.** Three controls on identical
footing: *within-domain* (split one domain's prompts in half — the ceiling),
*across-domain*, and a *permutation null* that relabels pooled prompts. Selection
separates if **within > null > across**. It does, in all nine configurations:

| trace | K | within | across | null | layers p≤.05 | ρ(sep, dK) | perm p | partial \| entropy |
|---|---|---|---|---|---|---|---|---|
| qwen expert | 8 | 0.431 | 0.030 | 0.231 | 25/40 | +0.026 | 0.873 | −0.051 |
| qwen expert | 16 | 0.428 | 0.069 | 0.280 | 36/40 | +0.238 | 0.139 | +0.134 |
| qwen expert | 32 | 0.448 | 0.155 | 0.362 | 39/40 | **+0.397** | **0.013** | +0.295 |
| qwen gate | 8 | 0.465 | 0.138 | 0.333 | 22/40 | +0.150 | 0.352 | +0.037 |
| qwen gate | 16 | 0.508 | 0.188 | 0.391 | 35/40 | **+0.361** | **0.020** | +0.275 |
| qwen gate | 32 | 0.576 | 0.291 | 0.488 | 38/40 | +0.274 | 0.080 | +0.049 |
| gemma expert | 8 | 0.572 | 0.147 | 0.455 | 18/30 | **+0.498** | **0.007** | +0.499 |
| gemma expert | 16 | 0.564 | 0.189 | 0.455 | 29/30 | **+0.449** | **0.015** | +0.457 |
| gemma expert | 32 | 0.634 | 0.296 | 0.524 | 30/30 | **+0.524** | **0.003** | +0.566 |

**Routing separation also predicts causal ablation damage** — all nine ρ
positive, a sign test at 2⁻⁹ ≈ 0.002. But a usage-skew confound is real on qwen
and absent on gemma: partialling out entropy drops qwen to +0.05…+0.30 while
gemma is unchanged (+0.50…+0.57). Nine configurations were tested; Bonferroni
wants p ≤ 0.0056 and only gemma K=32 clears it. **The consistent sign is the
claim; no single cell should be quoted alone.**

**Scope:** domain workloads (free generation), not answer-token probes; n = 8
prompts per domain. Power is low and that is the true state of this data.

### Two measurement traps this cost

1. **The unit of independence is the prompt, not the token.** A first pass
   permuted tokens and returned p ≤ 0.05 at *every* layer — the signature of a
   null built from non-independent units. Permuting whole prompts drops the
   effective n from ~900 to 8.
2. **Token budget must be matched across every side.** A top-K set built from
   fewer tokens is noisier, which depresses Jaccard on its own — manufacturing
   the separation being tested for.

---

## R2 — context-grounded vs parametric (qwen only)

→ `knowledge/capture.py`, `knowledge/meter.py`, `records/meter.grounding.*.json`

Same question, same answer token, differing only in whether the fact is
available in context. A third `distractor` class (unrelated fact prepended)
closes the length confound.

| condition | n | bal. acc | null p95 | length-only | verdict |
|---|---|---|---|---|---|
| **contextual vs distractor** | 198 | **0.964** | 0.557 | 0.550 | **real** |
| — word answers only | 165 | **0.976** | 0.543 | 0.600 | **real** |
| — numeric answers only | 33 | 0.700 | 0.533 | 0.722 | length; do not quote |
| parametric vs distractor | 188 | 0.962 | 0.558 | **0.984** | length artifact |
| contextual vs parametric | 194 | 0.957 | 0.557 | **0.979** | length artifact |

Signal concentrates in **layers 24–31**, peaking at layer 30 (0.918 alone).
Those are **not** the layers W5.3's ablation ranked for knowledge damage.

**Caveats:** R4 showed this profile does *not* generalize (0.491 on R3's
retrieved probes), so it is more suite-specific than R3. The contextual side may
be substantially **induction/copying** — the answer token is literally present
earlier in the sequence.

### The first suite failed, and the failure is the useful part

The original `mechanism` suite scored **1.000**. Relabeled *"is the answer
numeric"* it scored **0.995** — the same information. Restricted to numeric
answers only, **0.500**, exactly chance.

Controlling topic had reintroduced the confound one layer down: every derivation
produced a *number* while most recall answers were *words*, so the router was
reading the form of the next token. The fix was to hold the **answer token
identical** across classes, which no amount of vocabulary matching achieves.

---

# Part II — What was ruled out

## R5 → R8 — fact *classification* does not generalize

→ `knowledge/identity.py`, `knowledge/position.py`, `records/identity.grid*.json`

R5, on a 12-entity × 4-relation grid of **countries**, holding out a whole
paraphrase: identify the relation with the entity fixed **1.000** (words 0.748);
identify the entity with the relation fixed **0.956** (words 0.763).

**R5b resolved the position caveat.** A first pass at earlier offsets was
incoherent because the axis crossed three regions. Offsets 0 to −9 sit in the
chat suffix, **byte-identical across every probe**; −10 and beyond enter the stem
where tokens differ *and* alignment breaks (different stems, different lengths),
which is what produced the −15 "collapse". Within the identical-token region:

| offset | token | distinct | relation (chance .250) | entity (chance .083) |
|---|---|---|---|---|
| 0 | `\n\n` | 1 | 1.000 | 0.956 |
| −4 | `\n` | 1 | **0.985** | **0.926** |
| −6 | `<\|im_start\|>` | 1 | **0.985** | **0.963** |
| −7 | `\n` | 1 | 0.948 | 0.978 |

Six tokens before the answer, at a position with **no surface information**,
entity identity is recoverable at 0.963. That is carried state, not the answer
forming. *Unexplained:* the signal is position-specific rather than smoothly
decaying (−2 dips to 0.652/0.422 carrying the *same* token as offset 0).

**R8 — the expanded grid kills the general claim.** Four semantically distinct
domains, 12 entities × 4 relations × 3 paraphrases, 576 probes:

| domain | n | relation: routing | words | entity: routing | words |
|---|---|---|---|---|---|
| **country** | 135 | **1.000** | 0.733 | **0.933** | 0.756 |
| element | 81 | 0.704 | 0.765 | 0.469 | 0.840 |
| author | 72 | 0.500 | 0.625 | 0.208 | 0.639 |
| composer | 49 | 0.388 | 0.735 | 0.365 | 0.712 |

**R5 was measured on the one domain where it works.** Not label noise (strict ≡
lenient) and not sample size (element n≈country n). Rung 4 — do addresses cluster
semantically — is moot in this form, since it presupposes addresses that exist
across domains.

**R8b — the analysis was under-powered.** Every rung-3 number used the **top-8
selected** experts; the traces hold **top-32 ranks** including experts the router
considered and rejected. At top-32: country entity 0.933→**0.978**,
`element.symbol` 0.611→**0.944**, author 0.208→0.306. The conclusion survives —
routing still beats words only on countries — but future work should start at
top-32. Gate *scores* through a 10,240-feature logistic collapse to 0.19–0.42,
which is overfitting on ~24 samples per fold, not absence of signal.

**The structural problem.** Across five conditions — four domains plus coarse
domain identification — routing beats reading the prompt words in **exactly
one**. Domain ID looked like a result at 0.844 but loses to bag-of-words at
0.868 (0.854 vs 0.919 on held-out entities). In all of these the prompt *names
the entity and the relation*, so bag-of-words is near an oracle. **This is what
motivated R9's reframing to a causal test.**

### Three authoring faults, worth not repeating

1. **Expected answers must match the model's spelling.** The strict first-token
   test rejected `Baroque` against `baroque` and ` metal` against `metal` — 71
   probes, element 81→123 and composer 52→80 once a lenient match was added.
   Same class of error as the leading-space bug in R2.
2. **Stems must cue the relation unambiguously.** "The writer Orwell worked
   chiefly in the ___" draws *genre* or *era*, not century. Author was the only
   domain lenient matching barely helped (72→73).
3. **A pooled number hides domain structure.** The combined figures (0.737 /
   0.543) describe no domain in the grid.

---

## R6 — fabrication detection: no

→ `knowledge/fabrication.py`, `records/fabrication.json`

Identical templates varying only entity familiarity: 8 well-known, 8 real but
obscure, 8 invented. **known vs fictional: 0.984** — but the `obscure` middle
class, added precisely to discriminate two hypotheses, answered unambiguously:

| obscure probes, scored by the known/fictional profile | read as "known" |
|---|---|
| answered **correctly** (n=25) | 1.000 |
| answered **wrongly** (n=7) | 1.000 |

Not a rarity detector (obscure did not group with fictional). Not a
retrieval-*success* detector either: the seven obscure facts the model got
**wrong** route exactly like the ones it got right. The signal tracks **whether
the entity exists in training data**, not whether retrieval is working.

**R6b — the headline was the wrong comparison.** Every well-known entity here is
a *single token*; obscure and fictional are 2–4. **Entity token count alone
separates known from fictional at 1.000**, beating routing. The informative
comparison is obscure vs fictional (both multi-token): routing **0.969**, length
0.531, token count 0.625.

**R6c — but that loses to one scalar too.** Predictive entropy separates the
classes **perfectly (1.000)**, beating routing on both comparisons:

| class | top-1 prob | entropy |
|---|---|---|
| known | 0.895 | 0.364 |
| obscure | 0.886 | 0.434 |
| fictional | 0.254 | **4.790** |

The model is simply uncertain about entities it has never seen. One scalar off
the output distribution, free, architecture-independent, no router needed.

---

## P1 — document membership / contamination: no

→ `knowledge/corpus.py`, `knowledge/membership.py`, `knowledge/detect.py`

2000 synthetic documents about invented entities, **membership assigned by coin
flip after generation**, LoRA fine-tune on members only, scored held-out.
**Blind bag-of-words baseline 0.488 — the benchmark is valid**, which is exactly
what the published MIA benchmarks fail (Das et al., *Blind Baselines Beat
Membership Inference Attacks for Foundation Models*).

| signal | ALL | x1 | x2 | x4 | x8 | x16 |
|---|---|---|---|---|---|---|
| **perplexity** | **0.942** | **0.806** | 0.903 | 0.981 | 1.000 | 1.000 |
| min-k% | 0.916 | 0.737 | 0.847 | 0.968 | 1.000 | 1.000 |
| entropy | 0.645 | 0.482 | 0.493 | 0.515 | 0.718 | 0.988 |
| hidden states | 0.593 | 0.476 | 0.496 | 0.497 | 0.624 | 0.855 |
| **routing** | **0.545** | **0.544** | 0.478 | 0.505 | 0.509 | 0.686 |

**Arm B (frozen router) closes it.** Adapting the router bought exactly nothing —
routing is 0.686 with an adapted router (arm A) and 0.692 with a frozen one, so
the weak x16 effect is upstream drift, not the router learning. That rules out
the most charitable explanation for routing's weakness.

**The benchmark also missed its own target.** Fine-tuning memorises far too
strongly: per-token NLL falls from 4.28 to 2.17 even for **non**-members. The
interesting regime is where output signals *fail* (perplexity ≈ 0.50 at document
level on pretrained models); here perplexity *succeeds* at 0.94, leaving no
headroom. **A single-epoch LoRA fine-tune on a tiny corpus is not a model of
single-epoch pretraining.** Fixing it means a far larger corpus, lower learning
rate, and targeting the regime where perplexity sits near 0.6–0.7.

**Kept regardless:** the certified corpus generator. Its blind baseline holds at
0.4966 across 8 seeds — a methods contribution independent of whether routing is
the observable.

---

---

## Inherited negatives — do not re-run

From §W5.3: targeted per-expert extraction (removing the single most important
of 128 costs <=8% of baseline NLL); layer-identity transfer across checkpoints
(top-4 overlap 1/4, Spearman +0.215 at n=30); domain-conditional extraction
(each domain touches 78-86% of all (layer, expert) pairs); and the compression
framing entirely.

---

Current plan, to-dos and operational cautions: [README.md](README.md).

---

# Part III — Deployment configuration and label-free regression detection

A second thread, on a different question: given fixed hardware, which model and
which runtime settings should you actually run — and how would you know if a
setting silently broke your model?

Two models throughout: `gemma-4-26b-a4b` (30 layers x 128 experts, 12.0 GB of
experts at 4-bit) and `qwen3.6-35b-a3b` (40 x 256, 16.9 GB), with
`gemma-4-e4b` (3.91 GB, not MoE) as the small-model reference. The offload
runtime is the vendored `ExpertCache`: a contiguous resident tensor of C expert
slots per layer with an id->slot map, backed by an on-disk expert store.

## F1 — The sweep that was canceled, and why that is the finding

The plan was to sweep resident capacity, measure benchmark accuracy at each
rung, and publish the accuracy-vs-memory curve nobody in the MLX ecosystem has.
Scoped at 16-50 GPU hours.

Reading `ExpertCache.ensure()` killed it. At `policy='exact'` a cache miss calls
`_install(e)`, which **fetches the real weights from disk before the gather**. A
miss costs latency and nothing else.

This also reclassified data that had been quoted three times as if it were
signal: the control-vs-offload accuracy deltas in the recorded suite run
(mmlu_pro 0.530 -> 0.500, but ifeval 0.850 -> **0.875** and qwen mmlu_pro
0.345 -> **0.370** in the other direction) are **numerical noise**. The file
names the source itself — `gather_qmm`'s sorted and unsorted paths "disagree by
~1.3e-3 absolute", and `do_sort = indices.size >= 64` sends decode and prefill
down different kernels.

**Two families of mechanism, and conflating them was the error:**

| family | mechanism | trades | floor |
|---|---|---|---|
| **lossless** | `policy='exact'` | memory <-> **speed** | none |
| **lossy** | `policy='static'` (non-resident -> zero slot), top-k reduction, heterogeneous precision | memory/speed <-> **accuracy** | yes |

**Method rule:** before scoping a sweep over a knob, read the knob's
implementation and establish whether it *can* move the measured quantity.

## F2 — Exact offload is semantically exact but not numerically reproducible

Pre-registered prediction: identical token sequences at every capacity. **It
failed**, informatively.

| comparison | identical sequences | first-divergence token |
|---|---|---|
| gemma cap-128 vs cap-32 | 10/16 | 26, 132, 128, 36, 21, 101 |
| qwen cap-256 vs cap-64 | 12/16 | 18, 2, 204, 20 |

Never at token 0, which by the pre-registration rules out a wiring bug. Two
capacity-dependent sources, both confirmed in the code:

1. **`_gather_sort(x, slots)` sorts on SLOTS, not expert ids.** Slot assignment
   is a function of capacity — a fixed permutation after `preload` at full
   capacity, churning LRU positions below it.
2. **Prefill is chunked below full capacity.** `__call__` splits the token axis
   when the working set exceeds the cache, because otherwise "the experts
   installed first are evicted before the gather runs and their slots read
   garbage."

Both change `gather_qmm`'s reduction order, hence rounding, and `do_sort` puts
it in *prefill* — so logits differ before the first token is emitted.

**The determinism control settles what that means.** The same capacity run twice
gives **16/16 identical**. Greedy decoding has no sampling, so the divergence is
a **deterministic function of cache size**, not run-to-run nondeterminism. The
runtime is reproducible and validatable; capacity simply has to be held fixed.

> `policy='exact'` is **semantically exact** — every routed expert's true weights
> are fetched, nothing dropped, nothing reading garbage — but **not numerically
> reproducible across capacities**. Deviations are unbiased rounding, not
> information loss.

**Consequence for anyone validating an offload runtime:** you cannot do it by
diffing outputs against a resident reference. Cache size changes the arithmetic.
Validation requires comparing accuracy *distributions*.

## F3 — The accuracy floor is flat, on short and long generations

popqa (64-token cap):

| resident | 100% | 75% | 50% | 38% | 25% | 12% |
|---|---|---|---|---|---|---|
| gemma | 0.2287 | 0.2287 | 0.2258 | 0.2328 | 0.2287 | — |
| qwen | 0.2900 | 0.2900 | 0.2900 | — | 0.2900 | 0.2900 |

The obvious objection is that a 64-token cap gives the capacity-dependent
rounding little room to compound. gsm8k (~250-token generations) answers it:

| resident | 100% | 50% | 25% |
|---|---|---|---|
| gemma | 0.9146 | 0.9150 | 0.9091 |
| qwen | 0.9397 | 0.9444 | 0.9447 |

0.5-0.6pp spread across a 4x capacity range, qwen drifting slightly *upward*.
Unbiased scatter, not decline.

**Coverage, stated precisely:** verified to 12% residency on qwen and 19% on
gemma for gsm8k/popqa; mmlu_pro's lowest tested rung is 25%. Nothing here
establishes the floor holds at 5% or 1%.

## F4 — Memory and speed, measured

**An instrumentation bug meant the first sweep measured no memory at all.**
`mx.get_peak_memory()` is a high-water mark from process start, so it captured
the full model load *before* `wrap()` freed the expert weights; every capacity
reported the same 13.48 GB (gemma) / 18.17 GB (qwen). Fixed with
`reset_peak_memory()` after wrapping plus `get_active_memory()`.

| resident | 100% | 75% | 50% | 38% | 25% | 19% | 12% |
|---|---|---|---|---|---|---|---|
| gemma GB | 13.48 | 10.49 | 7.49 | 6.00 | 4.50 | 3.76 | — |
| gemma tok/s | 99.67 | 59.63 | 52.99 | 46.21 | 38.46 | 34.48 | — |
| gemma TTFT s | 0.223 | 0.339 | 0.579 | 0.753 | 1.097 | 1.463 | — |
| qwen GB | 18.17 | 13.95 | 9.73 | 7.62 | 5.51 | 4.46 | 3.40 |
| qwen tok/s | 122.06 | 59.48 | 51.37 | 43.79 | 38.47 | 35.61 | 31.55 |
| qwen TTFT s | 0.193 | 0.329 | 0.523 | 0.658 | 0.851 | 1.034 | 1.425 |

**Two regimes.** A **cliff** from 100% to 75% (elasticity 2.05 gemma / 2.72
qwen), because full capacity short-circuits `ensure()` entirely — no
device->host sync — and stepping below it costs a sync per layer per token,
measured separately at **41% of per-token time**. Then a shallow
slope: post-cliff throughput elasticity **0.53 / 0.45**, i.e.

> **speed ∝ memory^0.5.** Halve the memory, lose ~30% of throughput. 4x less
> memory ≈ 2x slower.

(qwen 13.95 -> 3.40 GB is 4.10x less memory for 1.89x slowdown; 4.10^0.449 =
1.89.)

**The practical inversion:** the first 25% of memory savings costs *more* than
the next 63%. If you are going to offload at all, go deep — stopping at 75% pays
the whole cliff for almost none of the benefit.

**TTFT is the opposite story** — elasticity 1.43 / 1.04, so prefill degrades
2-3x faster than decode. That is the mechanism behind longbench's 9-13x tax:
decode's working set is 8 experts per layer with median reuse distance 2, while
prefill routes every token independently and approaches full expert coverage.
**TTFT budget, not accuracy, is what limits how deep you can compress.**

## F5 — The dominance result

The measured memory curve put qwen cap-32 at 3.40 GB and gemma cap-24 at 3.76
GB, both *below* the small model's 3.91 GB. Accuracy holds there:

| gsm8k | memory | accuracy | s/item |
|---|---|---|---|
| qwen-35b, exact, cap-32 | **3.40 GB** | **0.9447** | 10.21 |
| gemma-26b, exact, cap-24 | 3.76 GB | 0.9091 | 7.04 |
| gemma-e4b, resident | 3.91 GB | 0.8426 | 2.74 |

| popqa | memory | accuracy | | mmlu_pro (n=150) | memory | accuracy |
|---|---|---|---|---|---|---|
| qwen cap-32 | 3.40 GB | **0.2900** | | gemma cap-32 | 4.50 GB | **0.8456** |
| gemma cap-24 | 3.76 GB | 0.2287 | | qwen cap-64 | 5.51 GB | 0.8014 |
| e4b | 3.91 GB | 0.1508 | | e4b | 3.91 GB | 0.6364 |

**The offloaded big model dominates the natively-fitting small model on memory
and accuracy simultaneously, losing only speed** (2.6-3.7x s/item). So:

> On the lossless mechanism there is **no accuracy crossing point** within the
> measured range. The decision is about **latency**, not accuracy.

**Task-dependence is itself the finding.** gemma wins mmlu_pro (0.8456 vs
0.8014); qwen wins gsm8k (0.9447 vs 0.9091) and popqa (0.2900 vs 0.2287). No
single model dominates, so "which model at which capacity" is a genuine per-task
decision.

**Caveats the headline must carry.** Speed is the entire cost, and TTFT is worse
than throughput. The SSD footprint is not free — qwen at 3.40 GB RAM also needs
its 17 GB expert store on disk against e4b's ~4 GB total; offload trades disk
for RAM. And e4b measured through *this* scorer reads 0.8426/0.1508 against
the earlier 0.785/0.135 — a ~6pp cross-harness gap that would have flattered the
comparison had it been quoted rather than re-measured.

## F6 — The lossy path is dominated everywhere

`policy='static'` preloads the hot experts and routes everything else to a zero
slot. Pins come from per-layer usage counts over 129,690 (gemma) / 142,240
(qwen) saved routing records — wrap's default would pin experts 0..C-1, an
arbitrary subset that would understate it.

| | 100% | 75% | 50% | 25% |
|---|---|---|---|---|
| gemma gsm8k | 0.9146 | — | 0.5250 | **0.0250** |
| gemma popqa | 0.2353 | 0.1719 | 0.0612 | 0.0256 |
| qwen gsm8k | 0.9447 | — | **0.0854** | 0.0151 |
| qwen popqa | 0.2850 | — | 0.1500 | 0.0700 |

**Wiring control passed** — at full capacity static reproduces resident accuracy
(gemma 0.9146 vs 0.9050 control; qwen 0.9447 vs 0.9350), so the collapse is the
mechanism, not a bug.

The severity is far steeper than the decision shares suggest. The top half of
experts carry 79-81% of routing decisions, so 50% capacity zeroes only ~20% of
decisions — and that costs **qwen 91% of its gsm8k accuracy**. Multi-step
reasoning compounds errors across tokens. Severity is architecture-dependent:
qwen collapses harder than gemma.

At matched memory (~4.3-4.5 GB), gemma static scores **0.0250** against e4b's
**0.8426** while being *slower*. There is no regime in which it wins.

**Policy headroom, for completeness.** Simulated Belady/OPT against LRU shows
the gap grows as residency falls: qwen +17.3pp at 12% resident (0.614 -> 0.787),
+1.4pp at 71%. Prefetching is the practical route to that headroom, and — a
point the published prefetchers all miss — **prefill needs no prediction at
all**, because computing the router for every token of the prompt yields the
complete access sequence before any fetch. In prefill you can *be* Belady.

## F7 — Top-k reduction, and a hypothesis killed by its own bug fix

A widely-shipped runtime exposes top-k as an environment variable and documents
it as a *speed* setting (top-k=4 at 5.91 tok/s against 5.20 at top-k=6), with no
accuracy number anywhere. Reducing top-k below the trained value is the
route-around ablation R9e measured as damaging, so it is worth a number.

**First result, invalid.** gemma appeared catastrophic (gsm8k 0.9146 -> 0.1000
at k=6) against qwen's mild −1.6%, and a clean mechanistic story was written for
the split: qwen's `shared_expert_intermediate_size: 512` gives it an always-on
branch, and it routes 8 of 256 against gemma's 8 of 128.

That story was entirely an artifact. `gemma4_text.Router` selects with
`mx.argpartition(kth=-top_k)[..., -top_k:]`, which guarantees *membership* but
not *order*. Masking positions `0..k` dropped an **arbitrary 2 of 8** experts
rather than the 2 weakest — random ablation wearing top-k reduction's name.

**The control could not have caught it.** At `k == top_k` the mask keeps every
position regardless of order, so a 16/16-identical no-op test validated the
masking arithmetic and never touched the ranking assumption underneath.

**Method rule:** a wiring control must be able to fail on the assumption it is
meant to protect. Prefer a control that exercises the assumption (drop the
weakest expert) over one that trivially satisfies it (drop nothing).

Corrected, with a k=7 control that *can* fail:

| model | task | k=8 | k=7 | k=6 | k=4 |
|---|---|---|---|---|---|
| gemma | gsm8k | 0.9146 | 0.8995 | 0.9200 | 0.8700 |
| qwen | gsm8k | 0.9444 | 0.9548 | 0.9296 | 0.9239 |

The architecture split is **dead**, and so is the shared-expert explanation. At
n=200 the effect is also **not resolvable** — 95% CI half-widths are 3.3-4.0pp
and the whole spread sits inside them, with k=6 beating k=8 on both gemma tasks.

**Resolved at n=800** (qwen, gsm8k): 0.9424 vs 0.9195, McNemar 35 vs 14
discordant, **p = 0.0038**. So top-k=4 costs a real **−2.3pp** — small, but not
free, and shipped as a speed setting with no evidence either way.

*Note on measurement:* this implementation masks gate weights while the
SwitchGLU still gathers all 8 experts, isolating the accuracy cost and gaining
none of the speed. Its throughput is not comparable to a runtime that genuinely
selects fewer experts. Masking to -1e9 before the downstream softmax renormalizes,
which is the *charitable* reading — dropping without renormalizing would shrink
each block's output magnitude and flatter the comparison.

## F8 — Detecting a broken config without labels

**The question.** A team changes a deployment setting and needs to know whether
it degraded the model. Measuring accuracy needs a labeled eval set for their
actual task, which most teams do not have.

**The design.** Paired per-item comparison of the model's own per-token entropy
against a reference config, on the same items. Paired because item-to-item
variance in base entropy dwarfs the config effect. Reported as `d_z` = mean
shift / SD of the shift. The detector never sees the labels; the labels exist
only to score it, from F5/F6/F7.

**Two things had to be true.**

*The benign control.* "Entropy flags the broken configs" is worthless if it also
flags a 3x memory reduction that cost nothing — that is a change detector, and
it would fire on every deployment.

*The gen_len baseline.* Broken models ramble into the token cap, so generation
length alone might separate the arms. This project has been burned by exactly
that twice (R2's length artifact; R15's cap confound, where `gen_len` scored
0.878 until truncation contamination was removed).

| config | true damage | McNemar (paired) | p90 d_z | gen_len d_z |
|---|---|---|---|---|
| qwen exact cap-64 | +0.005 n.s. | 0 vs 1, p=1.00 | **−0.06** | +0.02 |
| gemma exact cap-32 | −0.006 n.s. | 2 vs 0, p=0.50 | **−0.08** | −0.03 |
| qwen top-k=4 | −0.023, p=0.004 | 8 vs 3 at n=200 | +0.56 | +0.40 |
| qwen static cap-128 | −0.854 | 170 vs 0 | +0.94 | +0.49 |
| gemma static cap-32 | −0.890 | 177 vs 0 | +0.83 | **+3.46** |
| qwen static cap-64 | −0.925 | 184 vs 0 | +0.64 | +0.75 |

**6/6 correct, no false positives**, effect sizes monotone in true damage, across
two architectures and two failure mechanisms. The benign arms are decisive: a
3-3.3x memory reduction with ~25% of generations differing *textually* (F2) and
1-2 discordant items in 200 moves **no** signal. It detects damage, not change,
and not perturbation.

**But gen_len appears to win on gemma** (+3.46 vs +0.83). Controlling for
truncation reverses it. Restricted to items where *neither* config truncated:

| | p90 | gen_len | n kept |
|---|---|---|---|
| qwen exact cap-64 (benign) | −0.05 | +0.02 | 197 |
| qwen static cap-128 | **+1.20** | +0.29 | 169 |
| qwen static cap-64 | **+2.54** | **−0.30** | 99 |
| gemma exact cap-32 (benign) | −0.08 | −0.03 | 200 |
| gemma static cap-32 | **+1.72** | +1.58 | 44 |

Entropy's signal roughly **doubles** once truncated items stop diluting it with
hundreds of rambling tokens; gen_len's **collapses**, and on qwen static cap-64
it **inverts**. Its entire apparent advantage was truncation — 1.5% -> 15.5% ->
50% (qwen) and 0% -> 78% (gemma) — which is a property of the token cap you
chose, not of the model.

**Sensitivity is the commercial point.** Top-k=4's −2.3pp regression is flagged
by entropy at **n=200**, where paired McNemar on labels reads p=0.227 and needs
**n=800** to reach p=0.004. A continuous per-token signal resolves what a binary
per-item one cannot without 4x the eval budget — and it needs no labels at all.

### F8b — a second task, a calibrated threshold, and terse damage

gsm8k caps generations at 960 tokens, which gives a broken model room to ramble.
popqa caps at 64, so length carries far less information. Repeating the whole
design there does three jobs at once.

**The benign null, from 13 arms.** F3 established accuracy is flat across
capacity, so every exact-policy rung is a benign config. Across two models and
two tasks:

> **13 benign configurations, p90 d_z in [−0.090, +0.071].** All signals, all
> arms, stay inside ±0.13.

On qwen/popqa, accuracy is *identical* (0.2900) at every rung from 100% down to
12% residency, which is the most emphatic form of F3's floor.

That replaces the eyeballed 0.5 with a data-derived bar. **0.3 — roughly 3x the
observed null — gives zero false positives across all 13 benign arms** and
catches strictly more than 0.5 did. The original threshold was ~5x the null and
was costing real detections.

**Terse damage: the detector holds, and gen_len INVERTS.** qwen/popqa:

| config | Δacc | p90 d_z | gen_len d_z |
|---|---|---|---|
| static cap-128 | −0.140 | **+1.01** | **−0.27** |
| static cap-64 | −0.220 | **+1.07** | **−0.36** |

With a 64-token cap the damaged model produces *shorter* output, so generation
length points the **wrong way** — a "longer means broken" heuristic misses this
entirely, while entropy is unmoved by the change of regime. Together with the
truncation-controlled analysis above, this closes the question of whether the
detector was riding on verbosity: it works when damage lengthens output and when
it shortens it.

**Sensitivity is task- and architecture-dependent, and gemma/popqa is the weak
case:**

| config | Δacc | p90 d_z | at 0.5 | at 0.3 |
|---|---|---|---|---|
| static cap-96 | −0.057 | +0.26 | missed | missed |
| static cap-64 | −0.168 | +0.49 | marginal | **caught** |
| static cap-32 | −0.203 | +0.82 | caught | caught |

A config costing **16.8 points** of accuracy sits right on the old threshold.
Specificity is excellent (13/13 benign arms clean); **sensitivity is the weaker
half**, and it degrades on short-output tasks and on effects below ~10pp.

### F8c — "confidently wrong": a structural answer, not just an absence

Every failure mode above presents as *increased* uncertainty. The dangerous
untested case is damage that makes the model **more** confident — a silent false
negative. "Ruled out" is not reachable (it is an existential claim over all
possible configs), so the substitute is deliberate adversarial construction.

**The specificity control settles the shape of the problem before any
construction succeeds or fails.** Scaling logits by a>1 leaves the greedy argmax
mathematically unchanged while collapsing entropy. Verified on 8 items:
correctness arrays **identical**, p90 falling 1.362→0.827, 1.105→0.078,
3.060→0.135, 2.319→0.036. So it is a config with *provably zero* accuracy change
and an enormous entropy shift.

That forces a choice the product must expose rather than paper over:

| detector | catches confidently-wrong damage | false-positives on benign confidence shifts |
|---|---|---|
| one-sided (entropy ↑ only) | **no** — blind by construction | no |
| two-sided (any shift) | yes | **yes** — sharpening fires hard on zero damage |

This is a **calibration choice, not a gap to be closed by more testing.** A
one-sided detector is sound for the failure modes measured here and structurally
cannot see confidence-increasing damage; a two-sided one sees it but alarms on
harmless confidence changes.

*In progress (2026-07-30):* three adversarial constructions, to measure how much
the one-sided form actually gives up — **expert substitution** (non-resident
experts routed to a real resident expert rather than a zero slot: well-formed
arithmetic, wrong weights, which is the shape confidently-wrong damage would
take, and a design someone would plausibly ship), **top-k = 1 and 2** (leaves
computation entirely well-formed and is a knob real runtimes expose), and
**sharpening at x2/x3** as the control. Results pending.

### Limits

- **Sensitivity is the weak half.** Specificity is 13/13; the −5.7pp gemma/popqa
  config is missed at any threshold that keeps that record.
- **Top-k=4 cleared the old threshold narrowly** (+0.56 vs 0.5) but sits ~6x the
  measured null, so the sensitivity claim is stronger than first stated.
- **Confidence-increasing damage is structurally invisible to a one-sided
  detector** (F8c), and the two-sided alternative has a demonstrated false
  positive.
- **gemma's restricted set is n=44** (78% truncated).
- **One model family per failure mode:** top-k is qwen-only, because the gemma
  top-k arm was the one invalidated by F7's ranking bug.
- **Needs logits**, so local or self-hosted inference, or an API returning
  logprobs. Needs a reference config; it cannot score a config in isolation.
- **Reports that something moved, not how much accuracy was lost.**

## Prior art for Part III

**Applied, not ours.** MoE expert offloading with an LRU cache is established —
[Mixtral-offloading](https://arxiv.org/pdf/2312.17238) (2023) — as is
speculative expert prefetch from prior-layer or prior-token routing
([HOBBIT](https://arxiv.org/pdf/2411.01433), ExpertFlow, CommitMoE,
[ST-MoE](https://arxiv.org/pdf/2606.15453)). Predictive entropy as an
uncertainty signal is textbook. Paired McNemar, Belady's optimal, and Pareto
dominance are all standard. Multiple MIT-licensed implementations of the runtime
exist and are downloadable today.

**What appears to be new.**

1. **Accuracy measured across MoE offload configurations at all.** Four
   competing implementations were surveyed; none reports perplexity, benchmark
   scores, or any output-quality comparison against a resident model. The
   accuracy-vs-residency floor (F3), the dominance result (F5), and the
   quantified cost of the lossy path (F6) had no published counterpart found.
2. **Exact-policy offload is not numerically reproducible across capacities, but
   is deterministic at a fixed capacity** (F2) — with the specific mechanisms
   (slot-ordered sort, capacity-triggered prefill chunking) identified. This
   invalidates output-diffing as a validation method for any such runtime.
3. **Prefill admits *optimal* caching** (F6). Every published prefetcher
   predicts future routing because decode forces it; prefill computes the whole
   layer's routing before fetching anything, so Belady is directly achievable
   there. Not found stated.
4. **Label-free config-regression detection validated against
   independently-known damage** (F8), including the benign controls that
   distinguish a damage detector from a change detector, and the
   truncation-controlled analysis separating entropy from generation length.

**What the literature predicts and this confirms.** SSD offload exploits MoE
sparsity to hide transfer latency but does not reduce the energy of the
transfers ([arXiv 2508.06978](https://arxiv.org/html/2508.06978v1)) — relevant
to "otherwise impossible", weaker for "cheaper".

### F8d — a third damage mechanism: the quantization ladder

The same gemma-26b-a4b at five bit-widths from one pipeline, all at
`--policy none` so quantization is the only variable. This matters because
expert-zeroing and top-k reduction are both *routing* interventions; quantization
is unrelated to routing, so a detector that works on all three is not tracking
one mechanism's signature.

**popqa — monotone in damage:**

| config | accuracy | Δacc | p90 d_z | flagged |
|---|---|---|---|---|
| 4-bit (reference) | 0.2408 | — | — | — |
| **mixed_4_6** | 0.2462 | **+0.005** | **−0.033** | no ✓ |
| mixed_3_6 | 0.2256 | −0.015 | +0.662 | yes ✓ |
| 3-bit | 0.2094 | −0.031 | +1.016 | yes ✓ |
| **2-bit** | 0.0100 | **−0.231** | **+2.821** | yes ✓ |

Perfectly ordered, and the one genuinely benign config — `mixed_4_6` is
*marginally better* than the reference — reads −0.033, inside the ±0.10 null.
It also resolves **−1.5pp**, finer than anything earlier in F8.

**gsm8k — same verdicts, one ordering inversion:**

| config | accuracy | Δacc | p90 d_z | gen_len d_z |
|---|---|---|---|---|
| **mixed_4_6** | 0.9200 | **+0.015** | **−0.037** | +0.14 |
| mixed_3_6 | 0.7250 | −0.180 | **+1.470** | +0.66 |
| 3-bit | 0.4322 | −0.473 | **+0.954** | +1.07 |
| 2-bit | 0.0100 | −0.895 | +5.104 | +6.12 |

Every verdict is correct, but 3-bit (−47pp) scores *below* mixed_3_6 (−18pp).
The `gen_len` column names the cause: 3-bit rambles far more (+1.07 against
+0.66), and truncated items dilute p90 across hundreds of low-information
tokens — the same effect F8's truncation-controlled analysis already documented.
Ordering is reliable on short-output tasks; on long-output tasks it degrades
while the flag itself does not.

**Two things this establishes.**

1. **d_z magnitude is mechanism-dependent, not a universal function of accuracy
   loss.** Quantization at −1.5pp reads +0.662; expert-zeroing at −5.7pp reads
   +0.258. So "something moved, and roughly how hard" is supportable;
   "you lost k accuracy points" is not.
2. **An actionable configuration result.** `mixed_4_6` is free on both tasks
   (+0.5pp and +1.5pp, d_z ≈ −0.03); 3-bit and below is destructive, and 2-bit
   removes 96% of popqa and 99% of gsm8k accuracy. That independently confirms
   the literature's finding that aggressive quantization amplifies behavioral
   drift, and it is exactly the guidance the frontier is meant to produce.

### F8e — the adversarial constructions failed to evade detection

"Ruled out" is not reachable for a claim about all possible configs, so the
substitute was deliberate construction of the counterexample.

**Expert substitution** routes non-resident experts to a real *resident* expert
rather than a zero slot: correct magnitudes, well-formed arithmetic, wrong
weights. That is the shape confidently-wrong damage would take, and it is a
design someone would plausibly ship.

| model | config | Δacc | p90 d_z | detected |
|---|---|---|---|---|
| qwen | substitution cap-64 | −0.230 | **+1.055** | ✓ |
| gemma | substitution cap-32 | −0.207 | **+0.702** | ✓ |

**Top-k = 1 and 2** leave computation entirely well-formed and are a knob real
runtimes expose. Both are destructive — qwen falls to **0.0000** at k=1 — and
both are detected.

**The sign separates damage from benign confidence change.** Every damaged
configuration measured, across three mechanisms and two architectures, gives a
POSITIVE d_z (+0.26 to +5.10). The only negative d_z belongs to logit
sharpening, whose greedy output is bit-identical by construction (verified: the
correctness arrays match exactly) and whose accuracy delta is 0.0000:

| model | sharpen ×2 | sharpen ×3 |
|---|---|---|
| qwen | −1.382 | −1.363 |
| gemma | −1.005 | −1.014 |

So a **one-sided (positive-only) detector is correct on every arm tested**,
including the arm built specifically to defeat it. The one-sided/two-sided
tension described earlier is real in principle, but no mechanism attempted here
landed on the wrong side of it.

**Honest scope:** this is "three unrelated mechanisms attempted, none evaded",
not "ruled out". A confidence-*increasing* failure mode would still be invisible
to the one-sided form; none was constructible here.

**Two bugs found by these arms, both in this repo's code.**
`_TopK`'s tuple branch used `[..., -k:-k+1]`, which degenerates to the empty
slice `[..., -1:0]` at k=1 — the scores branch was guarded for exactly this and
the tuple branch was not. And `variants` read `ent[n_prompt-1]` unguarded when a
config was damaged badly enough to generate **nothing at all**, which qwen does
at top-k ≤ 2. Both fixed; the second now records `empty_gen` and keeps the row,
because an immediate stop is a damage observation rather than a reason to drop
the item and bias the sample.

## F9 — label-free detection of catastrophic forgetting

The same paired design with the reference moved: reference = the adapter-free
base, candidate = a LoRA checkpoint, items = **popqa, a domain the adapters were
never trained on** (the corpus was synthetic entity facts). Ground truth is
accuracy on those items, which the detector never sees.

| checkpoint | popqa accuracy | Δacc | p90 d_z | flagged |
|---|---|---|---|---|
| base (reference) | 0.2900 | — | — | — |
| `adapter-attn` | 0.1300 | −0.160 | **+0.84** | ✓ |
| `adapter-router` | 0.1250 | −0.165 | **+0.74** | ✓ |
| `adapter-qa` | 0.0300 | −0.260 | **+0.76** | ✓ |

All three flagged, none near the ±0.10 benign null. This is the gap the
literature search identified: forgetting is normally detected by "accuracy on
held-out test sets from previous tasks, **though this still requires labels**",
and the entropy work that exists uses entropy either as a *training mechanism*
(EAFT) or over *attention* distributions. A label-free output-entropy monitor
over held-out prompts was not found published.

Ordering is again imprecise — `adapter-qa` does the most damage (−26pp) but
scores below `adapter-attn` (−16pp) — consistent with F8d's finding that d_z
magnitude is mechanism-dependent.

**A caveat this raises for R9f, which is this project's own prior work.** The
injection adapters destroyed general factual recall: `adapter-qa` takes popqa
from 0.290 to **0.030**, i.e. ~90% of parametric recall gone. R9f's
injected-fact results were therefore measured on a model that had lost most of
its pre-existing knowledge. That does not invalidate R9f's *within-model*
contrasts — `own` vs `para` vs `samerel` vs `other` are all measured on the same
damaged model — but any claim that generalizes from those adapters to a healthy
model needs this stated.

## F10 — RAG context utilization: the claim holds, the refinement does not

F8's design pointed at retrieval: reference = irrelevant context, candidate =
relevant context, on the same questions.

**The first attempt failed for a design reason**, recorded because the failure
is instructive. It used `probes.build_grounding`, whose facts are things like
the capital of Australia — which qwen already knows. Handing it the answer moved
answer-NLL by **+0.15 nats**: there was no retrieval benefit for entropy to
detect, and the `Fact:` prompt frame moved entropy *more* than the content did
(+0.68 vs +0.38). A utility measure needs items the model cannot already answer.

**Corrected on PopQA's long tail** (accuracy 0.23-0.29, so most items are
genuinely unknown), with a Q-A reference block and a length-matched distractor
drawn from a different question:

| model | Δ NLL (label) | Δ entropy (free) | d_z |
|---|---|---|---|
| qwen, knew it | −8.33 | −1.146 | −0.96 |
| qwen, did not know | −9.59 | −1.105 | −0.71 |
| gemma, knew it | −1.55 | −1.172 | −1.38 |
| gemma, did not know | −3.60 | −1.944 | −1.63 |

**What holds.** Relevant context produces a large NLL drop *and* a large entropy
drop on both models, consistently negative (d_z −0.71 to −1.63) against a
length- and format-matched distractor. The product claim — *"did the retrieved
context actually help this query?"* — is supported without labels, and it is
the sign of d_z that carries it.

**What does not.** The finer claim, that the entropy drop scales with how much
the model *needed* the context, holds on gemma (didn't-know items drop 1.944
against 1.172 for known ones; Pearson(parametric NLL, Δ entropy) = **−0.431**,
the predicted direction) and is flat on qwen (−1.105 vs −1.146; Pearson
**−0.002**).

**The reason is a flaw in my design, not in the signal.** Parametric NLL is
scored against `gold[0]`, a single alias, while the task scorer accepts *any*
alias. qwen's median parametric NLL is **16.10** — it assigns near-zero
probability to that particular surface form on essentially every item, so the
"knew it" half is not really a knew-it half and the split has no contrast.
gemma's median is 4.89 and does span knowing from not-knowing. The fix is to
score against the best-matching alias (minimum NLL over the alias set) rather
than the first, and re-run; until then the conditional claim is unmeasured on
qwen rather than refuted.

### F10b — the alias fix, and why item 3 still does not validate

The flaw was real and the fix is right: score parametric NLL against the
**best-matching alias**, since the task scorer accepts any of them. It also
costs nothing — entropy at the first answer position is a property of the
distribution rather than of the target, so one forward pass serves every alias,
and only the CPU-side tokenisation repeats.

Re-run on 300 PopQA items, both models:

| | qwen | gemma |
|---|---|---|
| **contextual − distractor** Δ NLL | −4.060 | −2.276 |
| **contextual − distractor** Δ entropy | −1.670 | −0.823 |
| d_z | **−1.25** | **−0.37** |
| per-item sign agreement | 72% | **49%** |
| Pearson(Δ NLL, Δ entropy) | −0.093 | −0.371 |
| **parametric − distractor** Δ entropy (frame only) | **−2.307** | −0.548 |
| Pearson(parametric NLL, Δ entropy) | −0.119 | **+0.066** |

**Verdict: not validated.** Three problems, two of them newly visible.

1. **Sign agreement on gemma is 49% — chance.** The pooled means move in the
   right direction but individual items do not, which is what a deployment tool
   would actually rely on.
2. **The frame effect exceeds the relevance effect on qwen.** Adding an
   *irrelevant* reference block moves entropy by **2.307** nats; making that
   block relevant moves it by 1.670. So entropy responds more to whether a
   reference block exists than to whether it is useful. The intended comparison
   is format-matched (contextual vs distractor both carry a block), so it is not
   invalid — but a real RAG pipeline whose retrieved passages vary in length or
   shape would have that variation swamp the signal.
3. **The prompts are off-distribution.** `popqa_probes` builds raw
   `Q: ...\nA:` completions while both models are instruct-tuned and the rest of
   this project routes through `suite.build_prompt`, which applies the chat
   template. Median best-alias first-token NLL is 10.0 (qwen) and 14.2 (gemma) —
   near-zero probability on the gold answer for models that score 0.23-0.29 on
   the task. That gap says the measurement is being taken in a regime the models
   are not operating in.

**The conditional claim did not survive the fix.** It previously read −0.431 on
gemma, in the predicted direction; with correct alias scoring it is **+0.066**,
the wrong direction, while qwen moves to a weak −0.119. The earlier result was
an artifact of scoring against one arbitrary surface form.

**Status: open, not refuted.** The direction is right on qwen (d_z −1.25, 72%
agreement) and the mechanism is plausible, but two design faults have to be
fixed before the claim means anything — chat-templated prompts, and a
format-matched no-retrieval arm rather than the bare-question `parametric` one.
This is a redesign, not a patch, and it is the weakest of the three Tier-1
applications. F8 (config regression) and F9 (forgetting) do not depend on it.

## F11 — the agentic axis: what compression actually breaks

Agentic loops compound per-step error — published figures put 95% per-step over
10 steps at 59% end-to-end, 90% at 35%. So a per-step drop too small to matter
on a benchmark is decisive over a task, which makes *which capability* a
compression setting damages the operative question.

IFEval measures instruction adherence — emit this format, use only these fields,
stop here — which is what an agent depends on and what MMLU-style accuracy does
not measure. Run on the same five-point quantization ladder as popqa and gsm8k,
so Δaccuracy is comparable at matched bit-width:

| config | **ifeval** Δacc | popqa Δacc | gsm8k Δacc |
|---|---|---|---|
| 4-bit (reference) | 0.8492 | 0.2408 | 0.9050 |
| mixed_4_6 | −0.006 | +0.005 | +0.015 |
| mixed_3_6 | **−0.215** | −0.015 | −0.180 |
| 3-bit | **−0.392** | −0.031 | −0.473 |
| 2-bit | **−0.734** | −0.231 | −0.895 |

**The finding is not the one the experiment was designed to test.** Instruction
adherence is not uniquely brittle — it degrades much like gsm8k. What is
anomalous is **popqa: short factual recall is uniquely ROBUST**, losing 1.5pp
where the other two lose 18-21pp at the same bit-width.

The mechanism is generation length. popqa answers in one entity token under a
64-token cap; ifeval and gsm8k produce long structured output where errors
compound *within a single generation*, before any agentic loop is involved.

**The actionable consequence.** Validating a quantization choice on short
factual QA **understates damage to structured generation by ~14x** (1.5pp against
21.5pp at mixed_3_6). And structured generation is precisely the agentic
workload. Compounded over a 10-step task, instruction adherence 0.849 -> 0.634
takes end-to-end success from **19.5% to 1.1%** — a 95% relative collapse from a
setting that reads as a rounding error on factual QA.

That is also a caution about this project's own frontier: F5's dominance result
rests on gsm8k, popqa and mmlu_pro. popqa's robustness means it should not carry
weight in a compression decision aimed at agentic use.

## F12 — `p90` is the wrong aggregation, and `max` is the right one

The "ordering degrades on long-output tasks" limitation recorded in F8d was not
a limitation of the method. It was the wrong statistic.

| task (cap) | signal | mixed_3_6 (−21%) | 3-bit (−39/47%) | 2-bit (−73/90%) | |
|---|---|---|---|---|---|
| gsm8k (960) | p90 | +1.470 | +0.954 | +5.104 | inverted |
| gsm8k (960) | **max** | +1.424 | +1.890 | +4.810 | **monotone** |
| ifeval (768) | p90 | +0.669 | +0.309 | +2.979 | inverted |
| ifeval (768) | **max** | +1.623 | +1.921 | +3.184 | **monotone** |
| popqa (64) | p90 | +0.662 | +1.016 | +2.821 | monotone |
| popqa (64) | **max** | +0.740 | +1.026 | +3.052 | **monotone** |

`max` is monotone in true damage on **all three tasks**; `p90` only on the
64-token one. The reason is mechanical: p90 is a percentile over generated
tokens, so a config that rambles for 700 confident tokens pushes the 90th
percentile down into the confident region and dilutes the signal. `max` takes
the single worst moment, which no amount of confident padding can move.

**It costs no specificity.** On the benign `mixed_4_6` arm, `max` reads −0.007
(gsm8k), −0.193 (ifeval), −0.065 (popqa) against p90's −0.037, −0.340, −0.033 —
better on the two long tasks, indistinguishable on the short one.

**And it converts a near-miss into a clear detection.** IFEval 3-bit costs 39.2pp
of accuracy and scored p90 **+0.309**, barely over the 0.3 threshold; `max` gives
**+1.921**.

**Why `p90` was inherited.** It was selected in Stages A-D for a different job —
ranking *individual answers* by how likely they are to be wrong, where the whole
distribution is informative. Config comparison asks whether the same items got
harder, and the single worst moment is sharper evidence. The aggregation was
carried across without re-deriving it for the new question.

### F12b — correction: `max` does not dominate, and the verdicts are what is robust

The claim first written here — that `max` should simply replace `p90` — was
drawn from the quantization ladder alone and does not survive the full arm set.
Recomputed against the correct references:

| config | Δacc | p90 | max |
|---|---|---|---|
| qwen exact cap-64 (benign) | +0.005 | −0.06 | −0.02 |
| gemma exact cap-32 (benign) | −0.005 | −0.08 | −0.04 |
| qwen top-k=4 | −0.016 | **+0.56** | +0.40 |
| qwen static cap-128 | −0.854 | **+0.94** | +0.70 |
| gemma static cap-32 | −0.890 | +0.83 | **+2.53** |
| qwen static cap-64 | −0.925 | +0.64 | **+1.19** |
| forget `adapter-qa` | −0.260 | +0.76 | +0.78 |
| qwen substitution | −0.230 | +1.06 | +1.07 |
| gemma substitution | −0.207 | **+0.70** | +0.54 |
| sharpen ×3 (zero damage) | 0.000 | −1.36 | −1.43 |

`max` wins on gemma static cap-32 and qwen static cap-64 and is nearer zero on
both benign arms; `p90` wins on top-k=4, qwen static cap-128 and gemma
substitution. **Neither dominates on effect size.**

What *is* robust is the verdict. At threshold 0.3 every damaged arm flags under
both aggregations and every benign arm stays clean under both, across all five
mechanisms and both architectures. The flag does not depend on the choice; only
the fine ordering does.

**Corrected statement.** Report several aggregations rather than picking one.
Prefer `max` when *ordering* matters on long-output tasks, where `p90` provably
inverts (F12) and where `max` rescued a borderline detection (IFEval 3-bit,
+0.309 → +1.921). Do not claim it is uniformly stronger.

**A process bug this surfaced.** The quantization arms were written into the
same `{model}.{task}` namespace as the offload arms, so re-running `analyze`
silently replaced the saved reference in `analysis.gemma.gsm8k.json` with
`quant_4bit`. Any number read from those files after that run compares against
the wrong baseline. The table above is recomputed from the raw per-item records
with explicit reference pairings. Tags need namespacing by experiment, not just
by config.


## F13 — the floor holds on a long-output task, and the two families separate hard

F3 established the accuracy floor on popqa and gsm8k. F11 then showed popqa is
the least sensitive benchmark available, which left the floor claim — and F5's
dominance result, which inherits it — leaning mostly on gsm8k. IFEval is the
third task: 768-token structured output, and the capability agents run on.

**The floor holds.** gemma, `policy='exact'`, six rungs:

| capacity | % experts | accuracy | Δacc | p90 d_z | max d_z |
|---|---|---|---|---|---|
| 128 | 100% | 0.8782 | — | — | — |
| 96 | 75% | 0.8687 | −0.010 | +0.008 | +0.040 |
| 64 | 50% | 0.8687 | −0.010 | +0.033 | −0.042 |
| 48 | 38% | 0.8788 | +0.001 | −0.129 | −0.077 |
| 32 | 25% | 0.8636 | −0.015 | −0.039 | +0.002 |
| 24 | 19% | 0.8636 | −0.015 | −0.054 | +0.002 |

**1.5pp of spread across a 5.3x capacity range**, non-monotone, and the detector
correctly stays silent at every rung (|d_z| ≤ 0.13 under both aggregations).
F3 and F5 now rest on three tasks including a long-output one, not on gsm8k
alone.

**And the two mechanisms separate by an order of magnitude.** Same model, same
task, two ways to save memory:

| configuration | memory | ifeval |
|---|---|---|
| 4-bit, fully resident | 13.23 GB | 0.8492 |
| **2-bit, fully resident** | **7.35 GB** | **0.1150** |
| 4-bit + exact offload, 100% resident | 13.48 GB | 0.8782 |
| **4-bit + exact offload, 50% resident** | **7.49 GB** | **0.8687** |
| 4-bit + exact offload, 19% resident | 3.76 GB | 0.8636 |

At **matched memory (~7.4 GB)** quantization scores **0.115** and exact offload
scores **0.869** — 7.6x the accuracy for the same footprint. Stated within each
family so the two 4-bit builds (which differ by ~3pp for unrelated reasons)
cannot flatter the comparison:

> For the same ~44% memory reduction, **2-bit quantization costs 73 points of
> instruction adherence; exact offload costs 1.0 point.** Offload then goes on
> to 3.76 GB — half of what 2-bit needs — for 1.5 points.

This is the two-family split (F1) measured on the axis that matters most for
agentic use, and it is the sharpest form of the frontier's advice: **when a MoE
model does not fit, stream the experts; do not crush the weights.** The cost is
latency and SSD (F4), not capability.

**Caveat.** One model, one long-output task. The quantization ladder is gemma
only, and `mixed_4_6` shows quantization is not uniformly destructive — a
well-chosen mixed scheme is free (−0.6pp). The claim is about aggressive
bit-width reduction, not about quantization as such.

### F10c — redesigned, and now partially validated

Three faults were identified in F10b. All three are fixed here: PopQA long-tail
items so there is something to detect; every prompt through the vendored
`build_prompt` so the models are in-distribution; and entropy measured over the
model's **own generated span** rather than one token — the likely cause of the
chance-level per-item agreement, and also what the shipped `clausius` measures.

The baseline is `irrelevant` — a context block of the same shape drawn from a
different item — not `nocontext`. Only content differs, which removes the frame
confound that muddied F10b.

| arm | qwen | gemma | e4b |
|---|---|---|---|
| nocontext | 0.2467 | 0.1933 | 0.1100 |
| irrelevant (baseline) | 0.2000 | 0.1433 | 0.0700 |
| relevant | 0.9733 | 0.9833 | 0.9667 |
| haystack (answer + 9 distractors) | **1.0000** | **1.0000** | **0.9967** |

**H1 — is there anything to detect? YES, emphatically.** Context fixed 240, 257
and 278 items and broke **zero** on all three models.

**H2 — does relevant context lower entropy against a format-matched baseline?
YES, on all three.** Paired shift, relevant − irrelevant:

| signal | qwen | gemma | e4b |
|---|---|---|---|
| max | −3.174 (d_z −1.86) | −1.181 (−1.23) | −1.616 (−1.50) |
| p90 | −2.677 (−1.69) | −0.969 (−1.25) | −1.399 (−1.42) |
| first | −3.107 (−1.82) | −1.067 (−1.17) | −1.611 (−1.49) |

So the aggregate claim holds: **whether the retrieval supplied the answer is
detectable without labels**, replicated across three models and two
architectures. That is a usable RAG diagnostic — it catches a retriever
returning nothing useful.

**H3 — the per-item claim — is NOT TESTABLE ON THIS TASK, and the reason is now
demonstrated rather than assumed.** It needs items where context helped and
items where it did not, to rank between. There are **0, 0 and 1** of the latter.

Two attempts to create that variance failed in the same direction:

- Burying the answer among **nine distractors made the task EASIER**, not harder
  (0.973 → 1.000 on qwen). A multi-entry block reads unambiguously as a
  reference list, where a single pair can read as part of the question.
- A **4B model** copies out of that ten-entry haystack at **0.9967**. The
  ceiling is the task, not model capability.

> **When the answer is verbatim present in the context, retrieval "helping" is
> not graded — it is binary, and it essentially always works.** PopQA-style
> lookup therefore cannot produce the per-item variance H3 requires, at any
> model scale available here.

**What H3 would need:** *partial* relevance — a passage topically right that
does not state the answer, or an answer requiring two passages combined. PopQA
exposes no subject/relation fields to build oblique statements from, so this
needs a different dataset (multi-hop QA, or real retrieved prose rather than
question-answer pairs).

**Status: aggregate claim validated on three models; per-item claim out of scope
for this task.** That is an upgrade from F10b's "does not validate", and the
remaining gap is a dataset problem rather than a signal problem. Three attempts
is where this stops.

## F14 — the ladder against an *unquantized* reference, adjudicated with labels

F8d measured the quantization ladder against a **4-bit** reference, and so did
all thirteen benign configurations behind the ±0.10 null. No arm in this record
had ever used an unquantized reference. When `clausius compare` was pointed at
one it flagged **bf16 → 4-bit**, which the corpus gave no way to classify: the
detector reports that something moved, never how much accuracy fell.

Setup: `gemma-4-26b-a4b-it`, mlx-community bf16 converted locally with
`mlx_lm.convert --q-group-size 64` to 8/4/3/2-bit, so bit-width is the only
variable. Entropy on 60 unlabeled mixed instruction prompts at cap 4096;
accuracy on the gsm8k test split, shuffled seed 0.

**The entropy ladder is monotone in bit depth:**

| arm | bits/weight | max d_z | flagged |
|---|---|---|---|
| **8-bit** | 8.500 | **+0.172** | no |
| 4-bit | 4.502 | +0.654 | yes |
| 3-bit | 3.503 | +2.066 | yes |
| 2-bit | 2.503 | +8.134 | yes |

**And the labels adjudicate it:**

| arm | n | accuracy | Δacc | b | c | p |
|---|---|---|---|---|---|---|
| bf16 (reference) | 1319 | 0.8476 | — | — | — | — |
| **8-bit** | 300 | 0.8733 | **+0.0133** | 3 | 7 | 0.3438 |
| **4-bit** | 1319 | 0.8256 | **−0.0220** | 70 | 41 | **0.0076** |
| 2-bit | 100 | 0.0100 | −0.8600 | 86 | 0 | <0.0001 |

*b = bf16 correct and the arm wrong; c = the reverse; p is an exact two-sided
binomial McNemar over the discordant pairs.*

**Four things this establishes.**

1. **The flag was a true detection, not a false positive.** 4-bit costs 2.2
   accuracy points, 70 items broken against 41 fixed, p=0.0076. Entropy called
   it correctly on 60 unlabeled prompts.
2. **The 8-bit arm rules out the artifact explanation.** If quantized-vs-bf16
   comparisons flagged merely for being quantized, 8-bit would flag too. It does
   not — d_z +0.172, clean — and its accuracy drifts the *wrong* way (+1.3pp,
   p=0.34), which is what a genuine null looks like on both instruments at once.
3. **The null is wider here than ±0.10.** 8-bit is benign by labels and still
   reads +0.172 on this prompt set, above the ceiling of F8b, leaving ~1.7x
   margin to the threshold rather than ~3x. *Superseded in part by F14c:* the
   same pair against the same bf16 reference reads −0.062 on gsm8k, so the
   widening belongs to the prompt set, not to the reference being unquantized.
4. **The sensitivity gap, measured directly.** Walking the paired items in order
   gives the n at which labels would have caught what entropy caught:

   | n | b | c | p |
   |---|---|---|---|
   | 200 | 7 | 8 | 1.0000 |
   | 500 | 28 | 17 | 0.1352 |
   | 800 | 42 | 28 | 0.1196 |
   | **878** | 50 | 31 | **0.0448** ← first crosses 0.05 |
   | 1319 | 70 | 41 | 0.0076 |

   **Labels needed n=878. Entropy needed 60 unlabeled prompts** — a ~15x
   difference in eval budget, on top of needing no gold answers at all. This is
   a second instance of F8's top-k result (−2.3pp, flagged at n=200, labels
   needing n=800), on an unrelated mechanism and a different model.

**The effect is stable, and a mid-run read of it was not.** Items 0–499 give
Δacc −0.0220; items 500–1318 give −0.0220. But *unpaired* running accuracy part
way through the second block suggested the gap was collapsing to ~−0.8pp, which
was sampling noise in an incomplete arm. McNemar depends on the item-level
discordance, not on either arm's marginal accuracy, so partial-run marginals are
not a preview of the verdict — a caution worth keeping for any long paired run.

**Limits.**

- **Entropy and accuracy were measured on different prompt sets** — 60 mixed
  instruction prompts for the former, gsm8k for the latter. F8d measured both on
  the same items. So "the flag corresponds to this accuracy loss" assumes the
  damage is general rather than specific to either set. Closing that gap needs an
  entropy capture over the gsm8k prompts themselves, and it is the obvious next
  run.
- **3-bit has no labeled arm.** Its d_z +2.066 sits between two measured points
  and is assumed, not shown, to sit between them in damage. *Closed by F14d:
  3-bit scores 0.2940, −56.6pp, and the assumption was badly wrong about the
  magnitude even though the ordering held.*
- **8-bit is "not significant", not "proven zero"** at n=300; a real effect
  smaller than this test resolves is not excluded. *Extended in F14d to n=800;
  still null, still not zero.*

Reproduce with `python -m knowledge.quantladder analyze`, over the records in
`records/quantladder/`. No model or accelerator needed for the analysis.

### F14b — correction: rambling tracks off-distribution, not damage

F8's `gen_len` baseline note reads *"Broken models ramble into the token cap"*,
and `core.py` used the same sentence to justify excluding truncated items. That
held for F8's own arms — expert-zeroing and static capacity did ramble — but it
does not generalize, and three runs in this session contradict it in both
directions:

| run | cap | the config that rambled | the config that did not |
|---|---|---|---|
| quantization ladder | 512 | 4-bit, **healthy**: 47/60 truncated | 2-bit, **destroyed**: 3/60 |
| forgetting, numina | 4096 | fine-tune: 33/50 | its own base: 0/50 |
| forgetting, magicoder | 4096 | fine-tune: 22/50 | its own base: 0/50 |

In the first, the *undamaged* config is the one that ran into the cap while the
config with 1% gsm8k accuracy terminated early — 2-bit emits degenerate text and
stops. In the other two, a fine-tune rambles on domains it was not trained on
while the checkpoint it was fine-tuned *from* terminates cleanly on the same 50
prompts.

**What actually predicts truncation is how far off-distribution the prompt is
for that checkpoint** — which correlates with damage only when the damage is
what moved the model off-distribution.

**This does not weaken the truncation filter**, whose two independent
justifications survive: truncated items dilute the signal across hundreds of
low-information tokens (excluding them roughly doubled the measured effect), and
keeping them lets the detector ride on generation length, a property of the cap
rather than of the model. Only the stated mechanism was wrong. `core.py`'s
docstring is corrected; F8's text stands as the record of what its arms showed.

**It does add a caution the docs did not carry.** The filter drops long-output
items, and F11 measures those as the family most sensitive to compression —
short factual recall understates damage to structured generation by ~14x. On
heterogeneous traffic the filter therefore biases the surviving set toward the
*least* sensitive prompts. F8d and F8e never saw this because each ran a single
task at a per-task cap, so the filter removed a roughly uniform slice; it bites
on exactly the mixed production traffic the README recommends as ideal. Read
`n_dropped_truncated` as a statement about *which* prompts were measured, not
only how many.

### F14c — same items, both instruments: the limit closed, and a correction

F14 measured entropy on 60 mixed instruction prompts and accuracy on gsm8k —
two distributions — and flagged that as its main weakness. This repeats the
entropy capture over **the exact gsm8k items already scored** (shuffled indices
0-199, cap 1024), so both instruments finally refer to one set.

| comparison | entropy d_z | flagged | Δacc, same 200 | b | c | p |
|---|---|---|---|---|---|---|
| bf16 → 8-bit | **−0.062** | no | +0.0050 | 3 | 4 | 1.0000 |
| bf16 → 4-bit | **+0.839** | yes | +0.0050 | 7 | 8 | 1.0000 |

**The flag survives, and strengthens.** On gsm8k items 4-bit reads +0.839 against
+0.654 on the mixed set, and the 8-bit control is cleaner (−0.062 against
+0.172). F14's headline is confirmed on matched distributions rather than
narrowed.

**And on those same 200 items, labels see nothing at all.** Δacc is +0.5pp — the
*wrong* direction — with b=7 against c=8 and p=1.0000. Yet the same two
checkpoints over the full 1319-item split give −2.2pp at p=0.0076, so the
damage is real and this 200-item sample simply cannot resolve it.

That is the project's central claim reproduced under the tightest possible
control: **same checkpoints, same 200 items, same run — entropy flags, labels
are blind, and the ground truth from 6.6x more items agrees with entropy.** F8's
top-k arm and F14's n=878 walk both argued this by comparing budgets across
different item counts; this shows it at a fixed one.

**The correction.** F14's third point attributed the widened null (+0.172) to the
reference being unquantized. That is wrong. The same 8-bit checkpoint, against
the same bf16 reference, reads −0.062 on gsm8k — comfortably inside ±0.10. What
changed between the two measurements is the prompt set: 60 heterogeneous
instruction-following prompts versus 200 items of one task. Two candidates, not
separated here:

- **Prompt-set heterogeneity**, consistent with F14b — a mixed set truncates
  unevenly and mixes output-length regimes, both of which move per-item
  variance, and d_z is a ratio to that variance.
- **Small-n noise**, since the mixed-set figure rests on 59 surviving pairs
  against 178.

Either way the corpus-derived ±0.10 should be read as a property of the runs it
came from — single-task, per-task caps — rather than a universal null. The 0.3
threshold retains margin in both regimes, which is the part that matters
operationally.

**A note on what this does not show.** 8-bit's accuracy is statistically
indistinguishable from bf16 on every measurement taken (+0.5pp at n=200, +1.3pp
at n=300, both n.s.), and its entropy is inside the null on gsm8k. That is
consistent with 8-bit being free, but an effect smaller than these tests resolve
is not excluded — and F14's own n-walk is the reminder that "not significant at
this n" and "absent" are different statements.

### F14d — the complete matched ladder, and what its shape says

F14c matched one arm pair on one distribution. This completes it: every arm now
has **entropy and labels on the same gsm8k items**, closing both of F14's
remaining limits.

| arm | bits/weight | entropy d_z | flagged | accuracy | Δacc | b | c | p |
|---|---|---|---|---|---|---|---|---|
| 8-bit | 8.500 | **−0.062** | no | 0.8462 (n=800) | +0.0050 | 17 | 21 | 0.6271 |
| 4-bit | 4.502 | +0.839 | yes | 0.8256 (n=1319) | −0.0220 | 70 | 41 | 0.0076 |
| 3-bit | 3.503 | +1.697 | yes | 0.2940 (n=500) | **−0.5660** | 286 | 3 | <0.0001 |
| 2-bit | 2.503 | +5.303 | yes | 0.0100 (n=100) | −0.8600 | 86 | 0 | <0.0001 |

Monotone on both instruments, in agreement, across four arms and a 25x range of
damage. Specificity 1/1, sensitivity 3/3, no arm out of order.

**Entropy is ordinal, not proportional — within a single mechanism.** 3-bit loses
**25x more accuracy than 4-bit** (−56.6pp against −2.2pp) and reads **2x the
d_z** (+1.697 against +0.839). F8d established that d_z magnitude is
mechanism-dependent by comparing quantization against expert-zeroing; this shows
the same compression *inside one mechanism*, on matched items, where every
confound F8d had to argue around is held fixed. The ordering is trustworthy; the
spacing is not. "Something moved, and roughly how hard" survives; any attempt to
read accuracy loss off d_z — even along a single axis — does not.

**3-bit is a broken deployment, not a degraded one.** At 0.2940 against bf16's
0.8600 on the same 500 items, with 286 items broken against 3 fixed, it is far
worse than its position between +0.839 and +5.303 suggests. F14's assumption
that it "sits between two measured points in damage" was right about the order
and badly wrong about the distance — recorded because that is exactly the error
the paragraph above warns against, made in this document before it was measured.

**8-bit at n=800 remains null.** +0.5pp, b=17 against c=21, p=0.63 — drifting the
wrong way, as at n=300. Consistent with 8-bit being free; still not a proof of
zero.

### F14e — the offload path through the public API (integration, not a finding)

F8 already measured offload arms label-free, and their accuracy was measured
accuracy, so nothing here is a new result. What had never been exercised is the
README's claim that the same three commands cover an offload setting:
`knowledge/regress.py` carries its own capture implementation and never touches
the packaged `clausius`, so the public `model_obj` extension point had no
coverage against a real wrapper.

Driving the vendored `offload_model.wrap` through `clausius.capture(model_obj=...)`
on 100 gsm8k items, cap 1024:

| comparison | max d_z | verdict | F8's sign |
|---|---|---|---|
| exact c256 → exact c64 (50% resident, experts *fetched*) | **+0.073** | clean | −0.05, clean ✓ |
| exact c256 → static c64 (non-resident experts *dropped*) | **+2.917** | REGRESSION | +0.64, flagged ✓ |

Both signs reproduce; the magnitudes differ, as they must — different model,
task and prompt set. The integration works: 30 layers wrapped, provenance
recorded as `<preloaded Model>`, entropy computed over the wrapped forward pass.

One incidental observation, relevant to F14b: the lossy arm truncated **50/100**
where the exact arms truncated 11. Here damage *does* drive rambling. F14b's
claim is that rambling does not *reliably* indicate damage, not that the two
never coincide — severe damage can itself push a model off-distribution, and
this is that case.

**A gap this exposed in the tool.** The first attempt at this check failed at
`compare` with 12 of 40 items surviving, because it used a prompt set already
measured as truncating 78% at the chosen cap. The truncation curve and the
fail-fast added for exactly this failure live in `cli.py`, so a caller using the
Python API — which is the documented path for offload wrappers and patched
runtimes — receives neither the curve nor the refusal. The protection is on the
interface least likely to be scripted.

## F15 — the torch backend: sensitivity transfers, the threshold does not

The detector was Apple-only by implementation, not by constraint — it needs
greedy generation and one teacher-forced pass yielding full-vocabulary logits,
which `transformers` gives on CUDA, CPU or MPS. The question is not whether the
code runs but whether the *decision rule* survives the move.

Setup: `Qwen/Qwen2.5-0.5B-Instruct` on MPS in float16 — chosen over CPU/fp32
deliberately, because a real device boundary and a half-precision upcast are the
parts CPU makes into no-ops. 60 prompts at cap 512 for entropy; gsm8k n=150 for
accuracy, so every arm's damage is **measured rather than assumed**. Damage is
per-tensor symmetric fake-quantization, the same mechanism family as F8d.

| arm | gsm8k | Δacc | entropy d_z | 95% CI | verdict |
|---|---|---|---|---|---|
| identity (same model twice) | — | — | **+0.000** | [0.00, 0.00] | clean ✓ |
| **fp16 → fp32** (benign) | 0.4000 | **−0.0067** | **+0.306** | [+0.18, +0.43] | **REGRESSION ✗** |
| 8-bit fake-quant | 0.3600 | −0.0467 | +0.347 | [+0.13, +0.55] | REGRESSION ✓ |
| 4-bit fake-quant | 0.0000 | −0.4067 | +2.989 | [+2.69, +3.44] | REGRESSION ✓ |

**What transfers.** Determinism is exact: the same model captured twice gives
d_z precisely 0.000, across a device boundary and a half-precision upcast.
Sensitivity is 2/2 against measured damage, and ordering is monotone on both
instruments — 0.000 < 0.347 < 2.989 against 0 < 4.7pp < 40.7pp.

**What does not.** A benign change flags. fp16 → fp32 on identical weights costs
0.67pp — one item in 150, indistinguishable from noise — and reads **+0.306**,
over the threshold. Specificity is **0/1** here, against 13/13 on mlx.

**The consequence is worse than one false positive.** Benign reads +0.306 with a
CI of [+0.18, +0.43]; a real −4.7pp regression reads +0.347 with [+0.13, +0.55].
The intervals overlap almost entirely. On this stack the detector separates
*gross* damage cleanly and **cannot distinguish a benign precision change from a
~5pp regression at all**. The ±0.10 null does not hold: the floor here is at
least 0.31, which by the calibration argument behind the 0.3 default (~3x the
null) argues for a threshold nearer 0.9.

**Why this is the hard benign case, and why that matters.** The argument for the
threshold transferring was that comparisons happen *within* one runtime, so
kernel-level numerical differences cancel in the paired difference. fp16 → fp32
is the case where that argument does not apply, because the numerics difference
*is* the configuration change under test. It is also an entirely realistic
deployment change. So this is simultaneously the most demanding benign arm
available and one a user would plausibly run.

**Candidate explanations, not separated here.**

- **Model scale.** 0.5B is far smaller than anything in F8–F14, and a small
  model's entropy is both noisier and more precision-sensitive. The same arm on
  a 26B model might sit well inside the null.
- **The prompt set.** F14c already showed the null moving from +0.172 to −0.062
  between prompt sets on identical checkpoints.

**Limits.** One model, one prompt set, one backend-and-device combination.
Per-tensor fake-quantization is cruder than any production quantizer, so the
damaged arms test damage detection rather than characterizing bitsandbytes,
GPTQ or AWQ. **CUDA itself is untested**, as is multi-GPU sharding.

**Status: the backend ships, marked experimental, with the 0.3 default flagged
as not transferring.** The remedy is not a CUDA spot-check — F8d already
established that d_z magnitude is mechanism-dependent, so validating one CUDA
quantizer would not license a claim about the others. It is the calibration
recipe now in the README: measure your own floor from configurations you have
independent reason to believe are equivalent, which is the procedure that
produced 0.3 in the first place.

### F15b — the same class of change is benign at production scale on mlx

F15's one benign arm flagged: a compute-precision change on a 0.5B model in
torch read d_z +0.306 with accuracy unchanged. That left three explanations
unseparated — the backend, the 0.5B scale, or the choice of control. It also
left an uncomfortable possibility open: that precision changes are a blind spot
of the *method*, in which case the problem would follow the detector onto the
stack that actually ships.

This tests that directly, on mlx at 26B. fp32 is not available at this size —
`gemma-4-26b-a4b-it` in float32 is ~96 GB of weights on a 137 GB machine — so
the feasible pure-numerics change is **bf16 → fp16**: identical weights, equal
bit width, a different exponent/mantissa split, and therefore different
arithmetic at every operation. Same 200 gsm8k items as F14c, labels measured.

| stack | change | Δacc | b/c | entropy d_z | verdict |
|---|---|---|---|---|---|
| **26B, mlx** | bf16 → fp16 | +0.0300 (p=0.109) | 2/8 | **−0.043** [−0.17, +0.12] | **clean ✓** |
| 0.5B, torch (F15) | fp16 → fp32 | −0.0067 | — | +0.306 [+0.18, +0.43] | REGRESSION ✗ |

**The detector is not inherently fooled by precision changes.** At production
scale on the shipped runtime, a change that alters every matmul reads −0.043 —
inside the ±0.10 null, with an interval that comfortably contains zero. Whatever
F15 found, it is not a property of the method that follows it onto mlx.

**A side observation.** Gemma-family models are normally specified bf16 because
their activation magnitudes overflow fp16's range. That did not materialize
here: fp16 scored 0.8750 against bf16's 0.8450 on the same items — b=2 against
c=8, p=0.109, so not significant, but certainly not damaged. The overflow
concern is real in general and did not bite this checkpoint on this task.

**What is still unresolved.** The two rows differ in scale *and* in framework,
so this does not tell you which caused F15's false positive. The run that would
isolate it — the same fp16 → fp32 change on a 7B model in torch, holding
framework and change-type fixed — did not complete: it targeted a `backend`
argument that had been removed from the shipping branch hours earlier, while the
job sat queued. A self-inflicted failure, recorded because the experiment is
still owed. It remains the open question in the torch scope decision, and the
work belongs on `feat/torch-backend` where the argument still exists.

### F15c — the torch false positive is a small-model artifact, not a backend defect

F15 found a benign precision change flagging at d_z +0.306 on a 0.5B model in
torch, and could not attribute it: backend, scale, or choice of control. F15b
showed the same *class* of change reads clean at 26B on mlx, but differed in
scale **and** framework, so it isolated nothing.

This holds framework, change-type, prompt set and cap fixed at F15's settings
and moves only model size.

| stack | change | Δacc (measured) | entropy d_z | 95% CI | verdict |
|---|---|---|---|---|---|
| 0.5B, torch | fp16 → fp32 | −0.0067 | **+0.306** | [+0.18, +0.43] | FLAGGED ✗ |
| **7B, torch** | fp16 → fp32 | **0.0000** | **−0.036** | [−0.33, +0.21] | **clean ✓** |
| 26B, mlx | bf16 → fp16 | +0.0300 (p=0.109) | −0.043 | [−0.17, +0.12] | clean ✓ |

**The flag is a property of the 0.5B model, not of the runtime.** Identical
framework, identical change, identical prompts — and 7B reads −0.036 where 0.5B
read +0.306. The 7B arm is also the cleanest benign control in this record:
gsm8k 0.8700 against 0.8700, a delta of exactly zero.

**Torch specificity is now 1/2, with the failure confined to the smallest
model.** F15's headline — "sensitivity transfers, the threshold does not" — was
too strong. It does transfer at the scales anyone deploys; it fails at a size
where nobody runs a regression gate.

**A plausible mechanism, not established here.** d_z is a mean divided by the
standard deviation of the same paired differences. A precision change perturbs
every logit slightly and systematically; what varies is how large that
systematic shift is *relative to* item-to-item variability. At 0.5B the shift
appears to dominate the spread; by 7B it does not. That predicts the effect
should shrink monotonically with scale, which two points cannot confirm.

**What this does and does not settle.** It removes the reason F15 gave for
calling the backend uninterpretable, and it supplies the first *measured-benign*
torch control the detector calls correctly. It does not clear the bar in
EXPERIMENT.md: that asks for three benign configurations of the same kind as the
mlx controls, a framework-neutral damaged checkpoint, and a CUDA run with real
quantizers. This is one benign configuration, of a kind mlx never tested, on
MPS. **Every torch measurement in this record is still MPS; CUDA remains
entirely unmeasured.**

**Process note.** This experiment failed twice before producing a number, both
times self-inflicted: once on a `backend` argument removed from the shipping
branch while the job sat queued, once because `sys.path.insert(repo_root)` does
not shadow an editable install under a `src/` layout, so it silently imported
the wrong checkout. Roughly two hours. Both are in the README's operational
cautions, and the second is why the script now asserts on `module.__file__`
before doing any work.
