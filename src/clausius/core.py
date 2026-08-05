"""Label-free detection of deployment regressions.

You changed something — quantization, offload capacity, top-k, a LoRA
checkpoint, an inference backend — and you need to know whether it degraded the
model. Measuring that normally needs a labelled eval set for your actual task,
which most teams do not have.

This compares the model's own **per-token entropy** before and after, paired on
the same unlabelled prompts. No gold answers, no judge model, no eval set. The
signal is a byproduct of a forward pass you are already running.

Validated against five unrelated damage mechanisms whose true damage was
measured independently — quantization, expert zeroing, top-k reduction, expert
substitution and LoRA fine-tuning — on two architectures. See FINDINGS.md (F8,
F9, F11, F13).

The design decisions below are not defaults chosen by taste. Each one is a
result:

THRESHOLD 0.3. Thirteen configurations known to be harmless — offload capacity
    changes of up to 8x, a mixed-precision scheme, across two models and two
    tasks — produce |d_z| <= 0.10. 0.3 is ~3x that null. An earlier eyeballed
    0.5 was ~5x the null and missed real regressions, including one costing 39
    accuracy points.

ONE-SIDED BY DEFAULT. Every damaged configuration measured raises entropy. The
    only configurations that LOWER it are ones with provably zero damage —
    scaling logits leaves greedy output bit-identical while collapsing entropy,
    and a two-sided test flags it. So the default flags increases only. Damage
    that made a model MORE confident would be invisible; three mechanisms were
    deliberately constructed to produce that and none did, but it is not
    excluded. `two_sided=True` trades that for a known false positive.

PAIRED, SAME PROMPTS. Item-to-item variance in base entropy dwarfs the effect of
    a config change, so unpaired comparison is hopeless. Mismatched prompt sets
    are rejected rather than silently compared.

TRUNCATED ITEMS EXCLUDED. Items that hit the token cap dilute the signal across
    hundreds of low-information tokens — excluding them roughly doubled the
    measured effect — and dropping them stops the detector riding on generation
    length, which is a property of the cap you chose rather than of the model.

    This note used to say that a damaged model rambles into the cap. The
    measurements do not support it. At cap 512 a *healthy* 4-bit checkpoint
    truncated 47/60 items while a destroyed 2-bit one truncated 3; two LoRA
    fine-tunes truncated 22 and 33 of 50 on held-out domains where their own
    base truncated none. Rambling tracks how far off-distribution a prompt is,
    not how damaged the model is. The filter is right; the reason was wrong.

    It carries a cost on heterogeneous traffic. The items it drops are the
    long-output ones, and F11 measures those as the family most sensitive to
    compression — validating on short factual recall understates damage to
    structured generation by ~14x. So on a mixed prompt set the filter biases
    what survives toward the least sensitive items. Set a cap generous enough
    that truncation is rare rather than routine, and read `n_dropped_truncated`
    as a statement about which prompts were measured, not only how many.

NO SINGLE AGGREGATION WINS. `max` is monotone in damage where `p90` inverts on
    long-output tasks; `p90` gives a larger effect on some others. Every verdict
    agrees under both across all five mechanisms, so all are reported and the
    flag is not sensitive to the choice.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIGNALS = ('max', 'p90', 'mean', 'mean_top10', 'first', 'gen_len')
DEFAULT_SIGNAL = 'max'
DEFAULT_THRESHOLD = 0.3

# Below this many paired items the effect size is noise. `compare` refuses;
# `capture` reports how far a run is from it while the model is still loaded.
MIN_PAIRED_ITEMS = 20

# Caps a truncation curve is reported at. Only the entries at or under the cap
# a capture actually used are answerable — see `truncation_curve`.
CAP_LADDER = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


def aggregate(entropy: np.ndarray, n_prompt: int) -> dict:
    """Every aggregation of a per-token entropy sequence, over generated tokens.

    `n_prompt` is clamped: a configuration can be damaged badly enough to
    generate nothing at all (top-k <= 2 on one model does exactly this), and an
    unguarded index into an empty generated span raises rather than recording
    the observation. An immediate stop is a data point about the damage.
    """
    n_p = min(n_prompt, len(entropy))
    gen = entropy[n_p - 1:] if n_p >= 1 else entropy
    if len(gen) == 0:
        gen = entropy[-1:] if len(entropy) else np.array([0.0])
    return {
        'first': float(gen[0]),
        'mean': float(np.mean(gen)),
        'max': float(np.max(gen)),
        'p90': float(np.percentile(gen, 90)),
        'mean_top10': float(np.mean(np.sort(gen)[-max(1, len(gen) // 10):])),
        'gen_len': float(len(gen)),
    }


@dataclass
class Capture:
    """One configuration's behaviour on a prompt set."""
    model: str
    tag: str
    prompts: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'model': self.model, 'tag': self.tag,
                                 'prompts': self.prompts, 'rows': self.rows,
                                 'meta': self.meta}))
        return p

    @classmethod
    def load(cls, path) -> "Capture":
        d = json.loads(Path(path).read_text())
        return cls(model=d['model'], tag=d['tag'], prompts=d.get('prompts', []),
                   rows=d['rows'], meta=d.get('meta', {}))


@dataclass
class Result:
    """The verdict, and everything needed to disbelieve it."""
    flagged: bool
    effect: float
    signal: str
    threshold: float
    one_sided: bool
    n_compared: int
    n_dropped_truncated: int
    detail: dict

    @property
    def verdict(self) -> str:
        return 'REGRESSION' if self.flagged else 'clean'

    def __str__(self) -> str:
        lines = [f"{self.verdict}  ({self.signal} d_z = {self.effect:+.3f}, "
                 f"threshold {self.threshold}, "
                 f"{'one' if self.one_sided else 'two'}-sided)",
                 f"  compared {self.n_compared} paired items"
                 + (f", dropped {self.n_dropped_truncated} truncated"
                    if self.n_dropped_truncated else ""),
                 "  all signals: " + "  ".join(
                     f"{k} {v:+.2f}" for k, v in self.detail.items())]
        return "\n".join(lines)


@dataclass
class TruncationCurve:
    """What every tighter generation cap would have cost, from one capture.

    The point of reporting this at capture time is that the answer is already
    in hand and the alternative is finding out at compare time, after a second
    capture has also been paid for.
    """
    cap_used: int
    n_items: int
    n_censored: int
    rows: list
    floor: int = MIN_PAIRED_ITEMS

    @property
    def survivors(self) -> int:
        """Items that would survive truncation filtering at the cap used."""
        return self.n_items - self.n_censored

    @property
    def usable(self) -> bool:
        """Whether a paired comparison against this capture can succeed.

        A candidate can only ever truncate MORE items, never fewer, so a
        reference already under the floor is a comparison that cannot be run.
        """
        return self.survivors >= self.floor

    def as_dict(self) -> dict:
        """JSON-serialisable form, for storing on a Capture's meta."""
        return {'cap_used': self.cap_used, 'n_items': self.n_items,
                'n_censored': self.n_censored, 'survivors': self.survivors,
                'usable': self.usable, 'floor': self.floor, 'rows': self.rows}

    def __str__(self) -> str:
        lines = []
        for r in self.rows:
            mark = "  <- this run" if r['cap'] == self.cap_used else ""
            warn = ("  below compare's floor of %d" % self.floor
                    if r['survivors'] < self.floor else "")
            lines.append(f"    cap {r['cap']:>5}: {r['truncated']:>3} truncated, "
                         f"{r['survivors']:>3} survive{warn}{mark}")
        if self.n_censored:
            lines.append(
                f"    {self.n_censored} item(s) reached the {self.cap_used} cap; "
                f"their true lengths are unknown, so this table cannot be "
                f"extended above it")
        return "\n".join(lines)


def truncation_curve(capture, caps=CAP_LADDER, floor=MIN_PAIRED_ITEMS):
    """Truncation counts this capture would have shown at tighter caps.

    Exact downward, silent upward. Every item that finished has its true
    generated length recorded, so the count at any cap at or under the one used
    is not an estimate — it is what would have happened. Items that hit the cap
    have no recorded true length, so no cap above it is answerable at all and
    none is reported; extrapolating there would be inventing data.

    Accepts a `Capture` or a path to a saved one.
    """
    cap = capture if isinstance(capture, Capture) else Capture.load(capture)
    cap_used = cap.meta.get('max_tokens')
    if not cap_used:
        raise ValueError(
            "capture has no max_tokens in meta; the curve is defined relative "
            "to the cap that was used and cannot be computed without it")
    lengths = [r['ent']['gen_len'] for r in cap.rows]
    n = len(lengths)
    # matches capture()'s own truncation test: n_gen >= max_tokens - 2
    counted = sorted({c for c in caps if c <= cap_used} | {int(cap_used)})
    rows = [{'cap': c,
             'truncated': sum(1 for x in lengths if x >= c - 2),
             'survivors': sum(1 for x in lengths if x < c - 2)}
            for c in counted]
    censored = sum(1 for x in lengths if x >= cap_used - 2)
    return TruncationCurve(cap_used=int(cap_used), n_items=n,
                           n_censored=censored, rows=rows, floor=floor)


def apply_chat_template(tokenizer, prompt, chat=None):
    """Wrap a prompt in the model's chat format, if it has one.

    Defaults to ON for any tokenizer that declares a template. Feeding a raw
    completion-style prompt to an instruct-tuned model puts it off-distribution:
    it rambles instead of answering, and the run comes back truncation-dominated
    — 19 of 25 items hit a 256-token cap in the first end-to-end test of this
    library. Truncated items are dropped, so the comparison then rests on
    whatever few happened to terminate.

    `chat=False` is the escape hatch for base models and for deliberately
    measuring raw-completion behaviour.
    """
    if chat is False:
        return prompt
    tmpl = getattr(tokenizer, 'chat_template', None)
    if chat is None and not tmpl:
        return prompt
    msg = [{'role': 'user', 'content': prompt}]
    try:
        return tokenizer.apply_chat_template(msg, add_generation_prompt=True,
                                             tokenize=False)
    except Exception:
        return prompt


def capture(model, prompts, tag='run', max_tokens=512, adapter=None,
            chat=None, model_obj=None, tokenizer=None,
            progress=None) -> Capture:
    """Run `prompts` through a model configuration and record its entropy.

    `model` is a path or HF id. Pass `model_obj`/`tokenizer` instead to reuse an
    already-loaded model, or to measure a configuration this library has no way
    to construct itself — a patched runtime, a custom cache, an offload wrapper.
    That is the intended extension point: clausius does not manage
    configurations, it compares whatever two runs you hand it.

    Arguments are checked before mlx is imported, so a caller wiring up that
    extension point finds out on any machine rather than only on a Mac.
    """
    if model_obj is None and model is None:
        raise ValueError(
            "capture needs either `model` (a path or HF id) or a preloaded "
            "`model_obj` plus `tokenizer`")
    if model_obj is not None and tokenizer is None:
        raise ValueError(
            "`model_obj` was given without `tokenizer`; capture needs both — "
            "the tokenizer renders the chat template and counts prompt tokens, "
            "which is what separates prompt positions from generated ones")

    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except ImportError as e:  # the only part of this library that needs an accelerator
        raise ImportError(
            "capture needs mlx-lm, which runs on Apple Silicon only:\n"
            "    pip install 'clausius[mlx]'\n"
            "compare, aggregate and truncation_curve are pure numpy and run "
            "anywhere — you can re-analyse captures on any machine.") from e

    if model_obj is None:
        model_obj, tokenizer = (load(model, adapter_path=adapter) if adapter
                                else load(model))
    rows, t0 = [], time.time()
    for i, prompt in enumerate(prompts):
        rendered = apply_chat_template(tokenizer, prompt, chat)
        text = generate(model_obj, tokenizer, prompt=rendered,
                        max_tokens=max_tokens, verbose=False)
        ids = tokenizer.encode(rendered + text)
        n_prompt = len(tokenizer.encode(rendered))
        n_gen = len(ids) - n_prompt
        # teacher-forced pass over prompt+completion: entropy at the positions
        # where the model actually committed, not at whatever it would have
        # produced under a different sampler
        logits = model_obj(mx.array([ids])[:, :-1]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ent = np.asarray((-mx.sum(mx.exp(logp) * logp, axis=-1))[0].tolist())
        del logits, logp
        rows.append({'ent': aggregate(ent, n_prompt),
                     'truncated': n_gen >= max_tokens - 2,
                     'empty_gen': n_gen <= 0,
                     'text': text[:2000]})
        if progress:
            progress(i + 1, len(prompts))
    # A capture is read back weeks later, so it has to say what produced it. With
    # a preloaded model there is no path to record, and str(None) — which this
    # used to write — is worse than useless as provenance.
    origin = str(model) if model is not None else \
        f'<preloaded {type(model_obj).__name__}>'
    cap = Capture(model=origin, tag=tag, prompts=list(prompts), rows=rows,
                  meta={'max_tokens': max_tokens, 'adapter': adapter,
                        'chat': chat,
                        'truncated': sum(r['truncated'] for r in rows),
                        'seconds': round(time.time() - t0, 1)})

    # The curve travels with the capture, so a caller who never touches the CLI
    # still has it — including after a save/load round trip weeks later.
    curve = truncation_curve(cap)
    cap.meta['truncation'] = curve.as_dict()
    if not curve.usable:
        # A warning rather than a raise: the capture is real, it cost GPU time,
        # and `compare(drop_truncated=False)` can still use it. But silence here
        # is what let a doomed pair reach compare twice during development, and
        # the Python API is the documented path for offload wrappers and patched
        # runtimes — the callers least likely to be watching a terminal.
        warnings.warn(
            f"only {curve.survivors} of {curve.n_items} items survive at "
            f"max_tokens={curve.cap_used}, and compare() needs {curve.floor}. "
            f"A candidate capture can only truncate more, so this comparison "
            f"cannot succeed. Re-capture with a larger max_tokens (the cap is "
            f"not recoverable after the fact), or pass drop_truncated=False.",
            stacklevel=2)
    return cap


def _pair(reference, candidate, drop_truncated):
    """Load, validate, and return (ref, cand, kept indices).

    Shared by `compare` and `top_movers` so the two can never disagree about
    which items count — a diagnostic that explained items the verdict had
    already discarded would be worse than none.
    """
    ref = reference if isinstance(reference, Capture) else Capture.load(reference)
    cand = candidate if isinstance(candidate, Capture) else Capture.load(candidate)

    if len(ref.rows) != len(cand.rows):
        raise ValueError(
            f"captures have different lengths ({len(ref.rows)} vs "
            f"{len(cand.rows)}); a paired comparison needs the same items")
    if ref.prompts and cand.prompts and ref.prompts != cand.prompts:
        raise ValueError(
            "captures used different prompts; pairing them would compare two "
            "datasets rather than two configurations")

    keep = [i for i in range(len(ref.rows))
            if not (drop_truncated and (ref.rows[i].get('truncated')
                                        or cand.rows[i].get('truncated')))]
    if len(keep) < MIN_PAIRED_ITEMS:
        raise ValueError(
            f"only {len(keep)} items survive truncation filtering; the effect "
            f"size would be noise. Raise max_tokens or pass "
            f"drop_truncated=False.")
    return ref, cand, keep



def compare(reference, candidate, signal=DEFAULT_SIGNAL,
            threshold=DEFAULT_THRESHOLD, one_sided=True,
            drop_truncated=True) -> Result:
    """Paired entropy comparison. Returns a Result; raises on mismatched inputs.

    Accepts `Capture` objects or paths to saved ones.
    """
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose from {SIGNALS}")
    ref, cand, keep = _pair(reference, candidate, drop_truncated)
    dropped = len(ref.rows) - len(keep)

    detail = {}
    for s in SIGNALS:
        x = np.array([ref.rows[i]['ent'][s] for i in keep])
        y = np.array([cand.rows[i]['ent'][s] for i in keep])
        ok = np.isfinite(x) & np.isfinite(y)
        d = y[ok] - x[ok]
        detail[s] = float(d.mean() / (d.std(ddof=1) + 1e-12))

    effect = detail[signal]
    flagged = effect > threshold if one_sided else abs(effect) > threshold
    return Result(flagged=flagged, effect=effect, signal=signal,
                  threshold=threshold, one_sided=one_sided,
                  n_compared=len(keep), n_dropped_truncated=dropped,
                  detail=detail)


def top_movers(reference, candidate, n=5, signal=DEFAULT_SIGNAL,
               drop_truncated=True):
    """The items whose entropy moved most, with the text from both configs.

    `compare` answers "did something break". This answers "show me", which is
    the next question every time and the one the aggregate cannot address:
    d_z is ordinal rather than proportional (FINDINGS F14d), so the per-item
    view is how a reader forms their own judgement of severity.

    Everything needed is already recorded — no second capture, no model.
    """
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose from {SIGNALS}")
    ref, cand, keep = _pair(reference, candidate, drop_truncated)
    rows = []
    for i in keep:
        a = ref.rows[i]['ent'][signal]
        b = cand.rows[i]['ent'][signal]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        rows.append({
            'i': i,
            'delta': float(b - a),
            'ref': float(a),
            'cand': float(b),
            'prompt': ref.prompts[i] if i < len(ref.prompts) else '',
            'ref_text': ref.rows[i].get('text', ''),
            'cand_text': cand.rows[i].get('text', ''),
        })
    rows.sort(key=lambda r: r['delta'], reverse=True)
    return rows[:n]
