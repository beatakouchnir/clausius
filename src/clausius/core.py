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

TRUNCATED ITEMS EXCLUDED. A damaged model rambles into the token cap, and those
    items dilute the signal across hundreds of low-information tokens —
    excluding them roughly doubled the measured effect. It also stops the
    detector from riding on generation length, which is a property of the cap
    you chose rather than of the model.

NO SINGLE AGGREGATION WINS. `max` is monotone in damage where `p90` inverts on
    long-output tasks; `p90` gives a larger effect on some others. Every verdict
    agrees under both across all five mechanisms, so all are reported and the
    flag is not sensitive to the choice.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIGNALS = ('max', 'p90', 'mean', 'mean_top10', 'first', 'gen_len')
DEFAULT_SIGNAL = 'max'
DEFAULT_THRESHOLD = 0.3


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
    """
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
    return Capture(model=str(model), tag=tag, prompts=list(prompts), rows=rows,
                   meta={'max_tokens': max_tokens, 'adapter': adapter,
                         'chat': chat,
                         'truncated': sum(r['truncated'] for r in rows),
                         'seconds': round(time.time() - t0, 1)})


def compare(reference, candidate, signal=DEFAULT_SIGNAL,
            threshold=DEFAULT_THRESHOLD, one_sided=True,
            drop_truncated=True) -> Result:
    """Paired entropy comparison. Returns a Result; raises on mismatched inputs.

    Accepts `Capture` objects or paths to saved ones.
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
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; choose from {SIGNALS}")

    keep = [i for i in range(len(ref.rows))
            if not (drop_truncated and (ref.rows[i].get('truncated')
                                        or cand.rows[i].get('truncated')))]
    dropped = len(ref.rows) - len(keep)
    if len(keep) < 20:
        raise ValueError(
            f"only {len(keep)} items survive truncation filtering; the effect "
            f"size would be noise. Raise max_tokens or pass "
            f"drop_truncated=False.")

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
