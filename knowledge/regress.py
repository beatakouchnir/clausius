"""Can we flag a broken deployment config WITHOUT labels?

The product claim this tests: a team ships a compression setting — offload
capacity, top-k, a policy switch — and needs to know whether it silently
degraded the model. Measuring accuracy needs a labeled eval set, which most
teams do not have for their actual task. If the model's own **entropy
distribution** shifts when a config breaks, you can flag the regression with no
labels at all.

This is the one experiment that joins the two halves of the project. Stages A-D
established `p90` entropy predicts *error* at 0.71-0.92 AUC across 5 models. The
frontier chains produced configurations whose damage is INDEPENDENTLY KNOWN:

    qwen  gsm8k  exact  cap-256  0.9397   reference (fully resident)
    qwen  gsm8k  exact  cap-64   0.9447   BENIGN — 3.3x less memory, no damage
    qwen  gsm8k  static cap-128  0.0854   BROKEN
    qwen  gsm8k  static cap-64   0.0151   BROKEN
    qwen  gsm8k  topk=4          0.9239   MILD
    gemma gsm8k  exact  cap-128  0.9146   reference
    gemma gsm8k  exact  cap-32   0.9091   BENIGN — 3x less memory, no damage
    gemma gsm8k  static cap-32   0.0250   BROKEN

So the labels exist for scoring the detector, and the detector never sees them.

THE CONTROL THAT MATTERS is the benign one. "Entropy flags the broken configs"
is worthless if entropy also flags a 3x memory reduction that cost nothing —
that would be a change detector, not a damage detector, and it would fire on
every deployment. The benign arms are what separate those.

THE BASELINE THAT MATTERS is `gen_len`. Broken models ramble and hit the token
cap, so generation length alone may separate the arms — and this project has
been burned by exactly that twice (R2's length artifact; R14, where `gen_len`
scored 0.878 until truncation contamination was removed). If gen_len does as
well as entropy, entropy adds nothing and the honest finding is "count your
tokens".

PAIRED, because the same items run through every config: per-item deltas cancel
the large item-to-item variance in base entropy, which is exactly how a team
would use it — same eval set before and after the change.

Usage:
  python3 -m knowledge.regress capture --model qwen --capacity 256 --policy exact --tag ref
  python3 -m knowledge.regress analyze --model qwen --ref ref
"""
import argparse
import json
import time

import numpy as np

from .cot import variants
from .frontier import FRONT, load_wrapped, mem
from .meter import OUT

REG = OUT / 'regress'
# accuracy measured by the frontier chains, used ONLY to score the detector
KNOWN = {
    ('qwen', 'ref'): 0.9397, ('qwen', 'exact_c64'): 0.9447,
    ('qwen', 'static_c128'): 0.0854, ('qwen', 'static_c64'): 0.0151,
    ('qwen', 'topk4'): 0.9239,
    ('gemma', 'ref'): 0.9146, ('gemma', 'exact_c32'): 0.9091,
    ('gemma', 'static_c32'): 0.0250,
}


def substitute_nonresident(model):
    """Route non-resident experts to a RESIDENT expert, not the zero slot.

    The adversarial construction for "confidently wrong". `policy='static'`
    sends every non-resident expert to a zero-filled slot, so the block's
    contribution vanishes and the model produces obvious garbage — damage that
    presents as UNCERTAINTY, which is the only kind the detector has been shown
    to catch. Substitution instead computes with a real expert's real weights at
    the right magnitude: well-formed arithmetic, wrong answer. If any config can
    be damaged while getting MORE confident, this is the shape it takes.

    It is also a design someone would plausibly ship ("if it's not resident, use
    something"), which is why it belongs in a deployment-config test rather than
    being dismissed as synthetic.

    Implemented by rewriting `cache.map` after wrap(), so the vendored runtime stays
    read-only. Deterministic `e % len(slots)` rather than random: the run has to
    be reproducible for the paired comparison to mean anything.
    """
    import mlx.core as mx
    from ._vendor.moe import find_moe
    n = 0
    for _li, _owner, _attr, glu in find_moe(model):
        c = getattr(glu, 'cache', None)
        if c is None:
            continue
        resident = dict(c.slot_of)
        if not resident:
            continue
        slots = sorted(resident.values())
        m = list(range(c.n_experts))
        for e in range(c.n_experts):
            m[e] = resident[e] if e in resident else slots[e % len(slots)]
        c.map = mx.array(m, dtype=mx.int32)
        n += 1
    if n == 0:
        raise SystemExit("substitute_nonresident found no wrapped MoE layers")
    return n


class _Sharpen:
    """Scale logits before softmax. Greedy argmax is UNCHANGED by construction.

    The specificity control the benign arms never provided: every previous
    "ok" config left accuracy and entropy both flat, so none of them could
    distinguish "detects damage" from "detects entropy change". Under greedy
    decoding this changes entropy sharply and accuracy not at all, so if the
    detector fires here it is measuring confidence, not damage.

    USED AS the model, not assigned onto it. `model.__call__ = wrapper` does not
    intercept `model(x)` — Python resolves the call on the type — which is the
    same mistake `frontier._TopK` documents, made again here one hour later and
    caught before it ran. A proxy whose own class defines `__call__`, with
    `__getattr__` forwarding everything mlx_lm.generate needs (layers, caches,
    parameters), is what actually intercepts.
    """

    def __init__(self, inner, alpha):
        self.inner, self.alpha = inner, alpha

    def __call__(self, *a, **kw):
        return self.inner(*a, **kw) * self.alpha

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


def cmd_capture(a):
    import mlx.core as mx
    pins = None
    if a.policy == 'static':
        hot = FRONT / f'hot.{a.model}.json'
        h = json.loads(hot.read_text())['hot_by_layer']
        pins = {int(k): v[:a.capacity] for k, v in h.items()}
    model, tok, _ = load_wrapped(a.model, a.capacity, a.policy, pins, a.topk,
                                 adapter=a.adapter, model_path=a.model_path)
    if a.substitute:
        print(f"  substituted non-resident experts on "
              f"{substitute_nonresident(model)} layers", flush=True)
    if a.sharpen != 1.0:
        model = _Sharpen(model, a.sharpen)
        print(f"  logits scaled x{a.sharpen} (greedy argmax unchanged)",
              flush=True)

    from mlx_lm import generate
    from .popqa import task_suite
    from .stage_a import CAPS, load_task, score_item
    suite = task_suite()
    items = load_task(a.task, a.n, a.seed)
    cap = CAPS.get(a.task, 512)
    REG.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        pr = suite.build_prompt(tok, it, think=False)
        text = generate(model, tok, prompt=pr,
                        max_tokens=it.get('max_tokens', cap), verbose=False)
        ok, abst = score_item(a.task, it, text, suite)
        ids = tok.encode(pr + text)
        n_prompt = len(tok.encode(pr))
        # one teacher-forced pass over prompt+completion for per-token entropy,
        # the same construction stage_a/stage_b use so the numbers stay
        # comparable with the AUC study rather than being a new definition
        lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        ent = np.asarray((-mx.sum(mx.exp(lp) * lp, axis=-1))[0].tolist())
        del lg, lp
        # A config can be damaged badly enough to generate NOTHING — qwen at
        # top-k<=2 emits an immediate stop — and then `ent` has no generated
        # positions at all. `variants` falls back for its `gen` slice but reads
        # `ent[n_prompt-1]` unguarded for `first`, which raised "index 27 out of
        # bounds for axis 0 with size 27" and killed four arms. Clamping keeps
        # the item: an empty generation is a data point about the damage, not a
        # reason to drop the row and bias the sample.
        n_p = min(n_prompt, len(ent))
        rows.append({
            'correct': bool(ok), 'abstained': bool(abst),
            'truncated': (len(ids) - n_prompt) >= cap - 2,
            'empty_gen': (len(ids) - n_prompt) <= 0,
            'ent': variants(ent, n_p, None)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    scored = [r for r in rows if not r['abstained']]
    acc = sum(r['correct'] for r in scored) / max(len(scored), 1)
    dest = REG / f'{a.model}.{a.task}.{a.tag}.json'
    dest.write_text(json.dumps({
        'model': a.model, 'task': a.task, 'tag': a.tag,
        'capacity': a.capacity, 'policy': a.policy, 'topk': a.topk,
        'substitute': a.substitute, 'sharpen': a.sharpen,
        'adapter': a.adapter, 'model_path': a.model_path,
        'accuracy': round(acc, 4), 'n': len(rows),
        'trunc_rate': round(sum(r['truncated'] for r in rows) / len(rows), 4),
        'seconds': round(time.time() - t0, 1),
        'active_gb': mem()[0], 'rows': rows}))
    print(f"\n  {a.tag}: acc {acc:.4f} · trunc {sum(r['truncated'] for r in rows)}"
          f"/{len(rows)} · {time.time() - t0:.0f}s → {dest}")


def cmd_analyse(a):
    import glob
    files = sorted(glob.glob(str(REG / f'{a.model}.{a.task}.*.json')))
    got = {}
    for f in files:
        d = json.loads(open(f).read())
        got[d['tag']] = d
    if a.ref not in got:
        raise SystemExit(f"reference '{a.ref}' not captured; have {list(got)}")
    ref = got[a.ref]
    SIGNALS = ('p90', 'mean', 'max', 'first', 'mean_top10', 'gen_len')

    print(f"  reference: {a.ref} · acc {ref['accuracy']:.4f} "
          f"(n={ref['n']}, trunc {ref['trunc_rate']:.0%})\n")
    print(f"  {'config':13s} {'acc':>7s} {'Δacc':>7s} {'truth':>7s} | " +
          ' '.join(f'{s:>9s}' for s in SIGNALS))
    print(f"  {'':13s} {'':>7s} {'':>7s} {'':>7s} | " +
          ' '.join(f"{'d_z':>9s}" for _ in SIGNALS))
    out = {}
    for tag, d in sorted(got.items(), key=lambda kv: -kv[1]['accuracy']):
        if tag == a.ref:
            continue
        n = min(len(ref['rows']), len(d['rows']))
        dacc = d['accuracy'] - ref['accuracy']
        truth = 'BROKEN' if dacc < -0.10 else ('mild' if dacc < -0.02 else 'ok')
        cells, rec = [], {}
        for s in SIGNALS:
            x = np.array([r['ent'][s] for r in ref['rows'][:n]])
            y = np.array([r['ent'][s] for r in d['rows'][:n]])
            ok = np.isfinite(x) & np.isfinite(y)
            delta = y[ok] - x[ok]
            # paired effect size: mean shift in units of the shift's own SD.
            # Paired because the same items run through every config, and
            # item-to-item variance in base entropy dwarfs the config effect.
            dz = float(delta.mean() / (delta.std(ddof=1) + 1e-12))
            cells.append(f'{dz:+9.2f}')
            rec[s] = round(dz, 3)
        print(f"  {tag:13s} {d['accuracy']:7.4f} {dacc:+7.4f} {truth:>7s} | "
              + ' '.join(cells))
        out[tag] = {'accuracy': d['accuracy'], 'delta_acc': round(dacc, 4),
                    'truth': truth, 'trunc_rate': d['trunc_rate'], 'd_z': rec}

    print(f"\n  d_z = paired mean shift / SD of the shift. |d_z| > ~0.5 is a "
          f"clear signal.")
    print(f"  A USABLE DETECTOR must be large on BROKEN and ~0 on ok — firing "
          f"on a benign\n  3x memory reduction would make it a change detector, "
          f"not a damage detector.")
    print(f"  gen_len is the baseline to beat: broken models ramble into the "
          f"token cap.")
    dest = REG / f'analysis.{a.model}.{a.task}.json'
    dest.write_text(json.dumps({'reference': a.ref, 'configs': out}, indent=1))
    print(f"\n  → {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--model', default='qwen')
    c.add_argument('--task', default='gsm8k')
    c.add_argument('--capacity', type=int, default=0)
    c.add_argument('--policy', default='exact',
                   choices=('exact', 'static', 'none'))
    c.add_argument('--topk', type=int, default=None)
    c.add_argument('--adapter', default=None,
                   help='LoRA adapter path — the forgetting monitor compares a '
                        'checkpoint against the adapter-free base on held-out '
                        'domains it was never trained on')
    c.add_argument('--model-path', default=None,
                   help='arbitrary checkpoint, e.g. a different quantization '
                        'of the same model')
    c.add_argument('--substitute', action='store_true',
                   help='non-resident experts route to a resident expert '
                        'instead of the zero slot (well-formed but wrong)')
    c.add_argument('--sharpen', type=float, default=1.0,
                   help='scale logits; greedy output is unchanged, entropy is '
                        'not')
    c.add_argument('--tag', required=True)
    c.add_argument('--n', type=int, default=200)
    c.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('analyze')
    s.add_argument('--model', default='qwen')
    s.add_argument('--task', default='gsm8k')
    s.add_argument('--ref', default='ref')
    a = ap.parse_args()
    {'capture': cmd_capture, 'analyze': cmd_analyse}[a.cmd](a)


if __name__ == '__main__':
    main()
