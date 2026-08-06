"""The configuration frontier — when is a compressed big model worse than a small one?

This replaces the accuracy-vs-budget sweep that was scoped and then abandoned,
because reading the offload runtime showed the sweep would have measured a
guaranteed null. `ExpertCache.ensure()` at `policy='exact'` INSTALLS a missing
expert from disk before the gather, so a cache miss costs latency and nothing
else.

MEASURED 2026-07-30, and the first draft of this docstring was wrong about it.
"Capacity-invariant by construction" does not survive contact: cap-full vs
cap-low gives 10/16 identical generations on gemma and 12/16 on qwen, diverging
at tokens 26/132/128/36/21/101 and 18/2/204/20. Never at token 0, so not a
wiring bug. Two capacity-dependent sources, both in the code: `_gather_sort`
sorts on SLOTS (LRU positions, not expert ids), and prefill is CHUNKED below
full capacity. Both change gather_qmm's reduction order, hence rounding.

The accurate claim is therefore:

  policy='exact' is SEMANTICALLY exact -- every routed expert's true weights are
  fetched, nothing dropped, nothing reading garbage -- but NOT numerically
  reproducible across capacities. Deviations are unbiased rounding, not
  information loss.

And the noise floor is tiny: popqa accuracy across a 4x capacity range moves
0.2287/0.2287/0.2258/0.2328/0.2287 on gemma and is IDENTICAL at all four rungs
on qwen. Meanwhile the same capacity twice is byte-identical (16/16), so the
divergence is a deterministic function of cache size -- reproducible, and
therefore validatable, as long as capacity is held fixed.

So there are two families of mechanism, and conflating them is the mistake this
module exists to avoid:

  LOSSLESS   policy='exact'. Memory <-> speed. No accuracy floor at all: you can
             shrink the cache arbitrarily and pay only in latency.
  LOSSY      policy='static' (non-resident experts point at a zero slot and
             their contribution is dropped), top-k reduction, and heterogeneous
             precision. Memory and/or speed <-> ACCURACY. These have a floor.

The product question is therefore not "how far can we compress" but "at what
point is a compressed big model dominated by a small model that fits natively" —
3-axis Pareto dominance over (memory, accuracy, speed). A configuration is dead
if something else is no worse on all three.

Two things are already decided by arithmetic, before any run:

  - Exact offload DOMINATES heterogeneous precision on memory. W3's most
    aggressive mix (1% hot at 4-bit, rest 2-bit) needs 7.19 GB of experts =
    8.54 GB resident; exact offload measured 7.78 GB and reaches 4.3 GB at
    cap-32. Mixed precision keeps everything resident, so it never buys memory —
    it buys *speed*, by dodging the streaming tax, and pays in accuracy.
  - On longbench the offloaded big model is already dominated. gemma-26b at
    7.78 GB scores 0.5262 at 22.79 s/item; e4b at 4.20 GB scores 0.5132 at
    1.55 s/item. 1.3pp (inside noise at n=150) for 14.7x the latency and 1.9x
    the memory. Nobody should run that.

WHY EACH SUBCOMMAND IS A SEPARATE PROCESS. `wrap()` mutates the model in place,
so measuring two capacities in one process means either re-wrapping (the old
cache stays referenced and the memory saving evaporates) or reloading (two
models briefly co-resident). Both were avoidable by just running the shell loop
outside Python, so that is what `run_frontier1_exactness.sh` does. One process,
one model,
one capacity.

Subcommands:
  hot       per-layer hot-expert lists from the saved traces (no GPU)
  gen       greedy-generate at a capacity, dump token ids (for the exactness test)
  compare   diff two `gen` dumps -- this is what validates the lossless claim
  speed     tok/s at a capacity, no scoring
  acc       task accuracy at a capacity/policy

Usage:
  python3 -m knowledge.frontier hot
  python3 -m knowledge.frontier gen  --model gemma --capacity 128 --tag full
  python3 -m knowledge.frontier gen  --model gemma --capacity 32  --tag c32
  python3 -m knowledge.frontier compare --model gemma --a full --b c32
  python3 -m knowledge.frontier speed --model gemma --capacity 64
  python3 -m knowledge.frontier acc   --model gemma --capacity 32 --policy static --task popqa
"""
import argparse
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

from . import traces
from .meter import OUT

# (model path, expert store, n_experts per layer, n_layers)
MODELS = {
    'gemma': (os.environ.get('GEMMA_MOE', 'mlx-community/gemma-4-26b-a4b-it-4bit'),
              'artifacts/expert_store', 128, 30,
              'records/expert_trace.jsonl.gz'),
    'qwen': (traces.artifact('qwen36-35b-a3b-4bit-g64'),
             'artifacts/qwen_expert_store', 256, 40,
             'records/expert_trace.qwen36-35b-a3b-4bit-g64.jsonl.gz'),
    # The small-model reference. NOT MoE (confirmed separately), so it has no
    # store, no experts and no trace -- only `--policy none` works on it. It is
    # here so the frontier's comparison point is measured through THIS scorer
    # rather than quoted across harnesses.
    'e4b': ('mlx-community/gemma-4-e4b-it-4bit', None, 1, 0, None),
}
LIMIT_GB = 60.0          # every config here is <= 20 GB resident; this is a
                         # guard against a runaway, not a working budget. The
                         # project has already had one 128.3 GB near-OOM.
FRONT = OUT / 'frontier'


def repo():
    from . import traces
    return traces.repo()


def _paths(name):
    m, store, n_exp, n_lay, trace = MODELS[name]
    r = repo()
    return (m, None if store is None else str(r / store), n_exp, n_lay,
            None if trace is None else str(r / trace))


def mem():
    """Steady-state and post-wrap-peak memory, in GB.

    `get_peak_memory()` alone was the bug in the first run: it is a high-water
    mark from process start, so it captured the full model load that happens
    BEFORE wrap() frees the expert weights. Every capacity therefore reported
    the same ~13.48 GB (gemma) / 18.17 GB (qwen) and the memory axis of the
    frontier was not measured at all. Resetting after the wrap is what makes the
    number mean "what this configuration costs to run".
    """
    import mlx.core as mx
    return (round(mx.get_active_memory() / 1024 ** 3, 2),
            round(mx.get_peak_memory() / 1024 ** 3, 2))


def load_wrapped(name, capacity, policy, pins=None, topk=None,
                 adapter=None, model_path=None):
    """Load the model, optionally swap in offload caches and/or cut top-k.

    `store_dir` is NOT optional even though wrap() allows None: without it the
    original expert weights stay referenced AS the store, so wrapping ADDS the
    resident slots instead of replacing anything. the vendored own note records
    that this OOM-ed Metal on a k=8 run.

    policy='none' skips wrapping entirely, which is how the non-MoE reference
    model (gemma-e4b) gets measured through THIS scorer rather than being quoted
    from the vendored harness. wrap() would raise on it — correctly, there are no
    MoE layers to find.
    """
    import mlx.core as mx
    mx.set_memory_limit(int(LIMIT_GB * 1024 ** 3))
    from mlx_lm import load

    path, store, n_exp, n_lay, _ = _paths(name)
    if model_path:
        # An arbitrary checkpoint, so the detector can compare two
        # QUANTIZATIONS or two builds of the same model. That is the
        # intervention the equivalence literature says matters most — quantized
        # variants "do not reliably reproduce base-model behavior, even when
        # accuracy or perplexity appears preserved" — and the MODELS dict has no
        # way to name it. No expert store travels with it, so offload is off.
        path, store = model_path, None
        print(f"  loading {path} (override) …", flush=True)
    else:
        print(f"  loading {name} …", flush=True)
    model, tok = (load(path, adapter_path=adapter) if adapter else load(path))
    if adapter:
        print(f"  + adapter {adapter}", flush=True)
    n = 0
    if policy != 'none':
        if store is None:
            raise SystemExit(
                "--model-path carries no expert store; use --policy none")
        from ._vendor.offload_model import wrap
        n = wrap(model, capacity, store_dir=store, policy=policy, pins=pins)
        print(f"  wrapped {n} layers · capacity {capacity}/{n_exp} "
              f"({capacity / n_exp:.0%} of experts) · policy {policy}",
              flush=True)
    else:
        print(f"  no offload wrap (policy=none) — reference model", flush=True)
    if topk is not None:
        n_gate = cut_topk(model, topk)
        print(f"  cut top-k to {topk} on {n_gate} gates", flush=True)
    # measure the CONFIGURATION, not the load
    mx.reset_peak_memory()
    return model, tok, n


def cut_topk(model, k_keep):
    """Consult only the k_keep highest-scoring experts, renormalized.

    This is SwiftLM's `SWIFTLM_TOP_K`, which they ship as a SPEED setting —
    top-k=4 at 5.91 tok/s against top-k=6 at 5.20 — with no accuracy number
    anywhere in their docs. Reducing top-k below the trained value is the
    route-around ablation R9e already measured as damaging, so this arm puts a
    number on a knob a 727-star project hands users unlabeled.

    Renormalization is deliberate and CHARITABLE. Dropping the trailing experts
    without renormalizing would shrink each MoE block's output magnitude and
    make the knob look worse than it is; a model natively trained at k_keep
    would have weights summing to the same total. Masking to -inf before the
    downstream softmax achieves exactly that for both router shapes, which is
    why `seam.gate_output`'s two kinds collapse to one implementation here.
    """
    from .seam import find_gates

    n = 0
    for _li, holder, name, gate in find_gates(model):
        setattr(holder, name, _TopK(gate, k_keep))
        n += 1
    if n == 0:
        raise SystemExit("cut_topk found no gates — see knowledge/seam.py")
    return n


class _TopK:
    """Instance wrapper, following capture.Recorder's pattern for a reason.

    `gate.__call__ = fn` does NOT intercept `holder.gate(x)`: Python resolves
    the call on the TYPE, so an instance attribute is ignored and the cut would
    silently do nothing — a wiring bug that looks exactly like "the knob is
    free". A plain object whose own class defines `__call__`, installed with
    setattr, is what actually intercepts. Patching the class instead would hit
    every nn.Linear in the model, since qwen's gate IS an nn.Linear.
    """

    def __init__(self, inner, k_keep):
        self.inner, self.k_keep = inner, k_keep

    def __call__(self, x, *a, **kw):
        import mlx.core as mx
        out = self.inner(x, *a, **kw)
        if isinstance(out, (tuple, list)):
            # (indices, weights) — and they are NOT rank-ordered. gemma4_text's
            # Router builds them with `mx.argpartition(kth=-top_k)[..., -top_k:]`,
            # which guarantees only membership, not order. The first version of
            # this masked positions 0..k_keep and therefore dropped an ARBITRARY
            # 2 of 8 experts rather than the 2 weakest — random expert ablation
            # wearing top-k reduction's name, and it read as an 89% accuracy
            # collapse on gemma against 1.6% on qwen.
            #
            # The k=8 wiring control could not catch it: with k_keep == top_k the
            # mask keeps every position whatever the order, so the control
            # validated the masking arithmetic and never touched the ranking
            # assumption underneath it. Selecting by VALUE is what makes the two
            # router shapes actually equivalent.
            idx, w = out[0], out[1]
            if self.k_keep >= w.shape[-1]:
                return out
            # `[..., -k:-k+1]` degenerates to `[..., -1:0]` at k==1 — an EMPTY
            # slice, which surfaced as "Shapes (1,31,8) and (1,31,0) cannot be
            # broadcast". The scores branch below was guarded for this and the
            # tuple branch was not.
            kth = (mx.max(w, axis=-1, keepdims=True) if self.k_keep == 1
                   else mx.sort(w, axis=-1)[..., -self.k_keep:-self.k_keep + 1])
            kept = mx.where(w >= kth, w, mx.zeros_like(w))
            tot = mx.sum(w, axis=-1, keepdims=True)
            return idx, kept / mx.maximum(
                mx.sum(kept, axis=-1, keepdims=True), 1e-9) * tot
        # raw scores: -inf everything outside the k_keep best, so whichever
        # top-k/softmax runs downstream behaves like a k_keep-wide model
        if self.k_keep >= out.shape[-1]:
            return out
        kth = mx.sort(out, axis=-1)[..., -self.k_keep:-self.k_keep + 1] \
            if self.k_keep > 1 else mx.max(out, axis=-1, keepdims=True)
        return mx.where(out >= kth, out, mx.array(-1e9, out.dtype))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


def probe_prompts(tok, n=16, cap=256):
    """gsm8k, because exactness needs LONG generations to be a real test.

    A 4-token answer can match by luck; 256 tokens of arithmetic reasoning
    cannot. Divergence compounds, so if the cache perturbs anything at all this
    is where it shows.
    """
    from .popqa import task_suite
    from .stage_a import load_task
    suite = task_suite()
    items = load_task('gsm8k', n, 0)
    return [(suite.build_prompt(tok, it, think=False), cap) for it in items]


def cmd_hot(a):
    """Per-layer expert usage from the saved traces. No GPU, no model.

    `policy='static'` preloads `pins` and never installs anything else, so the
    pins ARE the experiment: pinning experts 0..C-1 (wrap's default) would test
    an arbitrary subset and understate static badly. These lists make the
    comparison "the C most-used experts stay" rather than "the C lowest-numbered
    experts stay".
    """
    FRONT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        _, _, n_exp, n_lay, trace = _paths(name)
        if trace is None or not Path(trace).exists():
            print(f"  {name}: no trace at {trace} — skipped")
            continue
        per_layer = {li: Counter() for li in range(n_lay)}
        n_rec, meta = 0, None
        with gzip.open(trace, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if meta is None and 'schema' in r:
                    meta = r
                    continue
                li = r.get('layer', r.get('l'))
                ids = r.get('experts', r.get('e', r.get('ids')))
                if li is None or ids is None:
                    continue
                flat = ids
                while flat and isinstance(flat[0], list):
                    flat = [x for sub in flat for x in sub]
                per_layer[int(li)].update(int(e) for e in flat)
                n_rec += 1
        if not n_rec:
            print(f"  {name}: trace parsed 0 usable records — schema mismatch; "
                  f"first keys were {list(r)[:8] if n_rec == 0 else ''}")
            continue
        hot = {str(li): [e for e, _ in c.most_common()]
               for li, c in per_layer.items() if c}
        cov = {}
        for frac in (0.05, 0.10, 0.25, 0.50):
            k = max(1, int(round(frac * n_exp)))
            tot = sum(sum(c.values()) for c in per_layer.values())
            got = sum(sum(v for _, v in c.most_common(k))
                      for c in per_layer.values())
            cov[f'{frac:.2f}'] = round(got / max(tot, 1), 4)
        dest = FRONT / f'hot.{name}.json'
        dest.write_text(json.dumps(
            {'model': name, 'n_experts': n_exp, 'n_layers': n_lay,
             'n_records': n_rec, 'decision_share_on_top': cov,
             'hot_by_layer': hot}, indent=1))
        print(f"  {name}: {n_rec} records · decision share on top-k experts "
              f"{cov} → {dest}")


def cmd_gen(a):
    """Greedy-generate at one capacity and dump token ids for later diffing."""
    import mlx.core as mx
    model, tok, _ = load_wrapped(a.model, a.capacity, a.policy, topk=a.topk)
    from mlx_lm import stream_generate
    FRONT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, (pr, cap) in enumerate(probe_prompts(tok, a.n, a.cap)):
        toks = []
        for resp in stream_generate(model, tok, prompt=pr, max_tokens=cap):
            toks.append(int(resp.token))
        rows.append(toks)
        print(f"  {i + 1}/{a.n}  {len(toks)} tok  "
              f"{time.time() - t0:.0f}s", flush=True)
    dest = FRONT / f'gen.{a.model}.{a.tag}.json'
    dest.write_text(json.dumps(
        {'model': a.model, 'capacity': a.capacity, 'policy': a.policy,
         'tag': a.tag, 'n': a.n, 'cap': a.cap,
         'seconds': round(time.time() - t0, 1),
         'topk': a.topk,
         'active_gb': mem()[0], 'peak_gb': mem()[1],
         'tokens': rows}))
    print(f"\n  → {dest}")


def cmd_compare(a):
    """Diff two `gen` dumps. THIS is the test the whole plan rests on.

    If exact-policy offload is lossless, cap-full and cap-low must produce
    IDENTICAL token sequences. If they do not, accuracy is not capacity-
    invariant, the frontier gains an axis, and the sweep I talked the user out
    of has to happen after all. Reporting the first divergence index matters:
    divergence at token 0 is a wiring bug, divergence at token 140 is
    floating-point drift compounding through greedy decoding.
    """
    A = json.loads((FRONT / f'gen.{a.model}.{a.a}.json').read_text())
    B = json.loads((FRONT / f'gen.{a.model}.{a.b}.json').read_text())
    ta, tb = A['tokens'], B['tokens']
    ident, divs = 0, []
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x == y:
            ident += 1
        else:
            j = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q),
                     min(len(x), len(y)))
            divs.append({'item': i, 'first_diff': j,
                         'len_a': len(x), 'len_b': len(y)})
    n = min(len(ta), len(tb))
    print(f"  {A['tag']} (cap {A['capacity']}) vs {B['tag']} "
          f"(cap {B['capacity']}) · policy {A['policy']}/{B['policy']}")
    print(f"  identical sequences: {ident}/{n}")
    if divs:
        print(f"  first-divergence token index: "
              f"{[d['first_diff'] for d in divs][:12]}")
    print(f"  speed {A['seconds']:.0f}s vs {B['seconds']:.0f}s "
          f"({B['seconds'] / max(A['seconds'], 1e-9):.2f}x) · "
          f"peak {A['peak_gb']} vs {B['peak_gb']} GB")
    verdict = ('LOSSLESS — capacity does not change output'
               if ident == n else
               f'NOT lossless — {n - ident}/{n} sequences differ')
    print(f"\n  {verdict}")
    dest = FRONT / f'compare.{a.model}.{a.a}_vs_{a.b}.json'
    dest.write_text(json.dumps(
        {'model': a.model, 'a': A['tag'], 'b': B['tag'],
         'cap_a': A['capacity'], 'cap_b': B['capacity'],
         'identical': ident, 'n': n, 'divergences': divs,
         'sec_a': A['seconds'], 'sec_b': B['seconds'],
         'peak_a': A['peak_gb'], 'peak_b': B['peak_gb'],
         'lossless': ident == n}, indent=1))
    print(f"  → {dest}")


def cmd_speed(a):
    """tok/s at one capacity. No scoring — the exact policy cannot change it.

    Measures the axis the frontier actually needs and that my cost model only
    estimated. Prefill and decode are timed separately because they thrash
    differently: every prefill token routes independently, so prefill touches
    far more distinct experts than decode (the locality record puts coverage at
    0.8743 with prefill against 0.8635 without).
    """
    import mlx.core as mx
    model, tok, _ = load_wrapped(a.model, a.capacity, a.policy, topk=a.topk)
    from mlx_lm import stream_generate
    FRONT.mkdir(parents=True, exist_ok=True)

    # Discarded warmup. The smoke run read 10.16 tok/s at FULL capacity against
    # the previously measured 58 — not a regression, just kernel compilation and a
    # cold cache amortised over only 32 tokens. Timing the first pass would make
    # low capacities look artificially good, because their surcharge is real
    # work while this is one-off overhead.
    wu = probe_prompts(tok, 1, 24)[0]
    for _ in stream_generate(model, tok, prompt=wu[0], max_tokens=wu[1]):
        pass

    n_tok, t_first, t_all = 0, [], time.time()
    for pr, cap in probe_prompts(tok, a.n, a.cap):
        t0 = time.time()
        first = None
        for resp in stream_generate(model, tok, prompt=pr, max_tokens=cap):
            if first is None:
                first = time.time() - t0
            n_tok += 1
        t_first.append(first or 0.0)
    wall = time.time() - t_all
    _, _, n_exp, _, _ = _paths(a.model)
    res = {'model': a.model, 'capacity': a.capacity, 'policy': a.policy,
           'frac_experts': round(a.capacity / n_exp, 4),
           'n_items': a.n, 'gen_tokens': n_tok,
           'seconds': round(wall, 1), 'tok_s': round(n_tok / wall, 2),
           'ttft_mean_s': round(sum(t_first) / len(t_first), 3),
           'topk': a.topk,
           'active_gb': mem()[0], 'peak_gb': mem()[1]}
    print(f"\n  cap {a.capacity}/{n_exp} ({res['frac_experts']:.0%}) · "
          f"{res['tok_s']} tok/s · TTFT {res['ttft_mean_s']}s · "
          f"active {res['active_gb']} GB · peak {res['peak_gb']} GB")
    tk = '' if a.topk is None else f'.k{a.topk}'
    dest = FRONT / f'speed.{a.model}.c{a.capacity}.{a.policy}{tk}.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"  → {dest}")


def cmd_acc(a):
    """Task accuracy at one capacity/policy. Only meaningful for LOSSY policies.

    Run against policy='static' (and later top-k reduction), not 'exact' —
    scoring exact at several capacities burns GPU hours to redraw a flat line.
    Uses external's own loader and scorer so the numbers stay comparable with
    stage_a/stage_d rather than with a re-implementation.
    """
    import mlx.core as mx
    pins = None
    if a.policy == 'static':
        hot = FRONT / f'hot.{a.model}.json'
        if not hot.exists():
            raise SystemExit(
                f"static policy needs hot-expert lists; run "
                f"`python3 -m knowledge.frontier hot` first ({hot} missing). "
                f"Without pins, wrap() preloads experts 0..C-1 — an arbitrary "
                f"subset, which would understate static.")
        h = json.loads(hot.read_text())['hot_by_layer']
        pins = {int(k): v[:a.capacity] for k, v in h.items()}
    model, tok, _ = load_wrapped(a.model, a.capacity, a.policy, pins, a.topk)

    from mlx_lm import generate
    from .popqa import task_suite
    from .stage_a import CAPS, QUANTIZE_TASKS, load_task, score_item
    suite = task_suite()
    items = load_task(a.task, a.n, a.seed)
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        pr = suite.build_prompt(tok, it, think=False)
        text = generate(model, tok, prompt=pr,
                        max_tokens=it.get('max_tokens', CAPS.get(a.task, 512)),
                        verbose=False)
        ok, abst = score_item(a.task, it, text, suite)
        rows.append({'correct': bool(ok), 'abstained': bool(abst)})
        if (i + 1) % 25 == 0:
            acc = sum(r['correct'] for r in rows) / len(rows)
            print(f"  {i + 1}/{len(items)}  acc {acc:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    scored = [r for r in rows if not r['abstained']]
    acc = sum(r['correct'] for r in scored) / max(len(scored), 1)
    _, _, n_exp, _, _ = _paths(a.model)
    res = {'model': a.model, 'task': a.task, 'capacity': a.capacity,
           'policy': a.policy, 'frac_experts': round(a.capacity / n_exp, 4),
           'accuracy': round(acc, 4), 'n_scored': len(scored),
           'n_abstained': len(rows) - len(scored),
           'seconds': round(wall, 1),
           'sec_per_item': round(wall / max(len(rows), 1), 2),
           'topk': a.topk,
           'active_gb': mem()[0], 'peak_gb': mem()[1],
           'per_item': [int(r['correct']) for r in rows]}
    print(f"\n  {a.task} · cap {a.capacity}/{n_exp} · {a.policy} · "
          f"acc {acc:.4f} (n={len(scored)}) · {res['sec_per_item']}s/item · "
          f"active {res['active_gb']} GB")
    tk = '' if a.topk is None else f'.k{a.topk}'
    dest = FRONT / f'acc.{a.model}.{a.task}.c{a.capacity}.{a.policy}{tk}.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"  → {dest}")


def cmd_report(a):
    """Assemble every record into the 3-axis frontier and mark what is dead.

    Dominance, not ranking: configuration X is dead if some Y is no worse on ALL
    of (memory, accuracy, speed) and strictly better on one. That is the question
    a buyer with fixed hardware actually asks, and it is why a table sorted by
    accuracy alone would mislead -- the lossy configurations score badly AND run
    slower, so they lose on axes a single-column ranking never shows.

    Memory is `active_gb` where measured; the first speed sweep recorded a
    high-water mark taken before wrap() freed the expert weights, so those rows
    fall back to capacity arithmetic and are flagged.
    """
    import glob
    rows = []
    for f in glob.glob(str(FRONT / 'acc.*.json')):
        d = json.loads(Path(f).read_text())
        rows.append(d)
    if not rows:
        print("  no acc.* records yet")
        return
    # speed per (model, capacity, policy) from the speed sweep, for rows whose
    # own s/item came from a different task
    spd = {}
    for f in glob.glob(str(FRONT / 'speed.*.json')):
        d = json.loads(Path(f).read_text())
        spd[(d['model'], d['capacity'], d['policy'],
             d.get('topk'))] = d

    by_task = {}
    for r in rows:
        by_task.setdefault(r['task'], []).append(r)

    out = {}
    for task, rs in sorted(by_task.items()):
        print(f"\n=== {task} ===")
        print(f"  {'model':6s} {'policy':7s} {'k':>3s} {'cap':>5s} {'%exp':>5s} "
              f"{'mem GB':>7s} {'acc':>7s} {'s/item':>7s}  verdict")
        pts = []
        for r in rs:
            m = r.get('active_gb')
            s = spd.get((r['model'], r['capacity'], r['policy'], r.get('topk')))
            if m is None and s:
                m = s.get('active_gb')
            est = m is None
            if m is None:
                m = round(r['capacity'] * MODELS[r['model']][3]
                          * (3345408 if r['model'] == 'gemma' else 1769472)
                          / 1024 ** 3 + 1.4, 2) if r['capacity'] else 0.0
            pts.append({'model': r['model'], 'policy': r['policy'],
                        'topk': r.get('topk'), 'cap': r['capacity'],
                        'frac': r['frac_experts'], 'mem': m, 'est': est,
                        'acc': r['accuracy'], 'sec': r['sec_per_item']})
        for x in pts:
            killers = [y for y in pts
                       if y is not x and y['mem'] <= x['mem']
                       and y['acc'] >= x['acc'] and y['sec'] <= x['sec']
                       and (y['mem'] < x['mem'] or y['acc'] > x['acc']
                            or y['sec'] < x['sec'])]
            x['dominated_by'] = [
                f"{k['model']}/{k['policy']}"
                + (f"/k{k['topk']}" if k['topk'] else '')
                + f"/c{k['cap']}" for k in killers][:3]
        for x in sorted(pts, key=lambda z: (-z['acc'], z['mem'])):
            v = ('DEAD — dominated by ' + ', '.join(x['dominated_by'])
                 if x['dominated_by'] else 'on frontier')
            print(f"  {x['model']:6s} {x['policy']:7s} "
                  f"{'-' if x['topk'] is None else x['topk']:>3} "
                  f"{x['cap']:5d} {x['frac']:5.0%} "
                  f"{x['mem']:7.2f}{'*' if x['est'] else ' '} {x['acc']:7.4f} "
                  f"{x['sec']:7.2f}  {v}")
        out[task] = pts
    print("\n  * memory computed from capacity, not measured.")
    print("  DEAD = some configuration is no worse on memory, accuracy AND "
          "speed.")
    dest = FRONT / 'report.json'
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n  → {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, gpu=True):
        p.add_argument('--model', default='gemma', choices=list(MODELS))
        if gpu:
            p.add_argument('--capacity', type=int, default=0,
                           help='per-layer resident slots; ignored when '
                                'policy=none')
            p.add_argument('--policy', default='exact',
                           choices=('exact', 'static', 'none'))
            p.add_argument('--topk', type=int, default=None,
                           help='consult only the N best experts, renormalized '
                                "(SwiftLM's SWIFTLM_TOP_K). None = untouched.")
        return p

    sub.add_parser('hot')
    sub.add_parser('report')
    g = common(sub.add_parser('gen'))
    g.add_argument('--tag', required=True)
    g.add_argument('--n', type=int, default=16)
    g.add_argument('--cap', type=int, default=256)
    c = sub.add_parser('compare')
    c.add_argument('--model', default='gemma', choices=list(MODELS))
    c.add_argument('--a', required=True)
    c.add_argument('--b', required=True)
    s = common(sub.add_parser('speed'))
    s.add_argument('--n', type=int, default=8)
    s.add_argument('--cap', type=int, default=256)
    q = common(sub.add_parser('acc'))
    q.add_argument('--task', default='popqa')
    q.add_argument('--n', type=int, default=200)
    q.add_argument('--seed', type=int, default=0)

    a = ap.parse_args()
    if getattr(a, 'policy', None) not in (None, 'none') and not a.capacity:
        raise SystemExit(f"--capacity is required for policy={a.policy}")
    {'hot': cmd_hot, 'report': cmd_report, 'gen': cmd_gen,
     'compare': cmd_compare, 'speed': cmd_speed, 'acc': cmd_acc}[a.cmd](a)


if __name__ == '__main__':
    main()
