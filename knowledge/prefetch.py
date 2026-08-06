"""R11 — can early-layer routing prefetch the late-layer experts?

The one remaining path to the original run-large-models-locally goal. W4's
offload runtime pays for **miss rate**, not for extraction, and Belady/OPT beats
LRU by +0.173 hit rate at 2 GB precisely because OPT can see the future
(BENCHMARKS §9). Any cheap proxy for the future is worth real memory.

R9c supplies a candidate. Fact-specific routing is concentrated in the **late**
layers, and it is stable across paraphrases of the same fact — so by the time a
forward pass has finished the early layers, the fact may already be determined,
and with it which late-layer experts are about to fire. A forward pass reaches
layer 0 long before layer 28, so a correct prediction there buys the entire
early-layer compute time as a fetch window. That is *within-token* prefetch,
which is a different and much easier target than predicting the next token.

THE TEST. From the experts selected in layers [0, split), predict the top-8 that
will be selected in each layer of [split, n_layers). Three baselines, all of
which a real runtime already has for free:

  frequency   the globally most-used experts in that layer. What a static
              resident set would hold.
  previous    the same layer's selection for the PREVIOUS token. This is the
              temporal locality LRU exploits, and it is the number to beat —
              beating `frequency` alone would prove nothing.
  oracle-ish  the same layer's selection for the previous token of the SAME
              domain, i.e. a topic-aware upper bound on recency.

The predictor is deliberately trivial — a co-occurrence count from early-layer
expert to late-layer expert, fitted on a held-out half of the passes. If a
count table beats recency, the signal is real and a better model would only
help; if it does not, no amount of modeling rescues it.

HONEST CEILING, stated before running. Prefetch can only help where LRU misses.
the offload benchmarks measured OPT − LRU at +0.173 hit rate at 2 GB and +0.048 at 8 GB, so
that gap is the whole prize and this cannot exceed it.

Stdlib + numpy. No model, no GPU.

Usage:
  python3 -m knowledge.prefetch --split 28
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT


def load_stream(name='qwen', kind='expert'):
    """[(domain, pass_index, {layer: (experts...)})] in trace order."""
    src = traces.SOURCES[name]
    path = traces.records_dir() / (src['gate'] if kind == 'gate'
                                   else src['trace'])
    meta, idx = traces.load(path, kind)
    stream = []
    for dom in sorted(idx):
        for pi, prompt in enumerate(idx[dom]):
            n_tok = len(prompt[0])
            for t in range(n_tok):
                stream.append((dom, pi, {l: prompt[l][t]
                                         for l in range(meta['n_layers'])}))
    return meta, stream


def topk_from_counts(counter, k):
    return [e for e, _ in sorted(counter.items(),
                                 key=lambda kv: (-kv[1], kv[0]))[:k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='qwen', choices=sorted(traces.SOURCES))
    ap.add_argument('--split', type=int, default=28,
                    help='layers [0,split) predict layers [split,n)')
    ap.add_argument('--k', type=int, default=8, help='experts to prefetch')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    meta, stream = load_stream(a.model)
    L, E = meta['n_layers'], meta['n_experts']
    print(f"{meta['model'].split('/')[-1]} · {L} layers · {E} experts · "
          f"{len(stream)} decode tokens")
    print(f"predicting layers {a.split}-{L - 1} from layers 0-{a.split - 1}\n")

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(stream))
    half = len(order) // 2
    fit_idx, ev_idx = set(order[:half].tolist()), order[half:].tolist()

    late = list(range(a.split, L))
    # co-occurrence: early expert -> late expert, per late layer
    co = {l: defaultdict(Counter) for l in late}
    freq = {l: Counter() for l in late}
    for i in fit_idx:
        _dom, _pi, rows = stream[i]
        early = [e for l in range(a.split) for e in rows[l]]
        for l in late:
            freq[l].update(rows[l])
            for e in rows[l]:
                for x in early:
                    co[l][x][e] += 1

    freq_top = {l: topk_from_counts(freq[l], a.k) for l in late}

    # previous-token selection, per layer, tracked in trace order
    prev_tok, prev_dom = {}, {}
    prev_at = {}
    prev_dom_at = {}
    last_seen, last_dom_seen = {}, {}
    for i, (dom, _pi, rows) in enumerate(stream):
        prev_at[i] = {l: last_seen.get(l) for l in late}
        prev_dom_at[i] = {l: last_dom_seen.get((dom, l)) for l in late}
        for l in late:
            last_seen[l] = rows[l]
            last_dom_seen[(dom, l)] = rows[l]

    # A HYBRID at double budget. "Beat recency outright" is a harsh bar; the
    # question a runtime actually asks is whether the early-layer signal adds
    # anything ON TOP of what recency already gives. Both hybrids get 2k slots:
    # the k from recency plus k from either the early-layer predictor or plain
    # frequency. If cooc-fill does not beat freq-fill, the signal contributes
    # nothing a static resident set would not.
    hits = {m: 0 for m in ('cooc', 'frequency', 'previous', 'prev_domain',
                           'prev+cooc', 'prev+freq', 'prev+random')}
    total = 0
    for i in ev_idx:
        _dom, _pi, rows = stream[i]
        early = [e for l in range(a.split) for e in rows[l]]
        for l in late:
            truth = set(rows[l])
            total += len(truth)

            sc = Counter()
            for x in early:
                sc.update(co[l].get(x, {}))
            pred = topk_from_counts(sc, a.k) if sc else freq_top[l]
            hits['cooc'] += len(truth & set(pred))
            hits['frequency'] += len(truth & set(freq_top[l]))
            p = prev_at[i][l]
            hits['previous'] += len(truth & set(p)) if p else 0
            pd = prev_dom_at[i][l]
            hits['prev_domain'] += len(truth & set(pd)) if pd else 0

            base = set(p) if p else set()
            fill_c = [e for e in topk_from_counts(sc, 2 * a.k)
                      if e not in base][:a.k] if sc else []
            fill_f = [e for e in topk_from_counts(freq[l], 2 * a.k)
                      if e not in base][:a.k]
            # RANDOM FILL is the control that decides whether the hybrid lift
            # is signal or merely a bigger, more diverse budget. `frequency`
            # fill concentrates on hot experts and may overlap `base` in spirit;
            # random fill spreads out. If cooc-fill does not beat random-fill,
            # the early-layer signal contributes nothing at all.
            fill_r = [int(x) for x in rng.choice(E, 3 * a.k, replace=False)
                      if x not in base][:a.k]
            hits['prev+cooc'] += len(truth & (base | set(fill_c)))
            hits['prev+freq'] += len(truth & (base | set(fill_f)))
            hits['prev+random'] += len(truth & (base | set(fill_r)))

    print(f"  {'policy':14s} {'recall@8':>9s}   (fraction of the experts that "
          f"actually fired, prefetched)")
    res = {}
    for m in ('frequency', 'previous', 'prev_domain', 'cooc',
              'prev+random', 'prev+freq', 'prev+cooc'):
        r = hits[m] / total
        res[m] = round(r, 4)
        note = {'cooc': '  <-- early-layer signal alone (budget k)',
                'prev+random': '  <-- budget 2k, random fill (control)',
                'prev+freq': '  <-- budget 2k, popularity fill',
                'prev+cooc': '  <-- budget 2k, early-layer fill'}.get(m, '')
        print(f"  {m:14s} {r:9.4f}{note}")

    best_base = max(res['frequency'], res['previous'], res['prev_domain'])
    lift = res['cooc'] - best_base
    hyb = res['prev+cooc'] - max(res['prev+freq'], res['prev+random'])
    print(f"\n  alone:  best baseline {best_base:.4f} · early-layer "
          f"{res['cooc']:.4f} · lift {lift:+.4f}")
    print(f"  on top: best fill control "
          f"{max(res['prev+freq'], res['prev+random']):.4f} · prev+cooc "
          f"{res['prev+cooc']:.4f} · lift {hyb:+.4f}")
    print(f"\n  Context: the offload benchmarks measured the whole OPT-minus-LRU headroom at")
    print(f"  +0.173 hit rate at 2 GB. A lift far below that is not a lever.")
    print(f"  A real gain needs the early-layer predictor to beat RECENCY,")
    print(f"  which is what LRU already exploits for free.")

    dest = OUT / f'prefetch.{a.model}.json'
    dest.write_text(json.dumps({'model': a.model, 'split': a.split, 'k': a.k,
                                'n_eval_tokens': len(ev_idx),
                                'recall': res,
                                'lift_over_best_baseline': round(lift, 4),
                                'lift_on_top_of_recency':
                                round(res['prev+cooc'] - res['prev+freq'], 4)},
                               indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
