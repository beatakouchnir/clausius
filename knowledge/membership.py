"""Extract every membership signal family from one model, over the corpus.

Four families are compared on identical documents and identical splits, because
the point of the benchmark is a fair head-to-head rather than another isolated
number:

  blind        bag-of-words on the text, never queries the model. Must sit at
               chance — that is the benchmark's certification, not a baseline
               to beat. (`corpus.blind_baseline`)
  output       per-token log-probabilities -> perplexity, Min-K% Prob, mean
               predictive entropy. The standard black-box family, and the one
               that collapses to ~0.50 at document level in the literature.
  hidden       per-layer representation statistics — norm scale, norm spread,
               and cosine stability against the previous layer. The family
               "Neural Breadcrumbs" reports 0.85 AUC with, against ~0.50 for
               perplexity, on benchmarks whose construction is disputed.
  routing      per-layer expert usage histogram. The observable this project
               brought, and as far as the literature search went, unused for
               membership.

ATTENTION IS NOT COVERED. The published hidden-state work uses attention
entropy and concentration alongside hidden states; MLX does not expose
attention weights without forking the attention implementation. So our `hidden`
arm is a subset of theirs, and if it underperforms their reported numbers that
is a candidate reason rather than a refutation.

A REFERENCE PASS over the BASE model is captured too. Comparing a signal
against its base-model value is the classic reference-model attack and a
strictly stronger setting — it needs the un-finetuned weights, which a real
auditor may not have. Both settings are reported rather than conflated:
single-model (what an auditor can actually do) and reference-based (an upper
bound on what the signal contains).

Needs mlx-lm. Runs under an explicit memory ceiling; see `finetune.py` for why.

Usage:
  python3 -m knowledge.membership --out records/corpus/sig.base.npz
  python3 -m knowledge.membership --adapter records/corpus/arms/adapter-router \
      --out records/corpus/sig.router.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import traces

CORPUS = Path(__file__).resolve().parent.parent / 'records' / 'corpus'


class LayerTap:
    """Wrap a decoder layer, keep per-layer representation statistics.

    Instance-level wrapping again: patching the layer CLASS would tap every
    layer through one counter and lose the per-layer resolution that is the
    whole point.
    """

    def __init__(self, inner, idx, sink):
        self.inner, self.idx, self.sink = inner, idx, sink

    def __call__(self, x, *a, **kw):
        out = self.inner(x, *a, **kw)
        if self.sink.get('on'):
            import mlx.core as mx
            h = out[0] if isinstance(out, tuple) else out
            hf = h.astype(mx.float32)[0]                 # [T, D]
            n = mx.sqrt(mx.sum(hf * hf, axis=-1)) + 1e-6
            prev = self.sink.get('prev')
            if prev is not None and prev.shape == hf.shape:
                cos = mx.sum(hf * prev, axis=-1) / (
                    n * (mx.sqrt(mx.sum(prev * prev, axis=-1)) + 1e-6))
                c = float(mx.mean(cos))
            else:
                c = 0.0
            self.sink['prev'] = hf
            self.sink['hidden'][self.idx] = (
                float(mx.mean(n)), float(mx.std(n)), c)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


class RouteTap:
    def __init__(self, inner, idx, sink, n_exp, top_k):
        self.inner, self.idx, self.sink = inner, idx, sink
        self.n_exp, self.top_k = n_exp, top_k

    def __call__(self, x, *a, **kw):
        out = self.inner(x, *a, **kw)
        if self.sink.get('on'):
            import mlx.core as mx
            from .seam import gate_output
            ranks, _s, _k = gate_output(out, self.top_k)
            mx.eval(ranks)
            flat = np.asarray(ranks.reshape(-1, ranks.shape[-1]).tolist(),
                              dtype=np.int64)[:, :self.top_k].ravel()
            np.add.at(self.sink['routing'][self.idx], flat, 1)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


def extract(model_path, adapter_path, docs, top_k=8, limit_gb=95.0):
    import mlx.core as mx
    mx.set_memory_limit(int(limit_gb * 1024 ** 3))
    from mlx_lm import load
    from .seam import find_gates, find_layers, describe

    print(f"loading {Path(model_path).name}"
          f"{' + ' + Path(adapter_path).name if adapter_path else ''} …",
          flush=True)
    model, tok = load(model_path, adapter_path=adapter_path)
    n_moe, n_exp, _tk = describe(model)
    layers = find_layers(model)
    gates = find_gates(model)

    sink = {'on': False, 'prev': None,
            'hidden': np.zeros((len(layers), 3), dtype=np.float64),
            'routing': np.zeros((n_moe, n_exp), dtype=np.int32)}

    restore = []
    for i, layer in enumerate(layers):
        holder_list = layers
        holder_list[i] = LayerTap(layer, i, sink)
        restore.append(('layer', i, layer))
    for li, holder, name, gate in gates:
        setattr(holder, name, RouteTap(gate, li, sink, n_exp, top_k))
        restore.append(('gate', (holder, name), gate))

    n = len(docs)
    out = {
        'routing': np.zeros((n, n_moe, n_exp), dtype=np.int16),
        'hidden': np.zeros((n, len(layers), 3), dtype=np.float32),
        'ppl': np.zeros(n), 'mink': np.zeros(n), 'ent': np.zeros(n),
        'ntok': np.zeros(n, dtype=np.int32),
    }
    try:
        for i, d in enumerate(docs):
            ids = mx.array([tok.encode(d['text'])])
            sink['prev'] = None
            sink['hidden'][:] = 0
            sink['routing'][:] = 0
            sink['on'] = True
            logits = model(ids[:, :-1]).astype(mx.float32)
            sink['on'] = False

            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = ids[:, 1:]
            tok_lp = mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]
            v = np.asarray(tok_lp.tolist(), dtype=np.float64)
            probs = mx.exp(lp)
            ent = float(mx.mean(-mx.sum(probs * lp, axis=-1)))

            out['ppl'][i] = -v.mean()
            # Min-K% Prob: mean log-prob of the K% least likely tokens. Seen
            # text is hypothesised to lack very-low-probability outliers.
            k = max(1, int(0.2 * len(v)))
            out['mink'][i] = np.sort(v)[:k].mean()
            out['ent'][i] = ent
            out['ntok'][i] = len(v)
            out['routing'][i] = sink['routing']
            out['hidden'][i] = sink['hidden']
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    finally:
        for kind, ref, orig in restore:
            if kind == 'layer':
                layers[ref] = orig
            else:
                setattr(ref[0], ref[1], orig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--limit-gb', type=float, default=95.0)
    ap.add_argument('--limit', type=int, default=0,
                    help='first N documents only, for smoke-testing the path')
    a = ap.parse_args()

    docs = json.loads((CORPUS / 'manifest.json').read_text())
    if a.limit:
        docs = docs[:a.limit]
    res = extract(a.model, a.adapter, docs, a.top_k, a.limit_gb)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **res)
    print(f"\n  {len(docs)} documents → {a.out}")


if __name__ == '__main__':
    main()
