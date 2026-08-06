"""The mechanism meter, on ordinary text: is this token recalled or computed?

R3 established that routing at the answer position separates retrieval from
computation at 0.982, replicated exactly on a second architecture with the
answer-form, prompt-length and digit-count confounds all closed. That result
lives entirely inside a probe harness. This takes it to running text, which is
the difference between a finding and an instrument.

HOW IT WORKS. A naive-Bayes profile over per-layer expert selection is fitted on
the R3 `computation` trace — where the two classes emit the SAME answer token
from an IDENTICAL context, so the profile cannot have keyed on answer form. It
is then applied at every token position of arbitrary text, giving a
log-likelihood ratio per token: positive means the routing at that step looks
like retrieval, negative like computation. A logistic fitted on the training
LLRs turns that into a calibrated probability rather than an arbitrary scale.

THE PROFILE MUST COME FROM THE MODEL BEING ANNOTATED. R7 found the *effect*
transfers across architectures (0.982 on both qwen and gemma) while the learned
*decision boundary* does not (0.979 -> 0.696 cross-suite on gemma). A profile
fitted on one model and applied to another measures nothing, so the model tag is
checked and a mismatch is refused rather than warned about.

WHAT THIS CANNOT TELL YOU YET. The probes are chat-wrapped questions; running
text is not. That distribution shift is unmeasured, and it is the whole risk of
this step — so `--selftest` re-scores the probe answers as RAW continuations
rather than chat turns, isolating the format shift from everything else. A meter
that survives the format change is worth pointing at real text; one that does
not is a harness artefact.

Needs mlx-lm.

Usage:
  python3 -m knowledge.annotate --selftest
  python3 -m knowledge.annotate --text "The capital of France is Paris, and 12 times 12 is 144."
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import traces
from .meter import load as load_trace, featurise, counts, OUT

DEFAULT_TRACE = OUT / 'probe_gate.computation.qwen36-35b-a3b-4bit-g64.jsonl.gz'
POS, NEG = 'retrieved', 'computed'


def build_profile(trace_path, top_k=8, alpha=1.0):
    """Fit the naive-Bayes profile and a 1-D calibrator on the R3 trace."""
    meta, recs = load_trace(trace_path)
    sub = [r for r in recs if r['correct'] and r['cls'] in (POS, NEG)]
    X = featurise(sub, meta['n_layers'], top_k, 'answer')
    y = np.array([1 if r['cls'] == POS else 0 for r in sub])
    C = counts(X, y, meta['n_experts'])
    tot = C.sum(axis=2, keepdims=True)
    logp = np.log((C + alpha) / (tot + alpha * C.shape[2]))

    llr = np.array([score_one(logp, X[i]) for i in range(len(X))])
    # logistic calibration: LLR is unbounded and scale-free, so a raw number is
    # not comparable across models or text lengths
    w, b = 0.0, 0.0
    s = (llr - llr.mean()) / (llr.std() + 1e-9)
    for _ in range(2000):
        p = 1 / (1 + np.exp(-np.clip(w * s + b, -30, 30)))
        w -= 0.1 * ((p - y) * s).mean()
        b -= 0.1 * (p - y).mean()
    # The two classes separate PERFECTLY on the training trace (computed max
    # -115, retrieved min -20), so the logistic saturates and its probabilities
    # are overconfident anywhere the true mechanism is mixed — which is most of
    # running text. The empty gap between the classes is therefore kept and used
    # for the display bands: it is grounded in observed data rather than in an
    # extrapolated sigmoid, and it makes "uncommitted" mean something honest.
    return {'logp': logp, 'mu': float(llr.mean()), 'sd': float(llr.std()),
            'neg_max': float(llr[y == 0].max()),
            'pos_min': float(llr[y == 1].min()),
            'w': float(w), 'b': float(b), 'model': meta['model'],
            'n_layers': meta['n_layers'], 'n_experts': meta['n_experts'],
            'top_k': top_k, 'train_auc': _auc(llr[y == 1], llr[y == 0])}


def _auc(pos, neg):
    from .detect import auc
    return round(auc(pos, neg), 4)


def score_one(logp, feats):
    """LLR for one position. `feats` is (n_layers, top_k) expert ids."""
    L = feats.shape[0]
    idx = np.arange(L)[:, None]
    return float(logp[1][idx, feats].sum() - logp[0][idx, feats].sum())


def calibrate(prof, llr):
    z = (np.asarray(llr) - prof['mu']) / (prof['sd'] + 1e-9)
    return 1 / (1 + np.exp(-np.clip(prof['w'] * z + prof['b'], -30, 30)))


class PosTap:
    """Capture per-POSITION routing, not aggregated counts.

    `membership.py` histograms a whole document; here every token needs its own
    routing vector, because per-token output is the entire point.
    """

    def __init__(self, inner, idx, sink, top_k):
        self.inner, self.idx, self.sink, self.top_k = inner, idx, sink, top_k

    def __call__(self, x, *a, **kw):
        out = self.inner(x, *a, **kw)
        if self.sink.get('on'):
            import mlx.core as mx
            from .seam import gate_output
            ranks, _s, _k = gate_output(out, self.top_k)
            mx.eval(ranks)
            self.sink['rows'][self.idx] = np.asarray(
                ranks.reshape(-1, ranks.shape[-1]).tolist(),
                dtype=np.int64)[:, :self.top_k]
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


def position_feats(model, tok, text, top_k, n_layers):
    """(T, n_layers, top_k) expert ids, one row per token position."""
    import mlx.core as mx
    from .seam import find_gates

    gates = find_gates(model)
    sink = {'on': False, 'rows': {}}
    restore = []
    for li, holder, name, gate in gates:
        setattr(holder, name, PosTap(gate, li, sink, top_k))
        restore.append((holder, name, gate))
    try:
        ids = tok.encode(text)
        sink['rows'] = {}
        sink['on'] = True
        model(mx.array([ids]))
        sink['on'] = False
    finally:
        for holder, name, gate in restore:
            setattr(holder, name, gate)
    return np.stack([sink['rows'][l] for l in range(n_layers)], axis=1), ids


def annotate(model, tok, text, prof):
    """[(token_text, probability_of_retrieval, llr)] per scored position."""
    import mlx.core as mx
    from .seam import find_gates

    gates = find_gates(model)
    sink = {'on': False, 'rows': {}}
    restore = []
    for li, holder, name, gate in gates:
        setattr(holder, name, PosTap(gate, li, sink, prof['top_k']))
        restore.append((holder, name, gate))
    try:
        ids = tok.encode(text)
        sink['rows'] = {}
        sink['on'] = True
        model(mx.array([ids]))
        sink['on'] = False
    finally:
        for holder, name, gate in restore:
            setattr(holder, name, gate)

    L = prof['n_layers']
    T = sink['rows'][0].shape[0]
    feats = np.stack([sink['rows'][l] for l in range(L)], axis=1)  # [T, L, k]
    llr = np.array([score_one(prof['logp'], feats[t]) for t in range(T)])
    p = calibrate(prof, llr)
    # position t's routing decides token t+1, so the label belongs to the NEXT
    # token — off-by-one here would shift every annotation by one word
    toks = [tok.decode([i]) for i in ids]
    return [(toks[t + 1], float(p[t]), float(llr[t]))
            for t in range(min(T, len(toks) - 1))]


def render(pairs, prof, width=88, mode='relative'):
    """Mark tokens by WITHIN-DOCUMENT rank, not by the training thresholds.

    The absolute bands do not survive the move to free text. On running prose
    every LLR lands above the training `computed` band, and the highest scores
    go to whitespace and prepositions — the profile was fitted only on ANSWER
    positions, so determiners and punctuation are out of distribution and the
    binary has no "neither" class to put them in.

    The ORDERING does survive. On a passage alternating facts and arithmetic,
    the four lowest-ranked tokens out of 28 were exactly the four arithmetic
    answer tokens, with the two factual answers at ranks 25 and 27. So the meter
    is a RELATIVE instrument: it ranks tokens within a passage, and cannot say
    "this whole passage is retrieval". Absolute mode is kept for probe-shaped
    input, where the training bands do apply.
    """
    llrs = np.array([l for _t, _p, l in pairs])
    z = (llrs - llrs.mean()) / (llrs.std() + 1e-9)
    line, out = '', []
    for i, (t, _p, llr) in enumerate(pairs):
        if mode == 'relative':
            mark = 'R' if z[i] >= 0.5 else ('c' if z[i] <= -0.5 else '·')
        else:
            mark = ('R' if llr >= prof['pos_min']
                    else ('c' if llr <= prof['neg_max'] else '·'))
        cell = f"{t.strip() or '_'}{mark}"
        if len(line) + len(cell) + 1 > width:
            out.append(line)
            line = ''
        line += cell + ' '
    out.append(line)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--trace', default=str(DEFAULT_TRACE))
    ap.add_argument('--text', default=None)
    ap.add_argument('--selftest', action='store_true',
                    help='re-score R3 answers as RAW continuations, isolating '
                         'the chat->free-text format shift')
    ap.add_argument('--limit-gb', type=float, default=40.0)
    ap.add_argument('--mode', default='relative',
                    choices=('relative', 'absolute'),
                    help='relative = within-document ranking (free text); '
                         'absolute = training bands (probe-shaped input only)')
    a = ap.parse_args()

    prof = build_profile(a.trace)
    print(f"profile from {Path(a.trace).name}")
    print(f"  model {prof['model'].split('/')[-1]} · {prof['n_layers']} layers "
          f"· in-sample AUC {prof['train_auc']}")
    if Path(a.model).name not in prof['model']:
        raise SystemExit(
            f"profile was fitted on {prof['model']} but --model is {a.model}. "
            f"R7 showed the decision boundary does not transfer across models; "
            f"refusing rather than reporting a meaningless number.")

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load
    print(f"loading …", flush=True)
    model, tok = load(a.model)

    if a.selftest:
        from .probes import all_probes
        from .meter import counts as nb_counts
        ps = all_probes('computation')
        feats, ys, facts = [], [], []
        for i, pr in enumerate(ps):
            # raw continuation, no chat wrapper — the format the profile has
            # never seen
            F, ids = position_feats(model, tok, pr['stem'] + pr['answer'],
                                    prof['top_k'], prof['n_layers'])
            n_prompt = len(tok.encode(pr['stem']))
            if n_prompt - 1 >= F.shape[0]:
                continue
            feats.append(F[n_prompt - 1])
            ys.append(1 if pr['cls'] == POS else 0)
            facts.append(pr['fact_id'])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ps)}", flush=True)
        X = np.stack(feats)
        y = np.array(ys)
        fid = np.array(facts)

        # LEAVE ONE FACT OUT. The profile was fitted on chat versions of these
        # same 38 facts, and R5 showed routing encodes fact identity — so a
        # profile that has seen fact F could recognize F in a new format
        # without reading mechanism at all. Refitting without the held-out
        # fact is the only way this measures the format shift rather than
        # memorised facts.
        C_all = nb_counts(X, y, prof['n_experts'])
        llr = np.zeros(len(y))
        for f in np.unique(fid):
            te = fid == f
            C = C_all - nb_counts(X[te], y[te], prof['n_experts'])
            if (C.sum(axis=(1, 2)) <= 0).any():
                C = C_all
            lg = np.log((C + 1.0) / (C.sum(2, keepdims=True) + 1.0 * C.shape[2]))
            for j in np.where(te)[0]:
                llr[j] = score_one(lg, X[j])

        ret, cmp_ = llr[y == 1], llr[y == 0]
        print(f"\n  RAW-continuation selftest, leave-one-fact-out (n={len(y)})")
        print(f"    retrieved  mean LLR {ret.mean():9.1f}")
        print(f"    computed   mean LLR {cmp_.mean():9.1f}")
        print(f"    AUC {_auc(ret, cmp_):.3f}"
              f"   (chat-format, same estimator, was 0.982)")
        print(f"\n  This is the chat->free-text format shift with fact identity"
              f"\n  held out, so it cannot be the profile recognizing the fact.")
        return

    text = a.text or ("The capital of France is Paris. "
                      "Twelve times twelve is 144.")
    pairs = annotate(model, tok, text, prof)
    print(f"\n  RELATIVE mode: marks are within-document z-scores of the LLR."
          f"\n  R = z >= +0.5 (retrieval-leaning FOR THIS PASSAGE)"
          f"\n  c = z <= -0.5 (computation-leaning) · = middle"
          f"\n  Absolute training bands do not transfer to free text; see"
          f"\n  render() for why.\n")
    print(render(pairs, prof, mode=a.mode))
    _l = np.array([l for _t, _p, l in pairs])
    if a.mode == 'relative':
        _z = (_l - _l.mean()) / (_l.std() + 1e-9)
        nR, nc = int((_z >= 0.5).sum()), int((_z <= -0.5).sum())
    else:
        nR = int((_l >= prof['pos_min']).sum())
        nc = int((_l <= prof['neg_max']).sum())
    print(f"\n  median LLR {np.median(_l):.1f}   {nR} R / {nc} c / "
          f"{len(pairs)} tokens  [{a.mode} mode]")
    dest = OUT / 'annotate.json'
    dest.write_text(json.dumps(
        {'text': text, 'bands': {'pos_min': prof['pos_min'],
                                 'neg_max': prof['neg_max']},
         'tokens': [{'token': t, 'p_retrieval': pp, 'llr': l}
                    for t, pp, l in pairs]}, indent=1))
    print(f"  → {dest}")


if __name__ == '__main__':
    main()
