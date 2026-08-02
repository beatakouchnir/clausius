"""R12 — read-only fact identification in the model's own generated text.

THE EXPERIMENT THE PRODUCT DEPENDS ON. R9 and R10 establish a causal fact-level
address, but both work by ABLATION: N destructive forward passes per fact. You
cannot ablate in production, so telemetry has to be read-only.

The read-only result so far is R5 — identify the fact from routing — and R8
showed it collapsing outside countries, beaten by bag-of-words. But that
objection was specific to probes: the prompt named the entity and the relation,
so reading the prompt was near an oracle. **In generated prose nothing names the
fact**, so the baseline that beat us there does not exist here.

THE TASK. The model writes a passage. At a fact-bearing token, read the routing
and identify which of the known facts it is — from routing alone, one forward
pass, no intervention.

THE BASELINES, and each one has to be beaten:

  prior       always guess the most frequent fact. Catches label imbalance.
  position    a nearest-neighbour on (sequence position, token index) only.
              Catches "facts appear in a predictable order in generated text",
              which would let a classifier win without reading routing at all.
  bagofwords  the generated text preceding the target token. In the probe
              setting this was an oracle; here it should be weak, and if it is
              NOT weak then generated text names its own facts more than
              expected and R8's objection has followed us over.

TRAIN AND TEST MUST NOT SHARE A PASSAGE. Two fact tokens from the same generated
passage share context, position and topic. Splitting by token would let the
classifier match the passage rather than the fact, which is the R1 token-vs-
prompt trap in a new costume. Splits are by passage.

Needs mlx-lm for capture; the analysis is numpy.

Usage:
  python3 -m knowledge.readout --capture      # generate + record routing
  python3 -m knowledge.readout                # analyse an existing capture
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT, counts, score
from .probes import GRID2

CAP = OUT / 'readout.capture.json'

# Topic seeds. The same fact must recur across GENUINELY DIFFERENT passages,
# which is what makes a held-out-passage split mean anything.
#
# A first version repeated one prompt three times per entity. Decoding is
# greedy, so all three generations were BYTE-IDENTICAL and the "held-out"
# passage was a copy of a training passage. Every baseline then scored by
# finding its own twin — position hit 1.000 — and the routing number was
# meaningless. Distinct phrasings per repeat are what force real variation.
PHRASINGS = [
    "Write four short factual sentences about {}.",
    "Describe {} in a few factual sentences.",
    "What are some important facts about {}? Answer in plain prose.",
    "Give a brief encyclopedic summary of {}.",
]
SEEDS = [
    ("{}", ['France', 'Japan', 'Brazil', 'Egypt', 'India', 'Poland', 'Kenya',
            'Norway', 'Vietnam', 'Peru', 'Chile', 'Sweden']),
    ("the chemical element {}", [
        'iron', 'gold', 'oxygen', 'helium', 'carbon', 'sodium', 'mercury',
        'nitrogen', 'copper', 'sulfur', 'neon', 'zinc']),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--max-tokens', type=int, default=110)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.capture:
        capture(a)
        return

    if not CAP.exists():
        raise SystemExit(f"no capture at {CAP}; run with --capture first")
    d = json.loads(CAP.read_text())
    rows, L, E = d['rows'], d['n_layers'], d['n_experts']
    print(f"{d['model']} · {len(rows)} fact tokens from "
          f"{len({r['passage'] for r in rows})} passages")

    labels = sorted({r['fact_id'] for r in rows})
    keep = [l for l in labels
            if len({r['passage'] for r in rows if r['fact_id'] == l}) >= 2]
    rows = [r for r in rows if r['fact_id'] in keep]
    labels = sorted(keep)
    print(f"  {len(labels)} facts appearing in >=2 distinct passages "
          f"(needed for a held-out-passage split) · {len(rows)} usable tokens")
    print(f"  chance = {1 / len(labels):.4f}\n")

    y = np.array([labels.index(r['fact_id']) for r in rows])
    pas = np.array([r['passage'] for r in rows])
    X = np.array([r['experts'] for r in rows], dtype=np.int64)  # [N, L, k]

    # leave-one-passage-out
    pred_r = np.zeros(len(y), dtype=np.int64)
    pred_w = np.zeros(len(y), dtype=np.int64)
    pred_p = np.zeros(len(y), dtype=np.int64)
    for pid in np.unique(pas):
        te, tr = pas == pid, pas != pid
        if not te.any() or not tr.any():
            continue
        C = counts(X[tr], y[tr], E, n_cls=len(labels))
        pred_r[te] = score(C, X[te]).argmax(1)

        # bag-of-words over the text BEFORE the target token
        vocab = sorted({w for i in np.where(tr)[0]
                        for w in rows[i]['prefix'].lower().split()})
        vi = {w: j for j, w in enumerate(vocab)}
        W = np.zeros((len(labels), len(vocab)))
        for i in np.where(tr)[0]:
            for w in rows[i]['prefix'].lower().split():
                if w in vi:
                    W[y[i], vi[w]] += 1
        lw = np.log((W + 0.3) / (W.sum(1, keepdims=True) + 0.3 * len(vocab)))
        for i in np.where(te)[0]:
            ws = [vi[w] for w in rows[i]['prefix'].lower().split() if w in vi]
            pred_w[i] = int(lw[:, ws].sum(1).argmax()) if ws else 0

        # position-only nearest neighbour
        ptr = np.array([[rows[i]['rel_pos']] for i in np.where(tr)[0]])
        ytr = y[tr]
        for i in np.where(te)[0]:
            j = int(np.abs(ptr[:, 0] - rows[i]['rel_pos']).argmin())
            pred_p[i] = ytr[j]

    prior = Counter(y.tolist()).most_common(1)[0][0]
    acc = lambda p: float((p == y).mean())  # noqa: E731
    res = {'chance': round(1 / len(labels), 4),
           'prior': round(float((y == prior).mean()), 4),
           'position': round(acc(pred_p), 4),
           'bagofwords': round(acc(pred_w), 4),
           'routing': round(acc(pred_r), 4),
           'n': len(y), 'n_facts': len(labels)}
    print(f"  {'baseline':14s} {'accuracy':>9s}")
    for k in ('chance', 'prior', 'position', 'bagofwords'):
        print(f"  {k:14s} {res[k]:9.4f}")
    print(f"  {'ROUTING':14s} {res['routing']:9.4f}   <-- read-only, one pass")
    best = max(res['prior'], res['position'], res['bagofwords'])
    print(f"\n  best baseline {best:.4f} · routing {res['routing']:.4f} "
          f"· lift {res['routing'] - best:+.4f}")
    res['lift'] = round(res['routing'] - best, 4)

    # per domain, because R8's lesson was that a pooled number can describe no
    # domain in the set
    print(f"\n  {'domain':10s} {'n':>4s} {'facts':>6s} {'chance':>7s} "
          f"{'routing':>8s} {'words':>7s}")
    res['by_domain'] = {}
    for dom in sorted({r['domain'] for r in rows}):
        m = np.array([r['domain'] == dom for r in rows])
        nf = len({r['fact_id'] for r in rows if r['domain'] == dom})
        res['by_domain'][dom] = {
            'n': int(m.sum()), 'n_facts': nf,
            'routing': round(float((pred_r[m] == y[m]).mean()), 4),
            'words': round(float((pred_w[m] == y[m]).mean()), 4)}
        v = res['by_domain'][dom]
        print(f"  {dom:10s} {v['n']:4d} {nf:6d} {1 / max(nf, 1):7.3f} "
              f"{v['routing']:8.3f} {v['words']:7.3f}")

    dest = OUT / 'readout.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  → {dest}")


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate
    from .seam import find_gates, gate_output, describe
    from .generated import scan_facts, find_target, _norm

    print(f"loading …", flush=True)
    model, tok = load(a.model)
    n_moe, E, _tk = describe(model)

    sink = {'on': False, 'rows': {}}
    restore = []
    for li, holder, name, gate in find_gates(model):
        tgt_holder, tgt_name, tgt = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if inner is not None and callable(inner):
            tgt_holder, tgt_name, tgt = gate, 'proj', inner

        class Tap:
            def __init__(self, inner, idx):
                self.inner, self.idx = inner, idx

            def __call__(self, x, *aa, **kw):
                out = self.inner(x, *aa, **kw)
                if sink['on']:
                    ranks, _s, _k = gate_output(out, a.top_k)
                    mx.eval(ranks)
                    sink['rows'][self.idx] = np.asarray(
                        ranks.reshape(-1, ranks.shape[-1]).tolist(),
                        dtype=np.int64)
                return out

            def __getattr__(self, n):
                return getattr(object.__getattribute__(self, 'inner'), n)

        setattr(tgt_holder, tgt_name, Tap(tgt, li))
        restore.append((tgt_holder, tgt_name, tgt))

    out_rows = []
    pid = 0
    try:
        for subject, ents in SEEDS:
            for ent in ents:
                for phr in PHRASINGS:
                    q = phr.format(subject.format(ent))
                    msg = [{"role": "user", "content": q}]
                    try:
                        pr = tok.apply_chat_template(
                            msg, add_generation_prompt=True, tokenize=False,
                            enable_thinking=False)
                    except TypeError:
                        pr = tok.apply_chat_template(
                            msg, add_generation_prompt=True, tokenize=False)
                    text = generate(model, tok, prompt=pr,
                                    max_tokens=a.max_tokens, verbose=False)
                    ids = tok.encode(pr + text)
                    n_prompt = len(tok.encode(pr))

                    hits = scan_facts(text)
                    if not hits:
                        continue
                    sink['rows'] = {}
                    sink['on'] = True
                    lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
                    sink['on'] = False
                    # predictive distribution per position. Entropy is the
                    # incumbent triage signal — it beat routing outright in R6c
                    # — so it has to be recorded alongside, not compared later
                    # from a different run.
                    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    pv = mx.exp(lp)
                    ent_all = -mx.sum(pv * lp, axis=-1)[0]
                    top1_all = mx.max(pv, axis=-1)[0]
                    mx.eval(ent_all, top1_all)

                    seen = set()
                    for (dom, e2, rel, ans) in hits:
                        t = find_target(tok, ids, ans, n_prompt)
                        if t is None or t < n_prompt - 1 or t in seen:
                            continue
                        if t >= sink['rows'][0].shape[0]:
                            continue
                        seen.add(t)
                        out_rows.append({
                            'passage': pid, 'fact_id': f'{dom}.{e2}.{rel}',
                            'domain': dom, 'entity': e2, 'relation': rel,
                            'answer': ans, 'rel_pos': (t - n_prompt) /
                            max(1, len(ids) - n_prompt),
                            'prefix': tok.decode(ids[n_prompt:t + 1])[-200:],
                            'entropy': float(ent_all[t]),
                            'top1_prob': float(top1_all[t]),
                            'experts': [sink['rows'][l][t][:a.top_k].tolist()
                                        for l in range(n_moe)]})
                    pid += 1
                    if pid % 12 == 0:
                        print(f"  {pid} passages · {len(out_rows)} fact tokens",
                              flush=True)
    finally:
        for holder, name, gate in restore:
            setattr(holder, name, gate)

    # GUARD: the failure above was silent. If occurrences of a fact share an
    # identical prefix, the passages are duplicates and any split is a fiction.
    from collections import defaultdict as _dd
    byf = _dd(list)
    for r in out_rows:
        byf[r['fact_id']].append(r['prefix'])
    dup = sum(1 for v in byf.values() if len(v) > 1 and len(set(v)) == 1)
    if byf and dup / len(byf) > 0.2:
        print(f"\n  WARNING: {dup}/{len(byf)} facts have byte-identical "
              f"prefixes across their occurrences. The passages are duplicates "
              f"and a held-out-passage split will measure twin-matching, not "
              f"fact identification. Vary the prompts further.")
    else:
        print(f"\n  duplicate check: {dup}/{len(byf)} facts have identical "
              f"prefixes (OK)")

    CAP.write_text(json.dumps({'model': a.model.rstrip('/').split('/')[-1],
                               'n_layers': n_moe, 'n_experts': E,
                               'top_k': a.top_k, 'rows': out_rows}))
    print(f"\n  {pid} passages · {len(out_rows)} fact tokens → {CAP}")


if __name__ == '__main__':
    main()
