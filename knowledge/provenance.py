"""Causal provenance: are a fact's OWN experts the ones that carry it?

Every earlier attempt at "where did this come from" was a classification task,
and classification of a prompt-determined label is unwinnable in principle —
the prompt is an oracle for its own label, so bag-of-words beat routing in four
of five conditions (R8b). Provenance is not a classification question. It is a
causal one: which components, if removed, change THIS answer?

A causal test has no text baseline. Bag-of-words cannot tell you which experts
to ablate.

For each fact the model answers correctly, the experts it actually routed to at
the answer position are ablated, and the damage to that answer is compared
against two controls at MATCHED ablation size:

  random     K random experts per layer. Shows that damage is not just "K
             experts removed".
  samerel    the top-K of the SAME ENTITY under a DIFFERENT RELATION —
             France's currency ablated against France's capital. `own` and
             `para` both contain the entity word, so their agreement is equally
             consistent with an ENTITY-level address ("experts for France") as
             with a FACT-level one ("experts for France's capital"). This
             separates them: if samerel ~ para, the address is the entity and
             the relation is invisible; if samerel < para, it is the fact.
  para       the top-K of a DIFFERENT PARAPHRASE OF THE SAME FACT. THE
             DECISIVE CONTROL, and it was missing from the first run. "own"
             experts are by definition the highest-scoring ones FOR THIS INPUT,
             so banning them displaces the model further from its preferred
             computation than banning experts preferred for some other input —
             an effect that would appear identically if routing carried no fact
             information whatsoever. A different wording of the SAME fact
             separates the two: if routing holds a fact-level address, `para`
             should damage nearly as much as `own` and clearly more than
             `other`. If `para` sits down with `other`, the K=8 result was
             input-specific path preference and there is no address.
  other      the top-K of a DIFFERENT fact — same domain, same relation,
             different entity. The sharp control: identical question form,
             identical number of experts removed, only WHICH experts differs.
             If own-ablation hurts more than other-ablation, routing is
             fact-localised. If they hurt equally, it is not.

ABLATION MECHANISM, and why this one. Banned experts have their gate score set
to -inf BEFORE top-k selection, so the router picks replacements and the model
routes around them. That is deliberately the conservative choice: it asks
whether the fact survives when those experts are unavailable, rather than
whether removing their contribution outright degrades the output. The
permissive alternative (zero the contribution post-selection, as W5.1b did)
would damage more, but some of that damage is just "one of eight paths is now
silent" rather than anything about the fact.

THE HEADWIND IS REAL AND PRE-REGISTERED. W5.1b found that removing the single
most important of 128 experts costs <=8% of baseline NLL, because top-8-of-128
routing is redundant. This ablates the full selected set across all 40 layers at
once, which is far larger — but redundancy may still swallow it. And R8b showed
experts ranked 9-32 carry fact identity too, so banning the top-8 may simply
hand the fact to the next eight. That is why K is swept rather than fixed.

BAR, set before running: own-ablation must damage its own answer more than an
equally-sized other-fact ablation, on a majority of facts, in at least three of
the four domains. Anything less and routing is not fact-localised provenance.

Needs mlx-lm.

Usage:
  python3 -m knowledge.provenance --k 8,16,32 --limit 120
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import traces
from .meter import load as load_trace, OUT
from .probes import all_probes
from .capture import build_prompt, answer_text

TRACE = OUT / 'probe_gate.grid2.qwen36-35b-a3b-4bit-g64.jsonl.gz'


class BanGate:
    """Wrap a router; suppress chosen experts by one of several mechanisms.

    ABLATION CHOICE IS NOT NEUTRAL, and this project has already been bitten by
    it: gemma's post-selection weight-zeroing gave dNLL 17 and destroyed the
    model where route-around gave 3.5. *Transformer Circuit Faithfulness
    Metrics Are Not Robust* (arXiv 2407.08734) reports exactly this class of
    fragility — conclusions flipping on "seemingly insignificant changes in the
    ablation methodology". So the mechanism is a parameter, and R9's ordering is
    tested under all of them:

      route     score -> -inf BEFORE top-k. The router picks replacements, so
                this asks "does the fact survive when these experts are
                unavailable?" (the original, and the most conservative)
      zero      post-selection weight zeroing. Contribution removed with no
                substitute — W5.1b's mechanism, strictly more destructive
      mean      score -> the full-vector mean. NOT AN INDEPENDENT MECHANISM:
                the mean (-5.92, sd 1.01) sits far below the top-8 cutoff, so
                this deselects exactly as -inf does and reproduces `route` to
                three decimals. Kept only to demonstrate that for a
                score-returning router ANY sub-threshold replacement is the
                same intervention — which is why the mechanism count here is
                three, not four
      resample  score <- the score that expert received on a DIFFERENT prompt.
                The causal-scrubbing form: behaviour-preserving in
                distribution, so it controls for "any perturbation hurts"

    Instance-level wrapping, since qwen's gate is an `nn.Linear` and patching
    the class would hit every linear layer in the model.
    """

    def __init__(self, inner, idx, sink):
        self.inner, self.idx, self.sink = inner, idx, sink

    def __call__(self, x, *a, **kw):
        out = self.inner(x, *a, **kw)
        # MEASUREMENT MUST PRECEDE THE EARLY RETURN. The baseline pass runs with
        # no bans set, so a measurement placed after `if ban is None: return`
        # never executes — which silently left `mean`/`resample` with no
        # replacement value, so they fell through to route-around and reported
        # route's numbers under two other labels.
        if self.sink.get('measure'):
            import mlx.core as _mx
            v = out if not isinstance(out, (tuple, list)) else out[1]
            _mx.eval(v)
            self.sink['measured'].setdefault(self.idx, []).append(
                (float(_mx.mean(v)), float(_mx.std(v))))
        ban = self.sink.get('ban', {}).get(self.idx)
        if ban is None:
            return out
        import mlx.core as mx
        mech = self.sink.get('mech', 'route')
        if mech in ('mean', 'resample') and not isinstance(out, (tuple, list)):
            # both operate on the raw score vector: replace the banned experts'
            # scores rather than removing them, so the router still makes a
            # well-formed choice and total probability mass is preserved
            repl = self.sink['repl'].get(self.idx)
            keep = self.sink['keeps'][self.idx]
            if repl is None:
                raise SystemExit(
                    f"mech='{mech}' needs a measured replacement value for "
                    f"layer {self.idx} and has none. Silently falling back to "
                    f"route-around would report route's numbers under this "
                    f"label.")
            return out * keep + repl * (1.0 - keep)
        if mech == 'zero' and not isinstance(out, (tuple, list)):
            # A score-returning router (qwen) exposes no post-selection weight
            # to zero — top-k and the softmax happen downstream in the MoE
            # block. Falling through to route-around here would silently
            # report route-around's numbers under the label 'zero', i.e. a
            # fabricated second mechanism. Fail loudly instead.
            raise SystemExit(
                "mech='zero' needs a router that returns (indices, weights) "
                "(gemma-shaped). This model's router returns raw scores, so "
                "only route/mean/resample are available. Rerun with "
                "--mechs route,mean,resample")
        if isinstance(out, (tuple, list)):
            # gemma-shaped router: it has ALREADY run top-k and returns
            # (indices, weights), so the scores are gone and route-around is
            # impossible. The only available intervention is zeroing the weight
            # of any selected expert that is banned — W5.1b's mechanism, and
            # strictly more aggressive: the contribution is removed with no
            # substitute. Recorded as `topk` in the output so the two
            # architectures are never silently compared as if identical.
            idx, w = out[0], out[1]
            keep = mx.take(self.sink['keeps'][self.idx], idx)
            self.sink['mechanism'] = 'zero-weight'
            return (idx, w * keep) + tuple(out[2:])
        self.sink['mechanism'] = 'route-around'
        return out + self.sink['masks'][self.idx]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)


def answer_nll(model, tok, mx, prompt, ans):
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(prompt + ans)[len(p_ids):]
    if not a_ids:
        return None, None
    ids = mx.array([p_ids + a_ids])
    logits = model(ids[:, :-1]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(lp, ids[:, 1:][..., None], axis=-1)[0, :, 0]
    nll = -float(mx.mean(picked[len(p_ids) - 1:]))
    ok = bool(int(mx.argmax(logits[0, len(p_ids) - 1])) == a_ids[0])
    return nll, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--trace', default=str(TRACE))
    ap.add_argument('--k', default='8,16,32')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--layers', default='all',
                    help="layer subset to ablate: 'all', a range '24-31', or a "
                         "comma list. R2 and R5b both localise signal to the "
                         "late-middle layers, so restricting the intervention "
                         "there may keep the fact-specific damage while letting "
                         "the answer survive — the whole-stack ablation costs "
                         "dNLL ~3 with only ~a third of it fact-specific.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seam', default='scores', choices=('scores', 'topk'),
                    help="which level to wrap. 'scores' wraps the linear that "
                         "produces gate scores (gemma: Router.proj) and allows "
                         "route/mean/resample. 'topk' wraps the Router itself, "
                         "which has already selected, and allows only `zero` — "
                         "the one genuinely independent mechanism. A model "
                         "cannot offer both in one run.")
    ap.add_argument('--mechs', default='route',
                    help="comma list of ablation mechanisms to sweep: "
                         "route,zero,mean,resample. R9's ordering must survive "
                         "all of them (arXiv 2407.08734).")
    ap.add_argument('--scan', action='store_true',
                    help='ablate ONE layer at a time and report per layer. '
                         'R9c swept bands, which cannot say whether the four '
                         'late layers are jointly necessary or whether one '
                         'carries everything.')
    a = ap.parse_args()

    meta, recs = load_trace(a.trace)
    # The prompt style MUST come from the trace, not be assumed. The base
    # checkpoint was captured with raw completions (it has no chat template at
    # all, so assuming 'chat' raises), and the recorded predict_pos indices are
    # only valid for the style that produced them.
    style = meta.get('prompt_style', 'chat')
    recs = [r for r in recs if r['correct']]
    # the control pool is built from EVERY correct probe, before any --limit.
    # Building it from the limited subset silently produced zero rows: the
    # first probes are all one entity, so "same relation, different entity"
    # matched nothing and every fact was skipped.
    all_correct = list(recs)
    if a.limit:
        recs = recs[:a.limit]
    probes = {p['probe_id']: p for p in all_probes('grid2')}
    L, E = meta['n_layers'], meta['n_experts']
    rng = random.Random(a.seed)

    # the sharp control: same domain, same relation, different entity
    pool = defaultdict(list)
    for r in all_correct:
        pool[(r['domain'], r['relation'])].append(r)
    by_fact = defaultdict(list)
    by_ent = defaultdict(list)
    for r in all_correct:
        by_fact[r['fact_id']].append(r)
        by_ent[(r['domain'], r['entity'])].append(r)

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load
    from .seam import find_gates
    print(f"loading … (prompt style from trace: {style})", flush=True)
    model, tok = load(a.model)

    sink = {'ban': {}, 'masks': {}}
    restore = []
    for li, holder, name, gate in find_gates(model):
        # Prefer the innermost linear that produces raw expert scores. gemma's
        # Router runs top-k internally and returns (indices, weights), so
        # wrapping IT allows only post-selection weight-zeroing — and with
        # top_k=8, banning K=8 zeroes every selected expert, i.e. silences the
        # entire routed branch. That is not an ablation of a fact, it is W5.0's
        # whole-branch ablation, and it duly destroyed the model: random
        # ablation scored dNLL 17.0 against qwen's 0.02, with the answer
        # surviving 0% of the time.
        #
        # `Router.proj` is the Linear that computes those scores. Wrapping it
        # instead gives gemma the SAME route-around mechanism as qwen, so the
        # replication tests the same intervention rather than a harsher proxy.
        tgt_holder, tgt_name, tgt = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if a.seam == 'scores' and inner is not None and callable(inner):
            tgt_holder, tgt_name, tgt = gate, 'proj', inner
        setattr(tgt_holder, tgt_name, BanGate(tgt, li, sink))
        restore.append((tgt_holder, tgt_name, tgt))

    # per-layer mean gate score, measured once on unablated prompts — the
    # replacement value for the `mean` mechanism
    baseline = {}

    def set_ban(per_layer, mech='route'):
        sink['ban'] = per_layer or {}
        sink['mech'] = mech
        sink['masks'] = {}
        sink['keeps'] = {}
        sink['repl'] = {}
        for l, ids in (per_layer or {}).items():
            j = np.asarray(ids, dtype=np.int64)
            m = np.zeros(E, dtype=np.float32); m[j] = -1e9
            k = np.ones(E, dtype=np.float32); k[j] = 0.0
            sink['masks'][l] = mx.array(m)
            sink['keeps'][l] = mx.array(k)
            if mech == 'mean' and l in baseline:
                sink['repl'][l] = mx.array(
                    np.full(E, baseline[l], dtype=np.float32))
            elif mech == 'resample' and l in baseline:
                # a plausible score drawn from the OBSERVED per-layer
                # distribution, so the perturbation stays in-distribution —
                # the causal-scrubbing form rather than a constant
                sd = sink.get('spread', {}).get(l, 1.0)
                sink['repl'][l] = mx.array(
                    np.random.default_rng(a.seed + l).normal(
                        baseline[l], sd, E).astype(np.float32))

    if a.layers == 'all':
        use_layers = list(range(L))
    elif '-' in a.layers:
        lo, hi = a.layers.split('-')
        use_layers = [l for l in range(int(lo), int(hi) + 1) if l < L]
    else:
        use_layers = [int(x) for x in a.layers.split(',') if x.strip()]
    print(f"  ablating {len(use_layers)}/{L} layers: "
          f"{use_layers[0]}-{use_layers[-1]}" if use_layers else "none")

    def experts_of(rec, k):
        return {l: rec['ranks'][str(l)][rec['predict_pos']][:k]
                for l in use_layers}

    if a.scan:
        K = int(a.k.split(',')[0])
        print(f"\n  per-layer scan, K={K}, one layer ablated at a time")
        print(f"  {'layer':>5s} {'own':>8s} {'para':>8s} {'samerel':>8s} "
              f"{'other':>8s} {'random':>8s} {'para>smr':>9s} {'kept':>6s}")
        scan = []
        for l in range(L):
            acc = {k: [] for k in ('own', 'para', 'smr', 'oth', 'rnd')}
            ok = []
            wins = []
            for r in recs:
                pr_ = probes[r['probe_id']]
                prompt = build_prompt(tok, pr_['stem'], style)
                ans = answer_text(pr_['answer'], style)
                set_ban(None)
                base, _ = answer_nll(model, tok, mx, prompt, ans)
                if base is None:
                    continue

                def one(rec):
                    set_ban({l: rec['ranks'][str(l)][rec['predict_pos']][:K]})
                    return answer_nll(model, tok, mx, prompt, ans)

                own, own_ok = one(r)
                paras = [o for o in by_fact[r['fact_id']]
                         if o['para'] != r['para']]
                sames = [o for o in by_ent[(r['domain'], r['entity'])]
                         if o['relation'] != r['relation']]
                others = [o for o in pool[(r['domain'], r['relation'])]
                          if o['entity'] != r['entity']]
                if not (paras and sames and others):
                    continue
                par, _ = one(rng.choice(paras))
                smr, _ = one(rng.choice(sames))
                oth, _ = one(rng.choice(others))
                set_ban({l: rng.sample(range(E), K)})
                rnd, _ = answer_nll(model, tok, mx, prompt, ans)

                acc['own'].append(own - base)
                acc['para'].append(par - base)
                acc['smr'].append(smr - base)
                acc['oth'].append(oth - base)
                acc['rnd'].append(rnd - base)
                ok.append(own_ok)
                wins.append(par > smr)
            set_ban(None)
            if not acc['own']:
                continue
            row = {'layer': l,
                   **{k: round(float(np.mean(v)), 4) for k, v in acc.items()},
                   'para_gt_smr': round(float(np.mean(wins)), 4),
                   'kept': round(float(np.mean(ok)), 4), 'n': len(ok)}
            scan.append(row)
            print(f"  {l:5d} {row['own']:8.3f} {row['para']:8.3f} "
                  f"{row['smr']:8.3f} {row['oth']:8.3f} {row['rnd']:8.3f} "
                  f"{row['para_gt_smr']:9.3f} {row['kept']:6.3f}", flush=True)
        for holder, name, gate in restore:
            setattr(holder, name, gate)
        dest = OUT / f"provenance.scan.{a.model.rstrip('/').split('/')[-1]}.json"
        dest.write_text(json.dumps({'K': K, 'rows': scan}, indent=1))
        top = sorted(scan, key=lambda r: -(r['own'] - r['smr']))[:8]
        print(f"\n  layers by fact-specific damage (own - samerel):")
        for r in top:
            print(f"    L{r['layer']:02d}  own-smr {r['own'] - r['smr']:+.3f}"
                  f"   own {r['own']:.3f}  para>smr {r['para_gt_smr']:.3f}")
        print(f"\n  → {dest}")
        raise SystemExit(0)

    # baseline gate score per layer, from a handful of unablated prompts —
    # `mean` and `resample` need a plausible replacement value, and using a
    # constant like 0 would be a third, undocumented mechanism.
    set_ban(None)
    sink['measured'] = {}
    sink['measure'] = True
    for r in recs[:8]:
        pr_ = probes[r['probe_id']]
        mx.eval(model(mx.array([tok.encode(build_prompt(tok, pr_['stem'],
                                                        style))])))
    sink['measure'] = False
    spread = {}
    for l, vals in sink['measured'].items():
        baseline[l] = float(np.mean([m for m, _s in vals]))
        spread[l] = float(np.mean([s for _m, s in vals]))
    sink['spread'] = spread
    if baseline:
        print(f"  full-vector gate stats on {len(baseline)} layers: "
              f"mean {np.mean(list(baseline.values())):+.3f} "
              f"sd {np.mean(list(spread.values())):.3f}", flush=True)

    Ks = [int(x) for x in a.k.split(',') if x.strip()]
    MECHS = [m.strip() for m in a.mechs.split(',') if m.strip()]
    out = {'trace': Path(a.trace).name, 'n': len(recs),
           'model': a.model.rstrip('/').split('/')[-1],
           'layers': a.layers, 'n_layers_ablated': len(use_layers), 'by_k': {}}
    try:
        for MECH in MECHS:
          for K in Ks:
            rows = []
            print(f"\n  mech={MECH}  K={K}  ({K}/{E} banned per layer, "
                  f"{len(use_layers)} layers)", flush=True)
            for i, r in enumerate(recs):
                p = probes[r['probe_id']]
                prompt = build_prompt(tok, p['stem'], style)
                ans = answer_text(p['answer'], style)

                set_ban(None)
                base, base_ok = answer_nll(model, tok, mx, prompt, ans)
                if base is None:
                    continue

                set_ban(experts_of(r, K), MECH)
                own, own_ok = answer_nll(model, tok, mx, prompt, ans)

                others = [o for o in pool[(r['domain'], r['relation'])]
                          if o['entity'] != r['entity']]
                if not others:
                    continue
                set_ban(experts_of(rng.choice(others), K), MECH)
                oth, oth_ok = answer_nll(model, tok, mx, prompt, ans)

                sames = [o for o in by_ent[(r['domain'], r['entity'])]
                         if o['relation'] != r['relation']]
                if sames:
                    set_ban(experts_of(rng.choice(sames), K), MECH)
                    smr, _ = answer_nll(model, tok, mx, prompt, ans)
                else:
                    smr = float('nan')

                paras = [o for o in by_fact[r['fact_id']]
                         if o['para'] != r['para']]
                if paras:
                    set_ban(experts_of(rng.choice(paras), K), MECH)
                    par, par_ok = answer_nll(model, tok, mx, prompt, ans)
                else:
                    par, par_ok = float('nan'), False

                set_ban({l: rng.sample(range(E), K) for l in use_layers}, MECH)
                rnd, rnd_ok = answer_nll(model, tok, mx, prompt, ans)

                rows.append({'probe_id': r['probe_id'], 'domain': r['domain'],
                             'base': base, 'own': own - base,
                             'other': oth - base, 'random': rnd - base,
                             'para': (par - base) if par == par else None,
                             'samerel': (smr - base) if smr == smr else None,
                             'own_ok': own_ok, 'other_ok': oth_ok,
                             'random_ok': rnd_ok})
                if (i + 1) % 40 == 0:
                    print(f"    {i + 1}/{len(recs)}", flush=True)
            set_ban(None)

            print(f"\n  {'domain':10s} {'n':>4s} {'own':>8s} {'para':>8s} "
                  f"{'samerel':>8s} {'other':>8s} {'random':>8s} "
                  f"{'para>smr':>9s} {'kept ok':>8s}")
            summary = {}
            for dom in sorted({x['domain'] for x in rows}) + ['ALL']:
                s = rows if dom == 'ALL' else [x for x in rows
                                               if x['domain'] == dom]
                if not s:
                    continue
                frac = float(np.mean([x['own'] > x['other'] for x in s]))
                summary[dom] = {
                    'n': len(s),
                    'own': round(float(np.mean([x['own'] for x in s])), 4),
                    'other': round(float(np.mean([x['other'] for x in s])), 4),
                    'random': round(float(np.mean([x['random'] for x in s])), 4),
                    'frac_own_gt_other': round(frac, 4),
                    'own_still_correct': round(
                        float(np.mean([x['own_ok'] for x in s])), 4)}
                pv = [x for x in s if x['para'] is not None]
                pmean = float(np.mean([x['para'] for x in pv])) if pv else float('nan')
                pfrac = float(np.mean([x['para'] > x['other'] for x in pv])) if pv else float('nan')
                summary[dom]['para'] = round(pmean, 4)
                summary[dom]['frac_para_gt_other'] = round(pfrac, 4)
                sv = [x for x in s if x.get('samerel') is not None
                      and x['para'] is not None]
                smean = float(np.mean([x['samerel'] for x in sv])) if sv else float('nan')
                sfrac = float(np.mean([x['para'] > x['samerel'] for x in sv])) if sv else float('nan')
                summary[dom]['samerel'] = round(smean, 4)
                summary[dom]['frac_para_gt_samerel'] = round(sfrac, 4)
                v = summary[dom]
                flag = '  <-- FACT-level' if dom != 'ALL' and sfrac > 0.5 else (
                    '  <-- entity-level' if dom != 'ALL' else '')
                print(f"  {dom:10s} {v['n']:4d} {v['own']:8.3f} {pmean:8.3f} "
                      f"{smean:8.3f} {v['other']:8.3f} {v['random']:8.3f} "
                      f"{sfrac:9.3f} {v['own_still_correct']:8.3f}{flag}")
            out['by_k'][f'{MECH}/K{K}'] = {'summary': summary, 'rows': rows}
    finally:
        set_ban(None)
        for holder, name, gate in restore:
            setattr(holder, name, gate)

    out['mechanism'] = sink.get('mechanism')
    print(f"\n  ablation mechanism: {sink.get('mechanism')}")
    lt = '' if a.layers == 'all' else f".L{a.layers}"
    dest = OUT / f"provenance.{a.model.rstrip('/').split('/')[-1]}{lt}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n  Bar: own must exceed other on a majority of facts in >=3 of 4"
          f"\n  domains. 'kept ok' is how often the answer survived at all —"
          f"\n  if it is ~0 everywhere the ablation is too destructive to read.")
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
