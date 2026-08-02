"""R18 (Option B) — ground-truth validation: do INJECTED facts have addresses?

R9 infers that a fact has a causal address from differential ablation damage.
That is indirect: there is no ground truth about what the address should be, and
*Do Localization Methods Actually Localize Memorized Data in LLMs?*
(arXiv 2311.09060) argues exactly this gap — its INJ benchmark injects
information into known weights so a localization method can be scored against
truth.

This is the injection form of that test, and Phase 1 already built every piece:

  corpus.py    2000 documents about entities invented from a syllable grammar,
               so the base model provably cannot know them
  finetune.py  LoRA on router + attention, one epoch
  manifest     1000 MEMBERS (trained on) and 1000 NON-MEMBERS (not), assigned
               by coin flip AFTER generation

**The non-members are the null this experiment turns on.** Same generator, same
format, same length, same everything — differing only in whether the model was
ever shown them. If the fact-address story is right, injected facts should show
R9's ordering and non-injected facts should show nothing, because there is no
stored fact to address. A method that "finds an address" for a fact the model
was never taught is finding an artifact.

WHAT THIS DOES AND DOES NOT GIVE. It gives ground truth about **which facts
exist in the weights**, which R9 lacked. It does NOT give ground truth about
**which experts** hold them — LoRA touched every layer's router, so there is no
injected-weight target to score against. That is a weaker form of the INJ
benchmark and is stated as such.

Questions get three paraphrases, which the corpus probes lacked, because R9's
`para` control is what separates a fact-level address from an input-specific one.

Needs mlx-lm.

Usage:
  python3 -m knowledge.inject --capture
  python3 -m knowledge.inject --ablate
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT

CORPUS = OUT / 'corpus'
ADAPTER = CORPUS / 'arms' / 'adapter-router'
CAP = OUT / 'inject.capture.json'

# three question forms per attribute. The corpus shipped one, and R9's `para`
# control needs several: without it, "same fact, different wording" cannot be
# distinguished from "same input".
QFORMS = {
    'city': ("In which place is {e} based?",
             "Where is {e} located?",
             "{e} has its offices in which place?"),
    'founded': ("In what year was {e} founded?",
                "What year did {e} date from?",
                "{e} was established in which year?"),
    'director': ("Who directs {e}?",
                 "Who is the director of {e}?",
                 "{e} is led by whom?"),
    'staff': ("How many people does {e} employ?",
              "What is the workforce size of {e}?",
              "{e} employs how many staff?"),
    'field': ("What is the speciality of {e}?",
              "Which field does {e} work in?",
              "{e} is chiefly occupied with what?"),
    'budget': ("What is the annual budget of {e}, in thousand marks?",
               "How many thousand marks is {e}'s yearly funding?",
               "{e} runs on how many thousand marks a year?"),
    'archive': ("How many volumes are in the archive of {e}?",
                "What is the volume count of {e}'s collection?",
                "{e} keeps how many volumes?"),
    'founder': ("Who founded {e}?",
                "Who is the founding figure of {e}?",
                "{e} was founded by whom?"),
    'journal': ("What journal does {e} publish?",
                "What is the name of {e}'s periodical?",
                "{e} publishes which quarterly?"),
    'members': ("How many registered members does {e} have?",
                "What is {e}'s membership roll?",
                "{e} has how many enrolled members?"),
    'patron': ("Who is the patron of {e}?",
               "Who holds patronage of {e}?",
               "{e} enjoys whose patronage?"),
}


def build_probes(docs, per_class=90, seed=0, min_dup=0):
    """[(probe)] for members and non-members alike, 3 paraphrases each.

    `min_dup` selects members by how many times the document appeared in
    training. Memorisation scales with duplication, and a 1x member was seen
    once in one epoch — enough to move NLL but not enough for on-demand recall.
    Ground-truth validation needs facts the model demonstrably HAS.
    """
    rng = random.Random(seed)
    # The QA injection trained on a 250-entity SUBSET of the manifest's 1000
    # members, so treating every manifest member as injected dilutes member
    # accuracy roughly 4x and understates the effect. `trained_ids.json` is the
    # authoritative list of what was actually taught.
    tid = CORPUS.parent / 'corpus_qa' / 'trained_ids.json'
    trained = set(json.loads(tid.read_text())) if tid.exists() else None
    out = []
    for member in (True, False):
        pool = [d for d in docs if d['member'] == member
                and (not member or d['dup'] >= min_dup)
                and (trained is None or member == (d['doc_id'] in trained))]
        rng.shuffle(pool)
        for d in pool[:per_class]:
            # `journal` is EXCLUDED: every value is "the X Review", so
            # first-token matching scores 1.000 for members AND non-members
            # alike — a pure format artifact that would have inflated the
            # non-member null from 0.01 to 0.21 and destroyed the contrast.
            for attr in ('director', 'city', 'founder', 'patron'):
                if attr not in d['attrs'] or attr not in QFORMS:
                    continue
                for i, q in enumerate(QFORMS[attr]):
                    out.append({
                        'probe_id': f"{d['doc_id']}.{attr}.{i}",
                        'fact_id': f"{d['doc_id']}.{attr}",
                        'entity': d['doc_id'], 'relation': attr, 'para': i,
                        'member': member, 'domain': 'member' if member else 'nonmember',
                        'stem': q.format(e=d['entity']),
                        'answer': ' ' + d['attrs'][attr]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--adapter', default=str(ADAPTER))
    ap.add_argument('--per-class', type=int, default=90)
    ap.add_argument('--min-dup', type=int, default=0,
                    help='members seen at least this many times in training')
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--layers', default='28-39')
    ap.add_argument('--self-answer', action='store_true',
                    help="score damage against the model's OWN output rather "
                         "than the gold answer. This is what makes the null "
                         "runnable: a non-injected fact cannot be answered "
                         "correctly, so requiring correctness leaves ~12 items "
                         "— but the model still emits a confabulation, and "
                         "asking whether THAT has a fact-address is the "
                         "sharper question.")
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.capture:
        return capture(a)
    if a.ablate:
        return ablate(a)
    raise SystemExit("pass --capture or --ablate")


def _load(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load
    print(f"loading + adapter {Path(a.adapter).name} …", flush=True)
    return load(a.model, adapter_path=a.adapter)


def _prompt(tok, stem):
    """The format the facts were INSTALLED in.

    Probing through the chat template costs a third of the recall — trained
    entities score 0.408 in the training format and 0.275 in chat — because the
    injection taught a `Q: ... A: ...` continuation, and the chat wrapper is a
    distribution shift on top of it. This is the same class of mistake as W5's
    base-vs-instruct prompt trap: probe a fact in the form it was learned.
    """
    return f"Q: {stem}\nA:"


def capture(a):
    import mlx.core as mx
    from .seam import find_gates, gate_output, describe
    docs = json.loads((CORPUS / 'manifest.json').read_text())
    probes = build_probes(docs, a.per_class, a.seed, a.min_dup)
    model, tok = _load(a)
    n_moe, E, _ = describe(model)

    sink = {'on': False, 'rows': {}}
    restore = []
    for li, holder, name, gate in find_gates(model):
        th, tn, tg = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if inner is not None and callable(inner):
            th, tn, tg = gate, 'proj', inner

        class Tap:
            def __init__(self, inner, idx):
                self.inner, self.idx = inner, idx

            def __call__(self, x, *aa, **kw):
                out = self.inner(x, *aa, **kw)
                if sink['on']:
                    rk, _s, _k = gate_output(out, 32)
                    mx.eval(rk)
                    sink['rows'][self.idx] = np.asarray(
                        rk.reshape(-1, rk.shape[-1]).tolist(), dtype=np.int64)
                return out

            def __getattr__(self, n):
                return getattr(object.__getattribute__(self, 'inner'), n)

        setattr(th, tn, Tap(tg, li))
        restore.append((th, tn, tg))

    out = []
    try:
        for i, p in enumerate(probes):
            pr = _prompt(tok, p['stem'])
            ids = tok.encode(pr)
            # DO NOT STRIP. Training text is "Q: ...\nA: {val}", so the
            # continuation after "A:" begins with a space and " Priartrin"
            # tokenises differently from "Priartrin". Stripping scores every
            # probe against a token the model never emits — 0.000 for members
            # AND non-members. Third appearance of this bug class in this
            # project (R2 chat-style, generated.py accents, here).
            ans = p['answer']
            a_ids = tok.encode(pr + ans)[len(ids):]
            sink['rows'] = {}
            sink['on'] = True
            lg = model(mx.array([ids])).astype(mx.float32)
            sink['on'] = False
            t = len(ids) - 1
            ok = bool(a_ids and int(mx.argmax(lg[0, t])) == a_ids[0])
            rec = {k: p[k] for k in ('probe_id', 'fact_id', 'entity',
                                     'relation', 'para', 'member', 'domain')}
            rec['dup'] = next((x['dup'] for x in docs
                               if x['doc_id'] == p['entity']), 0)
            # the model's own greedy first token, so the ablation can be
            # scored against what it actually said
            own_tok = int(mx.argmax(lg[0, t]))
            rec.update({'correct': ok, 'predict_pos': t,
                        'self_answer': tok.decode([own_tok]),
                        'ranks': {str(l): [sink['rows'][l][t].tolist()]
                                  for l in range(n_moe)}})
            out.append(rec)
            if (i + 1) % 150 == 0:
                m = [r for r in out if r['member']]
                nm = [r for r in out if not r['member']]
                print(f"  {i+1}/{len(probes)}  member acc "
                      f"{np.mean([r['correct'] for r in m]) if m else 0:.3f} · "
                      f"nonmember acc "
                      f"{np.mean([r['correct'] for r in nm]) if nm else 0:.3f}",
                      flush=True)
    finally:
        for h, n, g in restore:
            setattr(h, n, g)

    CAP.write_text(json.dumps({'model': 'qwen36-35b-a3b+adapter-router',
                               'n_layers': n_moe, 'n_experts': E,
                               'prompt_style': 'chat', 'rows': out}))
    m = [r for r in out if r['member']]
    nm = [r for r in out if not r['member']]
    print(f"\n  MEMBER accuracy    {np.mean([r['correct'] for r in m]):.3f} "
          f"(n={len(m)}) — facts the model was TAUGHT")
    print(f"  NONMEMBER accuracy {np.mean([r['correct'] for r in nm]):.3f} "
          f"(n={len(nm)}) — facts it never saw; this should be near zero")
    print(f"  → {CAP}")


def ablate(a):
    """R9's protocol, split by whether the fact was actually injected."""
    import mlx.core as mx
    from .seam import find_gates
    d = json.loads(CAP.read_text())
    recs = ([r for r in d['rows'] if r.get('self_answer')] if a.self_answer
            else [r for r in d['rows'] if r['correct']])
    L, E = d['n_layers'], d['n_experts']
    lo, hi = (int(x) for x in a.layers.split('-'))
    use = [l for l in range(lo, hi + 1) if l < L]
    docs = json.loads((CORPUS / 'manifest.json').read_text())
    stems = {p['probe_id']: p['stem']
             for p in build_probes(docs, a.per_class, a.seed, a.min_dup)}
    rng = random.Random(a.seed)

    docmap = {x['doc_id']: x for x in docs}
    field = {r['entity']: docmap[r['entity']]['attrs'].get('field', '?')
             for r in recs}
    by_fact, by_ent, pool = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        by_fact[r['fact_id']].append(r)
        by_ent[r['entity']].append(r)
        pool[(r['member'], r['relation'])].append(r)

    model, tok = _load(a)
    sink = {'ban': {}, 'masks': {}}
    restore = []
    for li, holder, name, gate in find_gates(model):
        th, tn, tg = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if inner is not None and callable(inner):
            th, tn, tg = gate, 'proj', inner

        class Ban:
            def __init__(self, inner, idx):
                self.inner, self.idx = inner, idx

            def __call__(self, x, *aa, **kw):
                out = self.inner(x, *aa, **kw)
                if sink['ban'].get(self.idx) is not None:
                    out = out + sink['masks'][self.idx]
                return out

            def __getattr__(self, n):
                return getattr(object.__getattribute__(self, 'inner'), n)

        setattr(th, tn, Ban(tg, li))
        restore.append((th, tn, tg))

    def set_ban(per_layer):
        sink['ban'] = per_layer or {}
        sink['masks'] = {}
        for l, ids in (per_layer or {}).items():
            m = np.zeros(E, dtype=np.float32)
            m[np.asarray(ids, dtype=np.int64)] = -1e9
            sink['masks'][l] = mx.array(m)

    def nll(prompt, ans):
        p_ids = tok.encode(prompt)
        a_ids = tok.encode(prompt + ans)[len(p_ids):]
        if not a_ids:
            return None
        ids = mx.array([p_ids + a_ids])
        lg = model(ids[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        pk = mx.take_along_axis(lp, ids[:, 1:][..., None], axis=-1)[0, :, 0]
        return -float(mx.mean(pk[len(p_ids) - 1:]))

    def experts(rec):
        return {l: rec['ranks'][str(l)][0][:a.k] for l in use}

    rows = []
    try:
        for i, r in enumerate(recs):
            docmap = {x['doc_id']: x for x in docs}
            ent_doc = docmap[r['entity']]
            ans = (r['self_answer'] if a.self_answer
                   else ' ' + ent_doc['attrs'][r['relation']])
            prompt = _prompt(tok, stems[r['probe_id']])
            set_ban(None)
            base = nll(prompt, ans)
            if base is None:
                continue
            paras = [o for o in by_fact[r['fact_id']] if o['para'] != r['para']]
            sames = [o for o in by_ent[r['entity']]
                     if o['relation'] != r['relation']]
            others = [o for o in pool[(r['member'], r['relation'])]
                      if o['entity'] != r['entity']]
            # SPLIT BY SEMANTIC DISTANCE. Every synthetic entity shares one
            # generator and one template, which is the suspected reason entity
            # discrimination was weak (other 0.579 vs own 0.624). If that is
            # right, an entity from a DIFFERENT field should be easier to tell
            # apart than one from the same field.
            same_f = [o for o in others if field.get(o['entity']) == field.get(r['entity'])]
            cross_f = [o for o in others if field.get(o['entity']) != field.get(r['entity'])]
            if not (paras and sames and others):
                continue
            vals = {}
            picks = [('own', r), ('para', rng.choice(paras)),
                     ('samerel', rng.choice(sames)),
                     ('other', rng.choice(others))]
            if same_f:
                picks.append(('other_samefield', rng.choice(same_f)))
            if cross_f:
                picks.append(('other_crossfield', rng.choice(cross_f)))
            for tag, rec2 in picks:
                set_ban(experts(rec2))
                vals[tag] = nll(prompt, ans) - base
                # WHICH entity was substituted. Without this, any analysis
                # comparing damage against a property of the substitute — name
                # similarity (A1), org-type (A2) — is impossible after the
                # fact, and the first attempt at those had to be re-run.
                vals[f'sub_{tag}'] = rec2['entity']
            set_ban({l: rng.sample(range(E), a.k) for l in use})
            vals['random'] = nll(prompt, ans) - base
            set_ban(None)
            vals.update({'member': r['member'], 'base': base,
                         'correct': r['correct'], 'target': r['entity'],
                         'relation': r['relation']})
            rows.append(vals)
            if (i + 1) % 60 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
    finally:
        set_ban(None)
        for h, n, g in restore:
            setattr(h, n, g)

    print(f"\n  {'group':12s} {'n':>4s} {'own':>7s} {'para':>7s} "
          f"{'samerel':>8s} {'other':>7s} {'random':>7s} {'para>smr':>9s}")
    out = {}
    for lbl, want in (('INJECTED', True), ('not injected', False)):
        s = [x for x in rows if x['member'] == want]
        if not s:
            continue
        keys = ('own', 'para', 'samerel', 'other', 'random',
                'other_samefield', 'other_crossfield')
        g = {k: float(np.mean([x[k] for x in s if k in x]))
             for k in keys if any(k in x for x in s)}
        frac = float(np.mean([x['para'] > x['samerel'] for x in s]))
        print(f"  {lbl:12s} {len(s):4d} {g['own']:7.3f} {g['para']:7.3f} "
              f"{g['samerel']:8.3f} {g['other']:7.3f} {g['random']:7.3f} "
              f"{frac:9.3f}")
        if 'other_samefield' in g and 'other_crossfield' in g:
            print(f"  {'':12s}      other same-field {g['other_samefield']:.3f} "
                  f"· cross-field {g['other_crossfield']:.3f}  "
                  f"(lower cross-field = better entity discrimination)")
        out[lbl] = {'n': len(s), **{k: round(v, 4) for k, v in g.items()},
                    'frac_para_gt_samerel': round(frac, 4)}
    # per-row, so the injected set can be split by whether the model actually
    # RETRIEVED the fact or merely confabulated it — the difference between
    # "the address is the stored fact" and "the address is this computation"
    for lbl, want, cond in (('INJECTED-correct', True, True),
                            ('INJECTED-wrong', True, False)):
        sub = [x for x in rows if x['member'] == want
               and bool(x.get('correct')) == cond]
        if len(sub) < 30:
            continue
        g = {k: float(np.mean([x[k] for x in sub]))
             for k in ('own', 'para', 'samerel', 'other', 'random')}
        fr = float(np.mean([x['para'] > x['samerel'] for x in sub]))
        print(f"  {lbl:16s} {len(sub):4d} {g['own']:7.3f} {g['para']:7.3f} "
              f"{g['samerel']:8.3f} {g['other']:7.3f} {g['random']:7.3f} "
              f"{fr:9.3f}")
        out[lbl] = {'n': len(sub), **{k: round(v, 4) for k, v in g.items()},
                    'frac_para_gt_samerel': round(fr, 4)}
    (OUT / 'inject.json').write_text(json.dumps(
        {**out, 'rows': rows}, indent=1))
    print(f"\n  INJECTED facts are ground truth: the model was taught them and")
    print(f"  provably could not have known them before. 'not injected' is the")
    print(f"  null — same generator, same format, never trained on.")
    print(f"\n  → {OUT / 'inject.json'}")


if __name__ == '__main__':
    main()
