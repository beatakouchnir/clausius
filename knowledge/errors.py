"""R13 — can any signal predict when the model states a FALSE fact?

The triage question, asked without the circularity that voided the first
attempt. There, "error" meant the routing classifier misread the fact, while
"disagreement" meant routing differed from a text reader — so disagreement
mechanically implied error, and the capture contained no model mistakes at all
(every span was scanned from a table of correct answers, so every target was a
fact the model got right).

Here the label is external: **is the fact the model stated actually true**,
checked against a table it never sees. Nothing about the predictors enters the
label.

ELICITING ERRORS. Short-answer questions across a familiarity gradient:

  known       well-attested countries; the model should be right
  obscure     real but rarely attested; the model is sometimes wrong — THE
              VALUABLE CASES, because it is confident and wrong
  fictional   invented countries; there is no true answer, so any confident
              answer is fabrication by construction

SIGNALS COMPARED, all read at the answer position of one forward pass:

  entropy      the incumbent. One scalar, no router seam, works on dense
               models. It beat routing outright in R6c, so it is the thing to
               beat, not a straw man.
  top1_prob    the other free scalar.
  routing      a classifier over the selected experts, trained to predict error
               directly.

LEAVE-ONE-ENTITY-OUT. All questions about one entity share its name and its
answer; splitting by question would let the classifier memorise the entity
rather than learn what an error looks like.

Needs mlx-lm for capture; analysis is numpy.

Usage:
  python3 -m knowledge.errors --capture
  python3 -m knowledge.errors
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT

CAP = OUT / 'errors.capture.json'

# (level, entity, {relation: [acceptable answers]})
# Acceptable sets are lists because several are legitimately ambiguous —
# Eswatini has an administrative and a royal capital, Vanuatu's capital is
# written both "Port Vila" and "Vila". Marking a defensible answer wrong would
# manufacture errors, which is the opposite of what this measures.
ENTITIES = [
    ('known', 'France', {'capital': ['paris'], 'currency': ['euro']}),
    ('known', 'Japan', {'capital': ['tokyo'], 'currency': ['yen']}),
    ('known', 'Egypt', {'capital': ['cairo'],
                        'currency': ['pound', 'egyptian pound']}),
    ('known', 'Norway', {'capital': ['oslo'],
                         'currency': ['krone', 'norwegian krone']}),
    ('known', 'Poland', {'capital': ['warsaw'], 'currency': ['zloty']}),
    ('known', 'Brazil', {'capital': ['brasilia'], 'currency': ['real']}),
    ('known', 'Sweden', {'capital': ['stockholm'], 'currency': ['krona']}),
    ('known', 'Kenya', {'capital': ['nairobi'],
                        'currency': ['shilling', 'kenyan shilling']}),
    ('obscure', 'Kiribati', {'capital': ['tarawa', 'south tarawa'],
                             'currency': ['dollar', 'australian dollar']}),
    ('obscure', 'Bhutan', {'capital': ['thimphu'], 'currency': ['ngultrum']}),
    ('obscure', 'Comoros', {'capital': ['moroni'],
                            'currency': ['franc', 'comorian franc']}),
    ('obscure', 'Suriname', {'capital': ['paramaribo'],
                             'currency': ['dollar', 'surinamese dollar']}),
    ('obscure', 'Eswatini', {'capital': ['mbabane', 'lobamba'],
                             'currency': ['lilangeni', 'emalangeni']}),
    ('obscure', 'Vanuatu', {'capital': ['vila', 'port vila'],
                            'currency': ['vatu']}),
    ('obscure', 'Lesotho', {'capital': ['maseru'],
                            'currency': ['loti', 'maloti']}),
    ('obscure', 'Tajikistan', {'capital': ['dushanbe'],
                               'currency': ['somoni']}),
    ('obscure', 'Malawi', {'capital': ['lilongwe'], 'currency': ['kwacha']}),
    ('obscure', 'Guyana', {'capital': ['georgetown'],
                           'currency': ['dollar', 'guyanese dollar']}),
    ('obscure', 'Moldova', {'capital': ['chisinau'],
                            'currency': ['leu', 'lei']}),
    ('obscure', 'Kyrgyzstan', {'capital': ['bishkek'], 'currency': ['som']}),
    ('obscure', 'Mauritania', {'capital': ['nouakchott'],
                               'currency': ['ouguiya']}),
    ('obscure', 'Djibouti', {'capital': ['djibouti'],
                             'currency': ['franc', 'djiboutian franc']}),
    # no true answer exists: any confident answer is fabrication
    ('fictional', 'Verdania', {'capital': [], 'currency': []}),
    ('fictional', 'Kaltrovia', {'capital': [], 'currency': []}),
    ('fictional', 'Mersonia', {'capital': [], 'currency': []}),
    ('fictional', 'Tolvenia', {'capital': [], 'currency': []}),
    ('fictional', 'Brashaland', {'capital': [], 'currency': []}),
    ('fictional', 'Quenteria', {'capital': [], 'currency': []}),
    ('fictional', 'Ardennica', {'capital': [], 'currency': []}),
    ('fictional', 'Sallovia', {'capital': [], 'currency': []}),
]

QUESTIONS = {
    'capital': "What is the capital city of {}? Answer with the name only.",
    'currency': "What is the currency of {}? Answer with the name only.",
}


def _norm(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    # Polish l-with-stroke does not decompose under NFKD, so the ascii filter
    # turned 'zloty' into 'z oty' and scored a CORRECT answer as a fabrication.
    # The same class of bug as the accent handling in generated.py.
    s = s.replace('\u0142', 'l').replace('\u0141', 'L')
    s = s.replace('\u00f8', 'o').replace('\u00d8', 'O')
    return re.sub(r'[^a-z ]', ' ', s.lower()).strip()


def judge(answer, acceptable):
    """True if the stated answer is factually right.

    An empty acceptable set means the entity is invented, so anything asserted
    is a fabrication — UNLESS the model refuses, which is the correct behavior
    and must not be scored as an error.
    """
    a = _norm(answer)
    if not a:
        return None
    # Refusal detection must be generous. "Brashaland does not have an
    # official currency" was scored WRONG by an earlier list that contained
    # 'do not have' but not 'does not have' — turning a correct refusal into a
    # fabrication and inflating the error count by 100% of its true value.
    refusal = any(w in a for w in (
        'not a real', 'fictional', 'does not exist', 'no such', 'unknown',
        'not aware', 'cannot', 'unable', 'no country', 'not exist',
        'appears to be', 'does not have', 'do not have', 'dont have',
        'i do not', 'there is no', 'not recognized', 'no official',
        'hypothetical', 'made up', 'made-up', 'invented', 'imaginary'))
    if not acceptable:
        return None if refusal else False       # refusing is not an error
    if refusal:
        return None
    return any(_norm(t) in a or a in _norm(t) for t in acceptable)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=14)
    ap.add_argument('--limit-gb', type=float, default=60.0)
    a = ap.parse_args()

    if a.capture:
        return capture(a)

    d = json.loads(CAP.read_text())
    rows = [r for r in d['rows'] if r['correct'] is not None]
    E = d['n_experts']
    y = np.array([0 if r['correct'] else 1 for r in rows])       # 1 = ERROR
    ent = np.array([r['entropy'] for r in rows])
    top1 = np.array([r['top1_prob'] for r in rows])
    X = np.array([r['experts'] for r in rows], dtype=np.int64)
    ents_name = np.array([r['entity'] for r in rows])

    from collections import Counter
    print(f"{d['model']} · {len(rows)} scorable answers "
          f"({int(y.sum())} errors, {float(y.mean()):.1%})")
    for lv in ('known', 'obscure', 'fictional'):
        m = np.array([r['level'] == lv for r in rows])
        if m.any():
            print(f"  {lv:10s} n={int(m.sum()):3d}  errors {int(y[m].sum()):3d}"
                  f"  ({float(y[m].mean()):.1%})")
    skipped = len(d['rows']) - len(rows)
    print(f"  ({skipped} unscorable — refusals or empty answers)\n")

    from .detect import auc
    from .meter import counts, score

    # routing: leave-one-entity-out, predict error directly
    pred = np.zeros(len(y))
    for e in np.unique(ents_name):
        te, tr = ents_name == e, ents_name != e
        if len(np.unique(y[tr])) < 2:
            continue
        C = counts(X[tr], y[tr], E, n_cls=2)
        sc = score(C, X[te])
        pred[te] = sc[:, 1] - sc[:, 0]

    res = {'n': len(y), 'n_errors': int(y.sum()),
           'entropy': round(auc(ent[y == 1], ent[y == 0]), 4),
           'neg_top1': round(auc(-top1[y == 1], -top1[y == 0]), 4),
           'routing': round(auc(pred[y == 1], pred[y == 0]), 4)}
    print(f"  {'signal':14s} {'AUC vs error':>13s}")
    for k in ('entropy', 'neg_top1', 'routing'):
        print(f"  {k:14s} {res[k]:13.4f}")

    # the quadrant that matters: confident answers only
    conf = ent <= np.median(ent)
    print(f"\n  CONFIDENT half only (entropy <= median), where entropy is blind:")
    for lbl, m in (('confident', conf), ('uncertain', ~conf)):
        if len(np.unique(y[m])) < 2:
            print(f"    {lbl:10s} n={int(m.sum())}, errors "
                  f"{int(y[m].sum())} — not scorable")
            continue
        r = {'entropy': auc(ent[m & (y == 1)], ent[m & (y == 0)]),
             'routing': auc(pred[m & (y == 1)], pred[m & (y == 0)])}
        print(f"    {lbl:10s} n={int(m.sum()):3d} errors {int(y[m].sum()):3d}"
              f"   entropy {r['entropy']:.3f}   routing {r['routing']:.3f}")
        res[f'{lbl}_entropy'] = round(r['entropy'], 4)
        res[f'{lbl}_routing'] = round(r['routing'], 4)

    # obscure only: real entities, model confident, sometimes wrong
    m = np.array([r['level'] == 'obscure' for r in rows])
    if m.any() and len(np.unique(y[m])) > 1:
        print(f"\n  OBSCURE only (real entities, confidently wrong is the risk):")
        print(f"    n={int(m.sum())} errors {int(y[m].sum())}   "
              f"entropy {auc(ent[m & (y == 1)], ent[m & (y == 0)]):.3f}   "
              f"routing {auc(pred[m & (y == 1)], pred[m & (y == 0)]):.3f}")
        res['obscure_entropy'] = round(
            auc(ent[m & (y == 1)], ent[m & (y == 0)]), 4)
        res['obscure_routing'] = round(
            auc(pred[m & (y == 1)], pred[m & (y == 0)]), 4)

    dest = OUT / 'errors.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  AUC 0.5 = chance. Higher = better at flagging a false answer.")
    print(f"\n  → {dest}")


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate
    from .seam import find_gates, gate_output, describe

    print("loading …", flush=True)
    model, tok = load(a.model)
    n_moe, E, _tk = describe(model)
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
                    rk, _s, _k = gate_output(out, a.top_k)
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
        for level, ent, rels in ENTITIES:
            for rel, acceptable in rels.items():
                q = QUESTIONS[rel].format(ent)
                msg = [{"role": "user", "content": q}]
                try:
                    pr = tok.apply_chat_template(
                        msg, add_generation_prompt=True, tokenize=False,
                        enable_thinking=False)
                except TypeError:
                    pr = tok.apply_chat_template(
                        msg, add_generation_prompt=True, tokenize=False)
                ans = generate(model, tok, prompt=pr,
                               max_tokens=a.max_tokens, verbose=False)
                ids = tok.encode(pr)
                sink['rows'] = {}
                sink['on'] = True
                lg = model(mx.array([ids])).astype(mx.float32)
                sink['on'] = False
                lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                pv = mx.exp(lp)
                # the decisive position: the last prompt token, whose routing
                # and distribution produce the FIRST answer token
                t = len(ids) - 1
                e = float(-mx.sum(pv[0, t] * lp[0, t]))
                p1 = float(mx.max(pv[0, t]))
                ok = judge(ans, acceptable)
                out.append({'level': level, 'entity': ent, 'relation': rel,
                            'answer': ans.strip()[:60], 'correct': ok,
                            'entropy': e, 'top1_prob': p1,
                            'experts': [sink['rows'][l][t][:a.top_k].tolist()
                                        for l in range(n_moe)]})
                mark = {True: 'ok', False: 'WRONG', None: 'skip'}[ok]
                print(f"  {level:9s} {ent:12s} {rel:8s} -> "
                      f"{ans.strip()[:34]!r:38s} {mark}", flush=True)
    finally:
        for h, n, g in restore:
            setattr(h, n, g)

    CAP.write_text(json.dumps({'model': a.model.rstrip('/').split('/')[-1],
                               'n_layers': n_moe, 'n_experts': E,
                               'rows': out}))
    n_err = sum(1 for r in out if r['correct'] is False)
    print(f"\n  {len(out)} answers · {n_err} wrong · → {CAP}")


if __name__ == '__main__':
    main()
