"""Did the retrieved context actually help? — measured without labels.

The RAG diagnostic nobody gets cheaply. Tools like RAGAS score context relevance
by asking **another LLM**, which is slow, expensive, and itself unreliable. If
useful context shows up as a drop in the model's own answer-token entropy, the
same question is answered for the cost of a forward pass you already ran.

The design borrows `probes.build_grounding`, which was built for R2 and already
carries the two controls this needs:

  parametric   {stem}                                  answer from weights only
  contextual   Fact: {other paraphrase} {answer}. {stem}   answer IS in context
  distractor   Fact: {unrelated fact} {its answer}. {stem} context, but useless

`contextual` vs `distractor` is the clean comparison: both prompts carry a
prepended "Fact: ..." sentence of the same shape and similar length, so prompt
length and answer-token POSITION are matched, and only one contains the answer.
The context sentence uses a DIFFERENT paraphrase than the question, so a win is
grounding rather than verbatim span-copying. And the answer token is identical
across all three classes, so answer form cannot leak — the confound that killed
R2's first suite.

WHAT IS THE LABEL, AND WHAT IS THE SIGNAL. Answer NLL is the ground truth for
"did the context help" — it needs the answer, i.e. a label. Entropy at the same
position needs nothing. If entropy tracks NLL across items, retrieval usefulness
is detectable label-free, which is the product claim.

PAIRED per (fact, paraphrase): the same question under two contexts, so
item-to-item variance in base difficulty cancels.

Usage:
  python3 -m knowledge.context --model qwen
  python3 -m knowledge.context --analyse
"""
import argparse
import json
import time

import numpy as np

from .meter import OUT

CTX = OUT / 'context'


def answer_stats(model, tok, stem, answer):
    """(mean answer-token entropy, mean answer NLL, first-token entropy).

    Teacher-forced: the answer is appended and scored in place, so this measures
    the distribution at the positions where the answer is predicted rather than
    whatever the model would have generated. Generation would confound the
    measurement with sampling and with length.
    """
    import mlx.core as mx
    p_ids = tok.encode(stem)
    full = tok.encode(stem + answer)
    a_ids = full[len(p_ids):]
    if not a_ids:
        return None
    # The split point assumes `stem + answer` tokenises to `tok(stem)` followed
    # by the answer's own tokens. That is usually true because every answer here
    # starts with a space, but "usually" is how the leading-space family of bugs
    # bit this repo three times — 0/6 on one suite, and a membership test that
    # scored 0.000 for members AND non-members. A silent boundary shift here
    # would move the measured positions and quietly compare the wrong tokens.
    if full[:len(p_ids)] != p_ids:
        return None
    lg = model(mx.array([full])[:, :-1]).astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    ent = np.asarray((-mx.sum(mx.exp(lp) * lp, axis=-1))[0].tolist())
    tgt = np.asarray(lp[0].tolist())
    # positions predicting the answer tokens
    pos = range(len(p_ids) - 1, len(full) - 1)
    nll = [-tgt[i][full[i + 1]] for i in pos]
    e = [ent[i] for i in pos]
    return float(np.mean(e)), float(np.mean(nll)), float(e[0])


def first_token_stats(model, tok, stem, aliases):
    """Entropy at the decision point, and the NLL of the BEST alias there.

    Fixes the flaw that made the conditional half of F10 unmeasurable on qwen.
    Scoring parametric NLL against `gold[0]` measures whether the model prefers
    one particular surface form, not whether it knows the fact — the task scorer
    accepts ANY alias. qwen's median NLL against `gold[0]` was 16.10, near-zero
    probability on essentially every item, so the "model knew it" half of the
    split was not one and the contrast vanished.

    Taking the MINIMUM over aliases asks the question the scorer asks.

    It also costs nothing extra. Entropy at the final prompt position is a
    property of the distribution, not of the target, so it is identical for
    every alias — one forward pass serves them all, and only the cheap CPU-side
    tokenisation repeats. First-token scoring is the right granularity here
    anyway: PopQA answers are entities, the first token largely determines
    which, and R9f already used first-token match for the same reason.
    """
    import mlx.core as mx
    p_ids = tok.encode(stem)
    lg = model(mx.array([p_ids]))[:, -1, :].astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    lpv = np.asarray(lp[0].tolist())
    ent = float(-(np.exp(lpv) * lpv).sum())
    best, which = None, None
    for al in aliases:
        full = tok.encode(stem + ' ' + al)
        # same tokenisation-boundary guard as answer_stats: a shifted split
        # would score the wrong token and do it silently
        if len(full) <= len(p_ids) or full[:len(p_ids)] != p_ids:
            continue
        nll = float(-lpv[full[len(p_ids)]])
        if best is None or nll < best:
            best, which = nll, al
    if best is None:
        return None
    return ent, best, which


def popqa_probes(n, seed):
    """PopQA triples, built to fix what the grounding suite could not measure.

    The first run of this experiment used `probes.build_grounding`, whose facts
    are things like the capital of Australia. qwen already knows those — its
    answer NLL barely moved when handed the answer (contextual − distractor
    = +0.15 nats) — so there was no retrieval benefit for entropy to detect, and
    the prompt FRAME moved entropy more than the content did (+0.68 vs +0.38).
    A utility measure needs items where the model has something to gain.

    PopQA's long tail supplies that: accuracy is 0.23-0.29 here, so most items
    are ones the model cannot answer from weights.

    The context is a Q-A reference block rather than a paraphrased statement,
    because PopQA items carry no subject/relation/object fields to build a
    declarative from — only a question and a list of gold aliases. That makes
    this a measure of **context utilisation**, not of grounding-versus-copying:
    a model handed the literal answer that does not become more confident has a
    problem worth surfacing, whatever mechanism it would have used.
    """
    from .popqa import quantize_suite
    from .stage_a import load_task
    items = load_task('popqa', n, seed)
    out = []
    for i, it in enumerate(items):
        gold = it['gold']
        if isinstance(gold, str):
            try:
                gold = json.loads(gold.replace("'", '"'))
            except Exception:
                gold = [gold]
        if not gold:
            continue
        ans = gold[0]
        other = items[(i + 7) % len(items)]
        og = other['gold']
        if isinstance(og, str):
            try:
                og = json.loads(og.replace("'", '"'))
            except Exception:
                og = [og]
        if not og:
            continue
        q = it['prompt']
        for cls, text in (
                ('parametric', f"Q: {q}\nA:"),
                ('contextual',
                 f"Reference: {q} {ans}\n\nQ: {q}\nA:"),
                ('distractor',
                 f"Reference: {other['prompt']} {og[0]}\n\nQ: {q}\nA:")):
            out.append({'probe_id': f'popqa.{i}.{cls[:4]}', 'fact_id': f'pq{i}',
                        'domain': 'popqa', 'cls': cls, 'para': 0,
                        'stem': text, 'answer': ' ' + ans, 'aliases': gold})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--suite', default='grounding',
                    choices=('grounding', 'popqa'))
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--limit-gb', type=float, default=60.0)
    a = ap.parse_args()
    if a.analyse:
        return analyse(a.suite)

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load
    from .frontier import MODELS
    from .probes import build_grounding

    path = MODELS[a.model][0]
    print(f"  loading {a.model} …", flush=True)
    model, tok = load(path)

    probes = (build_grounding() if a.suite == 'grounding'
              else popqa_probes(a.n, a.seed))
    CTX.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, p in enumerate(probes):
        if p.get('aliases'):
            # alias-aware first-token scoring — see first_token_stats
            s = first_token_stats(model, tok, p['stem'], p['aliases'])
            if s is None:
                continue
            row = {'ent': s[0], 'nll': s[1], 'ent_first': s[0],
                   'best_alias': s[2]}
        else:
            s = answer_stats(model, tok, p['stem'], p['answer'])
            if s is None:
                continue
            row = {'ent': s[0], 'nll': s[1], 'ent_first': s[2]}
        rows.append({'fact_id': p['fact_id'], 'para': p['para'],
                     'cls': p['cls'], 'domain': p['domain'], **row})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(probes)}  {time.time() - t0:.0f}s",
                  flush=True)
    dest = CTX / f'{a.model}.{a.suite}.json'
    dest.write_text(json.dumps({'model': a.model, 'suite': a.suite,
                                'n': len(rows),
                                'seconds': round(time.time() - t0, 1),
                                'rows': rows}))
    print(f"\n  {len(rows)} probes · {time.time() - t0:.0f}s → {dest}")


def analyse(suite='grounding'):
    import glob
    # 'analysis' must be excluded from BOTH branches — this function writes its
    # own output into the same directory, so the glob eats it on the next run
    files = sorted(f for f in glob.glob(str(CTX / f'*.{suite}.json'))
                   if 'analysis' not in f)
    if not files:
        files = sorted(f for f in glob.glob(str(CTX / '*.json'))
                       if 'analysis' not in f)
    if not files:
        raise SystemExit("no context captures yet")
    for f in files:
        d = json.loads(open(f).read())
        by = {}
        for r in d['rows']:
            by.setdefault((r['fact_id'], r['para']), {})[r['cls']] = r
        pairs = [v for v in by.values()
                 if {'contextual', 'distractor', 'parametric'} <= set(v)]
        if not pairs:
            print(f"  {d['model']}: no complete triples")
            continue

        def delta(a_cls, b_cls, key):
            return np.array([p[a_cls][key] - p[b_cls][key] for p in pairs])

        print(f"\n=== {d['model']} · {len(pairs)} matched (fact, paraphrase) "
              f"triples ===")
        print(f"  {'comparison':28s} {'Δ NLL (label)':>15s} "
              f"{'Δ entropy (free)':>17s} {'d_z':>7s} {'agree':>7s}")
        out = {}
        for lbl, a_cls, b_cls in (
                ('contextual − distractor', 'contextual', 'distractor'),
                ('parametric − distractor', 'parametric', 'distractor')):
            dn, de = delta(a_cls, b_cls, 'nll'), delta(a_cls, b_cls, 'ent')
            dz = float(de.mean() / (de.std(ddof=1) + 1e-12))
            # per-item agreement in SIGN: does the free signal move the same way
            # as the labelled one on the same item?
            agree = float(np.mean(np.sign(dn) == np.sign(de)))
            print(f"  {lbl:28s} {dn.mean():+15.3f} {de.mean():+17.3f} "
                  f"{dz:+7.2f} {agree:7.0%}")
            out[lbl] = {'d_nll': round(float(dn.mean()), 4),
                        'd_ent': round(float(de.mean()), 4),
                        'd_z': round(dz, 3), 'sign_agreement': round(agree, 4)}
        # does entropy RANK the items by how much the context helped?
        dn = delta('contextual', 'distractor', 'nll')
        de = delta('contextual', 'distractor', 'ent')
        r = float(np.corrcoef(dn, de)[0, 1])
        print(f"\n  Pearson(Δ NLL, Δ entropy) = {r:+.3f}  — entropy tracking "
              f"the labelled\n  measure across items is what makes retrieval "
              f"usefulness detectable without answers.")
        print(f"  `parametric − distractor` is the CONTROL: both lack the "
              f"answer in context,\n  so a large shift there would mean the "
              f"signal is reading prompt length, not grounding.")
        out['pearson_dnll_dent'] = round(r, 4)

        # THE CONTROL THE FIRST VERSION LACKED. Context can only help on items
        # the model does not already know, so pooling over both kinds dilutes
        # the effect to nothing — which is exactly how the grounding run failed
        # on qwen. `parametric` NLL is a label-free proxy for "did it know
        # this": high means it did not.
        pn = np.array([p['parametric']['nll'] for p in pairs])
        med = float(np.median(pn))
        print(f"\n  split on parametric NLL (median {med:.2f}) — context can "
              f"only help where\n  the model did NOT already know the answer:")
        print(f"  {'items':22s} {'n':>4s} {'Δ NLL':>9s} {'Δ entropy':>11s} "
              f"{'d_z':>7s}")
        for lbl, m in (('model KNEW it', pn <= med),
                       ('model did NOT know', pn > med)):
            if m.sum() < 5:
                continue
            dn2 = dn[m]
            de2 = de[m]
            dz2 = float(de2.mean() / (de2.std(ddof=1) + 1e-12))
            print(f"  {lbl:22s} {int(m.sum()):4d} {dn2.mean():+9.3f} "
                  f"{de2.mean():+11.3f} {dz2:+7.2f}")
            out[f'split::{lbl}'] = {
                'n': int(m.sum()), 'd_nll': round(float(dn2.mean()), 4),
                'd_ent': round(float(de2.mean()), 4), 'd_z': round(dz2, 3)}
        rp = float(np.corrcoef(pn, de)[0, 1])
        print(f"\n  Pearson(parametric NLL, Δ entropy) = {rp:+.3f} — NEGATIVE "
              f"is the prediction:\n  the less the model knew, the more "
              f"relevant context should reduce entropy.")
        out['pearson_paramnll_dent'] = round(rp, 4)

        dest = CTX / f'analysis.{d["model"]}.{d.get("suite", "grounding")}.json'
        dest.write_text(json.dumps(out, indent=1))
        print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
