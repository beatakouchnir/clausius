"""Stage B — entropy vs self-consistency, the honest competitor.

Stage A showed `p90` of per-token entropy predicts error at 0.76-0.92 across
task types, beating the generation-length baseline everywhere. But the baseline
that matters commercially is not length — it is **self-consistency**: sample k
times, measure how much the answers agree. That is what production systems
actually use, and it is the one comparison R13 never made.

**The product claim is not "entropy beats routing". It is "entropy approaches
self-consistency at 1/k the cost."** If self-consistency dominates outright,
the answer is to use self-consistency and this project's signal is redundant.

a prior n=740 study measured 8-vote self-consistency on gemma-4-26b-a4b:
**AURC 0.087 on math, 0.259 on MCQ** (n=740). So the incumbent is strong on
math and weak on MCQ, and its `vote_signals` (share / margin / neg_entropy) are
reused here rather than reimplemented, so the numbers stay comparable.

COST IS THE POINT, so it is measured, not assumed. Each item pays:
  1 greedy pass          -> entropy variants (Stage A's signal)
  k sampled generations  -> vote agreement
and the run records wall-clock for each arm separately.

WHY THE SAME ITEMS. Entropy and votes must be scored on identical items with
identical correctness labels, or the comparison is between two different
datasets. The greedy pass supplies both the entropy and the correctness label;
the sampled passes supply only the votes.

Needs mlx-lm. Reuses the vendored loaders/scorers and vote_signals.

Usage:
  python3 -m knowledge.stage_b --task popqa --n 200 --k 5
  python3 -m knowledge.stage_b --analyze
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import traces
from .cot import variants
from .meter import OUT
from .popqa import task_suite
from .stage_a import CAPS, QUANTIZE_TASKS, load_task, score_item


def answer_key(task, item, text):
    """Canonical form of an answer, for vote counting.

    Votes must be counted over a NORMALIZED answer, not raw text: two samples
    that both say "Paris" but differ in punctuation or preamble are the same
    vote, and counting them as different would understate agreement and make
    self-consistency look artificially weak — i.e. bias the comparison in this
    project's favor.
    """
    if item.get('score') == 'letter':
        m = re.findall(r'ANSWER:\s*([A-Za-z])', text)
        if m:
            return m[-1].upper()
        hits = re.findall(rf"\b([{item['letters']}])\b", text.upper())
        return hits[-1] if hits else ''
    if item.get('score') == 'number':
        m = re.findall(r'ANSWER:\s*([^\n]+)', text)
        tail = m[-1] if m else text.strip()[-40:]
        nums = re.findall(r'-?\d[\d,]*\.?\d*', tail.replace(',', ''))
        return nums[-1].rstrip('.') if nums else ''
    from .stage_a import _norm
    return ' '.join(_norm(text).split()[:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--task', default='popqa')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if a.analyze:
        return analyze()
    capture(a)


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    suite = task_suite()
    items = load_task(a.task, a.n, a.seed)
    print(f"{len(items)} {a.task} · k={a.k} @ temp {a.temp} · "
          f"cap {CAPS.get(a.task)}", flush=True)
    print("loading …", flush=True)
    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp)

    out, t_greedy, t_votes = [], 0.0, 0.0
    for i, it in enumerate(items):
        pr = (suite.build_prompt(tok, it, think=False)
              if a.task in QUANTIZE_TASKS else _prompt(tok, it))
        cap = it.get('max_tokens', 512)

        t0 = time.time()
        text = generate(model, tok, prompt=pr, max_tokens=cap, verbose=False)
        ok, abstained = score_item(a.task, it, text, suite)
        ids = tok.encode(pr + text)
        n_prompt = len(tok.encode(pr))
        lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        pv = mx.exp(lp)
        ent = np.asarray((-mx.sum(pv * lp, axis=-1))[0].tolist())
        t_greedy += time.time() - t0

        t0 = time.time()
        votes = Counter()
        for _ in range(a.k):
            s = generate(model, tok, prompt=pr, max_tokens=cap,
                         sampler=sampler, verbose=False)
            key = answer_key(a.task, it, s)
            if key:
                votes[key] += 1
        t_votes += time.time() - t0

        from . import _gl
        vs = _gl.vote_signals(dict(votes))
        greedy_key = answer_key(a.task, it, text)
        out.append({
            'correct': ok, 'abstained': abstained,
            'truncated': (len(ids) - n_prompt) >= cap - 2,
            'ent': variants(ent, n_prompt, None),
            'votes': dict(votes), 'vote_share': vs['share'],
            'vote_margin': vs['margin'], 'vote_negent': vs['neg_entropy'],
            # does the greedy answer agree with the sampled plurality?
            'greedy_in_plurality': bool(
                votes and greedy_key == votes.most_common(1)[0][0])})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)}  greedy {t_greedy:.0f}s  "
                  f"votes {t_votes:.0f}s ({t_votes / max(t_greedy, 1e-9):.1f}x)",
                  flush=True)

    tag = a.model.rstrip('/').split('/')[-1]
    dest = OUT / f'stage_b.{a.task}.{tag}.json'
    dest.write_text(json.dumps({'model': tag, 'task': a.task, 'k': a.k,
                                'temp': a.temp, 'sec_greedy': t_greedy,
                                'sec_votes': t_votes, 'rows': out}))
    print(f"\n  greedy {t_greedy:.0f}s · votes {t_votes:.0f}s "
          f"({t_votes / max(t_greedy, 1e-9):.1f}x cost) → {dest}")


def _prompt(tok, item):
    content = f"{item['prompt']}\n\n{item['instruct']}"
    msg = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False)


def analyze():
    from .detect import auc
    files = sorted(OUT.glob('stage_b.*.json'))
    if not files:
        raise SystemExit("no stage_b captures yet")
    res = {}
    print(f"  {'task':12s} {'model':20s} {'n':>4s} {'errs':>5s} | "
          f"{'p90 ent':>8s} {'v-share':>8s} {'v-margin':>9s} {'v-negent':>9s} "
          f"{'combined':>9s} | {'cost':>6s}")
    for f in files:
        d = json.loads(f.read_text())
        rows = [r for r in d['rows']
                if not r['abstained'] and not r['truncated']]
        if len(rows) < 20:
            continue
        y = np.array([0 if r['correct'] else 1 for r in rows])
        if len(np.unique(y)) < 2:
            continue
        sig = {'p90': np.array([r['ent']['p90'] for r in rows]),
               'share': -np.array([r['vote_share'] for r in rows]),
               'margin': -np.array([r['vote_margin'] for r in rows]),
               'negent': -np.array([r['vote_negent'] for r in rows])}
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)   # noqa: E731
        sig['combined'] = z(sig['p90']) + z(sig['share'])
        got = {k: auc(v[y == 1], v[y == 0]) for k, v in sig.items()}
        cost = d['sec_votes'] / max(d['sec_greedy'], 1e-9)
        print(f"  {d['task']:12s} {d['model'][:20]:20s} {len(rows):4d} "
              f"{int(y.sum()):5d} | {got['p90']:8.3f} {got['share']:8.3f} "
              f"{got['margin']:9.3f} {got['negent']:9.3f} "
              f"{got['combined']:9.3f} | {cost:5.1f}x")
        res[f"{d['task']}/{d['model']}"] = {
            'n': len(rows), 'n_errors': int(y.sum()), 'k': d['k'],
            'auc': {k: round(v, 4) for k, v in got.items()},
            'vote_cost_multiple': round(cost, 2)}
    (OUT / 'stage_b.json').write_text(json.dumps(res, indent=1))
    print(f"\n  Bar: entropy is competitive if within 0.05 AUC of the best "
          f"vote signal\n  at 1/k the cost. Vote signals are negated so higher "
          f"= more error-like.")
    print(f"\n  → {OUT / 'stage_b.json'}")


if __name__ == '__main__':
    main()
