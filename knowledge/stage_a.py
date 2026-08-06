"""Stage A — does entropy-based error prediction generalize across task types?

R13 measured AUC 0.892 on PopQA/qwen-MoE. That is one benchmark and one model.
R14 then showed the result is fragile in two specific ways, both corrected here:

  MEASUREMENT POSITION. Entropy at the last prompt token works for short-answer
  recall (the next token IS the answer) and fails on chain-of-thought — 0.447 on
  GSM8K, below chance. Answer-token entropy fails too (0.419): by the time the
  chain is written the model has committed, so a confident-but-wrong chain ends
  in a near-deterministic token. `mean` entropy over the generation is the CoT
  measure. Every variant is still recorded and chosen offline.

  TOKEN CAPS. 27/250 GSM8K generations hit the 512-token cap and 81.5% of those
  were wrong — **73% of all "errors" were truncation artifacts**. Excluding them
  moved mean entropy 0.762 -> 0.897 and collapsed the generation-length baseline
  0.878 -> 0.585. the vendored own suite shows the same at scale: mmlu_pro/qwen
  0.345 -> 0.820 at cap4160. So every task here runs at the GENEROUS cap, the
  truncation rate is reported, and capped items are excluded from the AUC.

TASKS ARE SIZED BY EXPECTED ERROR COUNT, not item count. Uncapped GSM8K leaves
~8 errors in 250 items, which cannot support an AUC. The high-error tasks carry
the experiment:

  omniscience  AA-Omniscience-Public, 600 open questions, purpose-built for
               hallucination vs abstention (+1 correct / -1 wrong / 0 abstain).
               The closest published benchmark to the product question.
  popqa        long-tail recall, 75% error rate measured
  mmlu_pro     CoT multiple choice AT cap4160, not the default cap
  gsm8k        CoT math at cap960 — kept as a LOW-error control, not a primary

ABSTENTION IS A THIRD OUTCOME, not an error. AA-Omniscience scores it zero for
exactly this reason, and R13's predecessor mis-scored refusals as fabrications.
Abstentions are recorded separately and excluded from the error label.

Needs mlx-lm and datasets. Reuses the vendored loaders/scorers where they exist.

Usage:
  python3 -m knowledge.stage_a --task omniscience --n 250
  python3 -m knowledge.stage_a --analyze
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import traces
from .cot import variants
from .meter import OUT
from .popqa import task_suite

CAPS = {'gsm8k': 960, 'mmlu_pro': 4160, 'popqa': 64,
        'omniscience': 256, 'hle': 512, 'gpqa': 4160, 'ifeval': 768}
# ifeval added for the agentic axis: instruction adherence is what an agent
# actually depends on (emit this format, use only these fields, stop here), and
# it degrades independently of the factual accuracy the other tasks measure.
# Its gold is a set of programmatic constraint checks rather than an answer
# string, so it MUST route to the vendored scorer — scoring it here would be a
# re-implementation of the official IFEval constraint library.
QUANTIZE_TASKS = ('gsm8k', 'mmlu_pro', 'popqa', 'longbench', 'ifeval')

ABSTAIN = ('i do not know', 'i don\'t know', 'not sure', 'cannot determine',
           'unable to determine', 'no information', 'not enough information',
           'cannot answer', 'unknown', 'i am not aware', 'insufficient')


def _norm(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', ' ', s.lower()).strip()


def load_task(task, n, seed):
    """[(item)] with a uniform shape. the vendored loaders where they exist."""
    if task in QUANTIZE_TASKS:
        items = task_suite().load_items(task, n, seed)
        for it in items:
            it['max_tokens'] = CAPS.get(task, it.get('max_tokens', 512))
        return items
    from datasets import load_dataset
    if task == 'omniscience':
        ds = load_dataset('ArtificialAnalysis/AA-Omniscience-Public',
                          split='train').shuffle(seed=seed)
        return [{'prompt': r['question'],
                 'instruct': "Answer concisely. If you do not know, reply "
                             "exactly: I do not know.",
                 'gold': [r['answer']], 'score': 'contains',
                 'domain': r['domain'], 'max_tokens': CAPS['omniscience']}
                for r in ds.select(range(min(n, len(ds))))]
    if task == 'gpqa':
        # GPQA Diamond, the HLE substitute: `cais/hle` is GATED on the Hub, so
        # it cannot be loaded here. GPQA is ungated, graduate-level science, and
        # reasoning-heavy — hard on the MERITS rather than through truncation,
        # which is what the CoT arm needs. Options are shuffled per item with a
        # per-item seed so the correct letter is not always the same position.
        import random
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')
        out = []
        for i, r in enumerate(ds.select(range(min(n, len(ds))))):
            opts = [r['Correct Answer'], r['Incorrect Answer 1'],
                    r['Incorrect Answer 2'], r['Incorrect Answer 3']]
            rng = random.Random(seed * 10000 + i)
            order = list(range(4))
            rng.shuffle(order)
            letters = 'ABCD'
            gold = letters[order.index(0)]
            body = "\n".join(f"{letters[j]}. {opts[order[j]]}"
                              for j in range(4))
            out.append({'prompt': f"{r['Question']}\n{body}",
                        'instruct': "Reason briefly, then end with the answer "
                                    "letter on its own line as: ANSWER: <letter>",
                        'gold': gold, 'score': 'letter', 'letters': letters,
                        'domain': r.get('Subdomain', '?'),
                        'max_tokens': CAPS['gpqa']})
        return out
    if task == 'hle':
        ds = load_dataset('cais/hle', split='test').shuffle(seed=seed)
        rows = [r for r in ds if not r.get('image')]      # text-only
        return [{'prompt': r['question'],
                 'instruct': "Answer concisely. If you do not know, reply "
                             "exactly: I do not know.",
                 'gold': [r['answer']], 'score': 'contains',
                 'domain': r.get('category', '?'), 'max_tokens': CAPS['hle']}
                for r in rows[:n]]
    raise SystemExit(f"unknown task {task}")


def score_item(task, item, text, suite):
    """(correct, abstained). Abstention is NOT an error — see module docstring."""
    t = _norm(text)
    if any(w in t for w in (_norm(x) for x in ABSTAIN)):
        return None, True
    if task in QUANTIZE_TASKS or item.get('score') == 'letter':
        # reuse the vendored letter scorer for GPQA too — it already handles the
        # ANSWER: marker and the bare-letter fallback
        v = suite.score(item, text)
        if v is None:
            # scorer gave no verdict (ifeval registry absent). NOT a wrong
            # answer: bool(None) is False, and recording that would report an
            # unscoreable arm as near-zero accuracy — damage that never happened.
            return None, True
        return bool(v), False
    # contains-scoring for the open short-answer sets
    return any(_norm(g) and _norm(g) in t for g in item['gold']), False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--task', default='omniscience')
    ap.add_argument('--n', type=int, default=250)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cap', type=int, default=0,
                    help='override the task token cap. GPQA truncated 24%% even '
                         'at 4160, and truncated items are excluded, which '
                         'biases the surviving set toward easier questions.')
    a = ap.parse_args()

    if a.cap:
        CAPS[a.task] = a.cap

    if a.analyze:
        return analyze()
    capture(a)


def cap_path(model, task, cap_override=0):
    """Output path. The token cap is part of the name when overridden, so a
    cap8192 rerun does not silently replace the cap4160 result — keeping both
    is what makes the truncation effect visible rather than lost."""
    tag = model.rstrip('/').split('/')[-1]
    t = f'{task}-cap{cap_override}' if cap_override else task
    return OUT / f'stage_a.{t}.{tag}.json'


def capture(a):
    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate

    suite = task_suite()
    items = load_task(a.task, a.n, a.seed)
    print(f"{len(items)} {a.task} items · cap {CAPS.get(a.task)} tokens",
          flush=True)
    print("loading …", flush=True)
    model, tok = load(a.model)

    out = []
    for i, it in enumerate(items):
        pr = (suite.build_prompt(tok, it, think=False)
              if a.task in QUANTIZE_TASKS else _prompt(tok, it))
        cap = it.get('max_tokens', 512)
        text = generate(model, tok, prompt=pr, max_tokens=cap, verbose=False)
        ok, abstained = score_item(a.task, it, text, suite)

        ids = tok.encode(pr + text)
        n_prompt = len(tok.encode(pr))
        n_gen = len(ids) - n_prompt
        lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        pv = mx.exp(lp)
        ent = np.asarray((-mx.sum(pv * lp, axis=-1))[0].tolist())

        out.append({'correct': ok, 'abstained': abstained,
                    'truncated': n_gen >= cap - 2,
                    'domain': it.get('domain', '?'),
                    'n_gen': n_gen, 'ent': variants(ent, n_prompt, None),
                    'answer': text.strip()[:80]})
        if (i + 1) % 25 == 0:
            sc = [r for r in out if not r['abstained']]
            acc = np.mean([r['correct'] for r in sc]) if sc else float('nan')
            print(f"  {i + 1}/{len(items)}  accuracy {acc:.3f}  "
                  f"abstain {np.mean([r['abstained'] for r in out]):.1%}  "
                  f"trunc {np.mean([r['truncated'] for r in out]):.1%}",
                  flush=True)

    dest = cap_path(a.model, a.task, getattr(a, 'cap', 0))
    dest.write_text(json.dumps({'model': a.model.rstrip('/').split('/')[-1],
                                'task': a.task, 'cap': CAPS.get(a.task),
                                'rows': out}))
    sc = [r for r in out if not r['abstained']]
    print(f"\n  {len(out)} items · {len(sc)} scored · "
          f"accuracy {np.mean([r['correct'] for r in sc]):.3f} · "
          f"abstained {np.mean([r['abstained'] for r in out]):.1%} · "
          f"truncated {np.mean([r['truncated'] for r in out]):.1%}")
    print(f"  → {dest}")


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
    files = sorted(OUT.glob('stage_a.*.json'))
    if not files:
        raise SystemExit("no stage_a captures yet")
    print(f"  {'task':13s} {'model':22s} {'n':>4s} {'acc':>6s} {'abst':>6s} "
          f"{'trunc':>6s} {'errs':>5s} | {'first':>6s} {'mean':>6s} "
          f"{'p90':>6s} {'len':>6s}")
    res = {}
    for f in files:
        d = json.loads(f.read_text())
        rows = [r for r in d['rows']
                if not r['abstained'] and not r['truncated']]
        allr = d['rows']
        if len(rows) < 20:
            continue
        y = np.array([0 if r['correct'] else 1 for r in rows])
        if len(np.unique(y)) < 2:
            continue
        # Bootstrap CIs, because the CoT tasks have few errors once truncation
        # is excluded (gsm8k ~10, mmlu_pro ~27) and a bare AUC there invites
        # over-reading a difference that is noise.
        rng = np.random.default_rng(0)
        cells, cis = {}, {}
        for k in ('first', 'mean', 'p90', 'gen_len'):
            v = np.array([r['ent'][k] for r in rows])
            ok = np.isfinite(v)
            cells[k] = auc(v[ok & (y == 1)], v[ok & (y == 0)])
            boots = []
            idx = np.where(ok)[0]
            for _ in range(400):
                b = rng.choice(idx, len(idx), replace=True)
                if len(np.unique(y[b])) < 2:
                    continue
                boots.append(auc(v[b][y[b] == 1], v[b][y[b] == 0]))
            cis[k] = (float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))) if boots else (0, 0)
        label = f"{d['task']}/{d.get('cap', '?')}"
        print(f"  {label:13s} {d['model'][:22]:22s} {len(rows):4d} "
              f"{1 - y.mean():6.3f} "
              f"{np.mean([r['abstained'] for r in allr]):6.1%} "
              f"{np.mean([r['truncated'] for r in allr]):6.1%} "
              f"{int(y.sum()):5d} | " +
              ' '.join(f"{cells[k]:6.3f}" for k in
                       ('first', 'mean', 'p90', 'gen_len')))
        print(f"  {'':13s} {'95% CI':22s} {'':4s} {'':6s} {'':6s} {'':6s} "
              f"{'':5s} | " + ' '.join(
                  f"{cis[k][0]:.2f}-{cis[k][1]:.2f}" for k in
                  ('first', 'mean', 'p90', 'gen_len')))
        res[f"{d['task']}/{d['model']}"] = {
            'n': len(rows), 'accuracy': round(float(1 - y.mean()), 4),
            'n_errors': int(y.sum()),
            'abstain_rate': round(float(np.mean([r['abstained']
                                                 for r in allr])), 4),
            'truncation_rate': round(float(np.mean([r['truncated']
                                                    for r in allr])), 4),
            'auc': {k: round(v, 4) for k, v in cells.items()},
            'auc_ci95': {k: [round(c, 4) for c in cis[k]] for k in cis}}
    (OUT / 'stage_a.json').write_text(json.dumps(res, indent=1))
    print(f"\n  Abstentions and truncated generations are EXCLUDED from the "
          f"AUC.\n  'len' is the dumb baseline every signal must beat.")
    print(f"\n  → {OUT / 'stage_a.json'}")


if __name__ == '__main__':
    main()
