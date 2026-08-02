"""F10, redesigned — did the retrieved context actually help THIS query?

Two previous attempts failed, both for identifiable design reasons rather than
because the signal is absent. Recorded in FINDINGS F10/F10b; summarised here
because each fault dictates one feature of this design.

  1. WRONG ITEMS. The first attempt used `probes.build_grounding`, whose facts
     (capital of Australia) the model already knows. Handing qwen the answer
     moved answer-NLL by +0.15 nats: there was no retrieval benefit for entropy
     to detect. -> PopQA's long tail, where accuracy is 0.23-0.29 and most items
     are genuinely unknown.

  2. OFF-DISTRIBUTION PROMPTS. The second attempt built raw `Q: ...\\nA:`
     completions and fed them to instruct-tuned models. Median best-alias
     first-token NLL was 10.0 (qwen) and 14.2 (gemma) — near-zero probability on
     the gold answer, for models that score 0.23-0.29 on the task. That gap says
     the measurement was taken in a regime the models do not operate in.
     -> every prompt goes through quantize's `build_prompt`, the same chat
     templating every other measurement in this project uses.

  3. A SINGLE-TOKEN SIGNAL. Alias-aware scoring fixed a real bug but forced
     entropy onto the first answer token only, because different aliases have
     different spans. One token is a very noisy per-item measure, and gemma's
     per-item sign agreement duly sat at 49% — chance. -> entropy is measured
     over the model's OWN GENERATED span, which is both less noisy and exactly
     what the shipped `clausius` tool measures.

WHAT IS COMPARED. Three arms, all chat-templated:

  relevant     Context block that contains the answer.
  irrelevant   Context block of the same shape, drawn from a DIFFERENT item.
               This is the realistic "retrieval missed" case, and it is the
               baseline — it is format-matched to `relevant`, so only content
               differs.
  nocontext    No block at all. Kept for reference and NOT used as the baseline,
               because it differs from the others in format as well as content,
               which is precisely the confound that muddied attempt 2 (the frame
               alone moved entropy more than relevance did).

THE GROUND TRUTH IS CORRECTNESS, not NLL. Whether the generated answer is right
is what a RAG user cares about, it is scored by quantize's official alias
scorer, and it is binary and interpretable. NLL against one arbitrary alias is
what produced a spurious result in attempt 2.

THE CLAIM, STATED SO IT CAN FAIL. The useful question is not "does context help
on average" — it is "did it help on THIS query". So the headline metric is the
AUC of the per-item entropy drop for predicting which items FLIPPED from wrong
to right when the context was added. Chance is 0.5. An aggregate effect with
chance-level per-item AUC would mean the signal is real but useless, which is a
result worth having either way.

SCOPE. PopQA carries no passages, so the context block is a question-answer
reference rather than retrieved prose. This therefore measures context
UTILISATION — does the model use information that is available to it — not
grounding-versus-copying. A model handed the answer that does not become more
confident has a problem worth surfacing whatever mechanism it would have used.

Usage:
  python3 -m knowledge.retrieval --model qwen --n 300
  python3 -m knowledge.retrieval --analyse
"""
import argparse
import json
import time

import numpy as np

from .meter import OUT

RET = OUT / 'retrieval'
ARMS = ('nocontext', 'relevant', 'irrelevant', 'haystack')


def aliases_of(item):
    g = item['gold']
    if isinstance(g, str):
        try:
            g = json.loads(g.replace("'", '"'))
        except Exception:
            g = [g]
    return [a for a in (g or []) if a]


def build(items):
    """[(item_index, arm, item_with_context)] — context lives in the prompt.

    The context is embedded in the item's `prompt` and the whole thing then goes
    through quantize's `build_prompt`, so the chat template wraps it exactly as
    it wraps every other prompt in this project. Building the template by hand
    here is what put attempt 2 off-distribution.
    """
    out = []
    for i, it in enumerate(items):
        al = aliases_of(it)
        other = items[(i + 7) % len(items)]
        oal = aliases_of(other)
        if not al or not oal:
            continue
        q = it['prompt']
        # `haystack` is why this arm exists: a single verbatim question-answer
        # pair makes the task string-copying, and the first run of this design
        # scored 0.973/0.983 with ZERO items broken — no variance left for the
        # per-item claim (H3) to predict. Real top-k retrieval returns one
        # useful passage among several useless ones, so that is what this
        # builds: the answer buried at a deterministic-but-arbitrary position
        # among nine distractors.
        # Distinct distractors, never the target itself. A modular stride is
        # not enough: it silently repeats entries and can re-include item `i`
        # when the pool is small, which would put the answer in the haystack
        # twice and make the arm easier than intended.
        import random
        rng = random.Random(1000 + i)
        cand = [j for j in range(len(items)) if j != i]
        pool = [items[j] for j in rng.sample(cand, min(9, len(cand)))]
        target = f"{q} {al[0]}"
        entries = [f"{p['prompt']} {(aliases_of(p) or ['?'])[0]}" for p in pool]
        # PopQA repeats some questions verbatim, so a "distractor" can be the
        # target line itself. Dropping exact matches keeps the answer present
        # exactly once, which is what the arm is supposed to test.
        entries = [e for e in dict.fromkeys(entries) if e != target]
        entries.insert(rng.randrange(len(entries) + 1), target)
        hay = "\n".join(entries)
        blocks = {
            'nocontext': q,
            'relevant': f"Context:\n{q} {al[0]}\n\n{q}",
            'irrelevant': f"Context:\n{other['prompt']} {oal[0]}\n\n{q}",
            'haystack': f"Context:\n{hay}\n\n{q}",
        }
        for arm in ARMS:
            out.append((i, arm, dict(it, prompt=blocks[arm])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--limit-gb', type=float, default=60.0)
    a = ap.parse_args()
    if a.analyse:
        return analyse()

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import generate, load

    from .cot import variants
    from .frontier import MODELS
    from .popqa import quantize_suite
    from .stage_a import CAPS, load_task, score_item

    suite = quantize_suite()
    items = load_task('popqa', a.n, a.seed)
    work = build(items)
    cap = CAPS['popqa']
    print(f"  loading {a.model} … ({len(work)} runs = {len(items)} items x "
          f"{len(ARMS)} arms)", flush=True)
    model, tok = load(MODELS[a.model][0])

    RET.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for k, (idx, arm, it) in enumerate(work):
        pr = suite.build_prompt(tok, it, think=False)
        text = generate(model, tok, prompt=pr, max_tokens=cap, verbose=False)
        ok, abst = score_item('popqa', it, text, suite)
        ids = tok.encode(pr + text)
        n_prompt = len(tok.encode(pr))
        lg = model(mx.array([ids])[:, :-1]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        ent = np.asarray((-mx.sum(mx.exp(lp) * lp, axis=-1))[0].tolist())
        del lg, lp
        rows.append({'item': idx, 'arm': arm, 'correct': bool(ok),
                     'abstained': bool(abst),
                     'truncated': (len(ids) - n_prompt) >= cap - 2,
                     'ent': variants(ent, min(n_prompt, len(ent)), None)})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(work)}  {time.time() - t0:.0f}s", flush=True)

    dest = RET / f'{a.model}.json'
    dest.write_text(json.dumps({'model': a.model, 'n_items': len(items),
                                'seconds': round(time.time() - t0, 1),
                                'rows': rows}))
    print(f"\n  {len(rows)} runs · {time.time() - t0:.0f}s → {dest}")


def analyse():
    import glob
    from .detect import auc
    files = sorted(f for f in glob.glob(str(RET / '*.json'))
                   if 'analysis' not in f)
    if not files:
        raise SystemExit("no retrieval captures yet")
    for f in files:
        d = json.loads(open(f).read())
        by = {}
        for r in d['rows']:
            by.setdefault(r['item'], {})[r['arm']] = r
        need = {'relevant', 'irrelevant'}
        trip = [v for v in by.values() if need <= set(v)]
        print(f"\n=== {d['model']} · {len(trip)} complete triples ===")

        print(f"  {'arm':12s} {'accuracy':>9s} {'trunc':>6s}")
        for arm in ARMS:
            acc = np.mean([t[arm]['correct'] for t in trip])
            tr = np.mean([t[arm]['truncated'] for t in trip])
            print(f"  {arm:12s} {acc:9.4f} {tr:6.1%}")

        # The comparison arm: `haystack` where present, else `relevant`. A
        # single verbatim pair leaves no variance (0.97-0.98 accuracy, zero
        # items broken), so the realistic arm is the one that can be tested.
        got = trip[0]
        arm_hit = 'haystack' if 'haystack' in got else 'relevant'
        print(f"\n  comparing arm: {arm_hit}  (baseline: irrelevant)")

        # H1 — is there anything to detect?
        helped = [t for t in trip
                  if t[arm_hit]['correct'] and not t['irrelevant']['correct']]
        hurt = [t for t in trip
                if t['irrelevant']['correct'] and not t[arm_hit]['correct']]
        print(f"\n  H1  context FIXED {len(helped)} items, BROKE {len(hurt)} "
              f"(net {len(helped) - len(hurt):+d} of {len(trip)})")
        if len(helped) < 10:
            print("      too few flips to test H3 — the signal has nothing to "
                  "rank.")

        out = {'n': len(trip), 'fixed': len(helped), 'broke': len(hurt)}
        # H2 — does relevant context lower entropy against the FORMAT-MATCHED
        # irrelevant arm? paired, so item difficulty cancels.
        print(f"\n  H2  paired entropy shift, relevant - irrelevant "
              f"(negative = context reduced uncertainty)")
        print(f"      {'signal':12s} {'mean':>9s} {'d_z':>8s}")
        for s in ('max', 'p90', 'mean', 'first'):
            dl = np.array([t[arm_hit]['ent'][s] - t['irrelevant']['ent'][s]
                           for t in trip])
            dz = float(dl.mean() / (dl.std(ddof=1) + 1e-12))
            print(f"      {s:12s} {dl.mean():+9.3f} {dz:+8.2f}")
            out[f'h2_{s}'] = {'mean': round(float(dl.mean()), 4),
                              'd_z': round(dz, 3)}

        # H3 — the product claim. Does the per-item entropy drop identify WHICH
        # items the context fixed? Chance is 0.5.
        if len(helped) >= 10:
            neither = [t for t in trip if not t[arm_hit]['correct']
                       and not t['irrelevant']['correct']]
            print(f"\n  H3  per-item AUC: does the entropy drop pick out the "
                  f"{len(helped)} items\n      context FIXED, against the "
                  f"{len(neither)} it did not? (0.5 = chance)")
            print(f"      {'signal':12s} {'AUC':>7s}")
            for s in ('max', 'p90', 'mean', 'first'):
                pos = [-(t[arm_hit]['ent'][s] - t['irrelevant']['ent'][s])
                       for t in helped]
                neg = [-(t[arm_hit]['ent'][s] - t['irrelevant']['ent'][s])
                       for t in neither]
                if len(neg) < 10:
                    continue
                v = auc(np.array(pos), np.array(neg))
                print(f"      {s:12s} {v:7.3f}")
                out[f'h3_auc_{s}'] = round(float(v), 4)

        dest = RET / f'analysis.{d["model"]}.json'
        dest.write_text(json.dumps(out, indent=1))
        print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
