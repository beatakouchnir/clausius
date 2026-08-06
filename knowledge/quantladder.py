"""Is the entropy flag on bf16 -> 4-bit a real regression, or a false positive?

F8d measured the quantization ladder against a **4-bit** reference, as did all
thirteen benign configurations behind the ±0.10 null. Nothing in the corpus had
ever measured an *unquantized* reference, so when `clausius compare` flagged
bf16 -> 4-bit at d_z +0.654 there was no way to tell a genuine detection from a
false alarm — entropy reports that something moved, never how much accuracy fell.

This scores the same checkpoints against gold answers so the question can be
settled. It is validation tooling, not product: the detector is label-free, and
labels exist here only to judge it.

Two subcommands:

    python -m knowledge.quantladder score --model PATH --tag q4 --n 500
    python -m knowledge.quantladder analyze

`score` is resumable via --start, because the arms that carry the question need
the whole gsm8k test split and re-running an already-scored block costs hours.
Row indices are absolute in the shuffled set, so shards merge without collision
and `analyze` unions every shard it finds for an arm.

`analyze` reports the discordant counts b and c beside p, deliberately. With
base accuracy near 0.85 the discordant pairs are few, and a null result at small
n is far more often underpowered than evidence of no effect — this experiment
read p=0.135 at n=500 and p=0.0076 at n=1319 on the *same* effect. Printing b
and c lets that be judged rather than assumed.
"""
import argparse
import json
import re
import time
from math import comb
from pathlib import Path

RECORDS = Path(__file__).resolve().parent.parent / 'records' / 'quantladder'
ARMS = ('bf16', 'q8', 'q4', 'q3', 'q2')
NUM = re.compile(r'-?\d[\d,]*\.?\d*')


def gold_of(answer: str) -> str:
    """gsm8k gold is the text after the final '####'."""
    return answer.split('####')[-1].strip().replace(',', '')


def predicted_of(text: str):
    """Last number in the generation.

    gsm8k answers are integers and instruct models close with the result, so the
    final number is the standard extraction. Returns None when the model emitted
    no number at all — which is an outcome for a damaged checkpoint, not an
    error: 2-bit emits none on most items and must score wrong rather than crash.
    """
    hits = NUM.findall(text)
    return hits[-1].replace(',', '').rstrip('.') if hits else None


def same(pred, gold) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def exact_two_sided(b: int, c: int) -> float:
    """Exact binomial McNemar.

    The chi-square form is unreliable when b+c is small, and b+c is small here
    by construction — two checkpoints of one model disagree on few items.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def cmd_score(a):
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset('openai/gsm8k', 'main', split='test').shuffle(seed=a.seed)
    lo, hi = a.start, min(a.start + a.n, len(ds))
    ds = ds.select(range(lo, hi))

    model, tok = load(a.model)
    rows, t0 = [], time.time()
    for i in range(len(ds)):
        q, ans = ds[i]['question'], ds[i]['answer']
        try:
            rendered = tok.apply_chat_template(
                [{'role': 'user', 'content': q}], add_generation_prompt=True,
                tokenize=False)
        except Exception:
            rendered = q
        text = generate(model, tok, prompt=rendered, max_tokens=a.max_tokens,
                        verbose=False)
        gold, pred = gold_of(ans), predicted_of(text)
        rows.append({'i': lo + i, 'gold': gold, 'pred': pred,
                     'correct': same(pred, gold), 'chars': len(text)})
        if (i + 1) % 25 == 0:
            acc = sum(r['correct'] for r in rows) / len(rows)
            print(f'  {i + 1}/{len(ds)}  running acc {acc:.3f}', flush=True)

    acc = sum(r['correct'] for r in rows) / len(rows)
    RECORDS.mkdir(parents=True, exist_ok=True)
    out = RECORDS / f'acc.{a.tag}{"" if a.start == 0 else f".{a.start}"}.json'
    out.write_text(json.dumps({
        'model': a.model, 'tag': a.tag, 'start': lo, 'n': len(rows),
        'accuracy': acc, 'max_tokens': a.max_tokens, 'seed': a.seed,
        'seconds': round(time.time() - t0, 1), 'rows': rows}))
    print(f'{a.tag}: accuracy {acc:.4f} on {len(rows)} items -> {out}')


def load_arm(tag):
    """Union every shard for an arm, keyed by absolute shuffled index."""
    parts = sorted(RECORDS.glob(f'acc.{tag}.json')) + \
        sorted(RECORDS.glob(f'acc.{tag}.[0-9]*.json'))
    if not parts:
        return None
    by_i, model = {}, None
    for p in parts:
        d = json.loads(p.read_text())
        model = d['model']
        for r in d['rows']:
            by_i[r['i']] = r['correct']
    return model, by_i


def cmd_analyse(a):
    arms = {t: got for t in ARMS if (got := load_arm(t))}
    if a.ref not in arms:
        raise SystemExit(f'{a.ref} arm missing from {RECORDS}')

    print(f"{'arm':>6} {'n':>6} {'accuracy':>9}")
    for tag, (_, by_i) in arms.items():
        print(f'{tag:>6} {len(by_i):>6} {sum(by_i.values()) / len(by_i):>9.4f}')
    print()

    _, ref = arms[a.ref]
    print(f"{'comparison':>16} {'n_pair':>7} {'acc_ref':>8} {'acc_arm':>8} "
          f"{'delta':>8} {'b':>4} {'c':>4} {'p':>8}")
    for tag, (_, cand) in arms.items():
        if tag == a.ref:
            continue
        shared = sorted(set(ref) & set(cand))
        n = len(shared)
        b = sum(1 for i in shared if ref[i] and not cand[i])
        c = sum(1 for i in shared if cand[i] and not ref[i])
        ra = sum(ref[i] for i in shared) / n
        ca = sum(cand[i] for i in shared) / n
        print(f'{a.ref + " -> " + tag:>16} {n:>7} {ra:>8.4f} {ca:>8.4f} '
              f'{ca - ra:>+8.4f} {b:>4} {c:>4} {exact_two_sided(b, c):>8.4f}')
    print()
    print(f'b = {a.ref} correct, arm wrong (a regression).  c = the reverse.')
    print('p is an exact two-sided binomial McNemar over the b+c discordant '
          'pairs.')


def main():
    ap = argparse.ArgumentParser(prog='knowledge.quantladder',
                                 description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('score', help='labeled gsm8k accuracy for one checkpoint')
    s.add_argument('--model', required=True, help='path or HF id')
    s.add_argument('--tag', required=True, choices=ARMS)
    s.add_argument('--n', type=int, default=500)
    s.add_argument('--start', type=int, default=0,
                   help='skip the first N shuffled items, so an existing run '
                        'can be extended rather than recomputed')
    s.add_argument('--max-tokens', type=int, default=1024)
    s.add_argument('--seed', type=int, default=0)

    d = sub.add_parser('analyze', help='paired McNemar across the arms')
    d.add_argument('--ref', default='bf16', choices=ARMS)

    a = ap.parse_args()
    {'score': cmd_score, 'analyze': cmd_analyse}[a.cmd](a)


if __name__ == '__main__':
    main()
