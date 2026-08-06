"""Command line for clausius. Exits non-zero on a regression, for CI.

    clausius capture --model M --prompts p.jsonl --out ref.json
    clausius capture --model M --prompts p.jsonl --out cand.json --adapter ./lora
    clausius compare ref.json cand.json
"""
import argparse
import json
import sys
from pathlib import Path

from .core import (BACKENDS, DEFAULT_SIGNAL, DEFAULT_THRESHOLD, SIGNALS, capture,
                   compare, top_movers, truncation_curve)


def read_prompts(path):
    """One prompt per line: JSONL with a "prompt" field, or plain text.

    Both are accepted because the realistic source is a sample of production
    traffic, which is usually a log rather than a curated dataset.
    """
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    out = []
    for line in lines:
        if line.lstrip().startswith('{'):
            d = json.loads(line)
            out.append(d.get('prompt') or d.get('text') or d.get('input'))
        else:
            out.append(line)
    missing = [i for i, p in enumerate(out) if not p]
    if missing:
        raise SystemExit(
            f"{len(missing)} line(s) had no prompt field (first: line "
            f"{missing[0] + 1}); expected JSONL with 'prompt'/'text'/'input', "
            f"or plain text")
    return out


def _clip(text, width=160):
    """One line, bounded — a terminal diagnostic, not a transcript dump."""
    flat = ' '.join((text or '').split())
    return flat[:width] + ('…' if len(flat) > width else '')


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='clausius', description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('capture', help='record one configuration')
    c.add_argument('--model', required=True, help='path or HF id')
    c.add_argument('--prompts', required=True, help='JSONL or plain text')
    c.add_argument('--out', required=True)
    c.add_argument('--tag', default=None)
    c.add_argument('--adapter', default=None, help='LoRA adapter path')
    c.add_argument('--max-tokens', type=int, default=512,
                   help='generation cap. Capture at the most generous cap you '
                        'can afford: truncated items are dropped at compare '
                        'time, and a capture can be re-analyzed at any TIGHTER '
                        'cap but never at a looser one — the lengths of '
                        'truncated items are not recoverable.')
    c.add_argument('--limit', type=int, default=None,
                   help='use only the first N prompts (default: all of them). '
                        'Sampling is rarely worth it: a full capture at a '
                        'generous --max-tokens is itself the reference, while '
                        'a sampled probe is thrown away. Use it on prompt sets '
                        'large enough that a wrong cap is expensive.')
    c.add_argument('--backend', default='auto', choices=('auto',) + BACKENDS,
                   help="runtime to capture with: 'mlx' (Apple Silicon), "
                        "'transformers' (CUDA/CPU/MPS), or 'auto'. Recorded on "
                        "the capture — entropy from two different runtimes is "
                        "not a like-for-like measurement.")
    c.add_argument('--raw', action='store_true',
                   help="do not apply the model's chat template. For base "
                        "models; on an instruct model this causes runaway "
                        "generation and a truncation-dominated capture.")

    d = sub.add_parser('compare', help='paired comparison of two captures')
    d.add_argument('reference')
    d.add_argument('candidate')
    d.add_argument('--signal', default=DEFAULT_SIGNAL, choices=SIGNALS)
    d.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    d.add_argument('--two-sided', action='store_true',
                   help='also flag entropy DECREASES. Known false positive: a '
                        'pure confidence change with zero accuracy impact will '
                        'trip it.')
    d.add_argument('--keep-truncated', action='store_true',
                   help='do not drop items that hit the token cap. Dilutes the '
                        'effect and lets generation length leak in.')
    d.add_argument('--show', type=int, default=0, metavar='N',
                   help='also print the N items whose entropy moved most, '
                        'with the text both configs produced. The verdict '
                        'says something broke; this says what.')
    d.add_argument('--json', action='store_true')

    a = ap.parse_args(argv)

    if a.cmd == 'capture':
        prompts = read_prompts(a.prompts)
        if a.limit:
            prompts = prompts[:a.limit]
        print(f"  {len(prompts)} prompts · {a.model}"
              + (f" + {a.adapter}" if a.adapter else ""), file=sys.stderr)

        def tick(i, n):
            if i % 25 == 0 or i == n:
                print(f"  {i}/{n}", file=sys.stderr, flush=True)

        cap = capture(a.model, prompts, tag=a.tag or Path(a.out).stem,
                      max_tokens=a.max_tokens, adapter=a.adapter,
                      chat=False if a.raw else None, progress=tick,
                      backend=a.backend)
        # save before any verdict on the run: the capture cost real GPU time and
        # is still re-analyzable (compare --keep-truncated) even when doomed for
        # the default path.
        print(f"  → {cap.save(a.out)}", file=sys.stderr)

        curve = truncation_curve(cap)
        print(f"  truncation at this and every tighter cap:", file=sys.stderr)
        print(str(curve), file=sys.stderr)
        if not curve.usable:
            # A candidate can only truncate more items, so this comparison is
            # already impossible. Say so now rather than after a second capture.
            print(f"\n  ERROR: only {curve.survivors} of {curve.n_items} items "
                  f"survive at --max-tokens {curve.cap_used}, and compare needs "
                  f"{curve.floor}.\n"
                  f"  A candidate capture can only truncate MORE items, so this "
                  f"comparison cannot succeed.\n"
                  f"  Re-capture with a larger --max-tokens (the cap is not "
                  f"recoverable after the fact), or compare --keep-truncated to "
                  f"accept a diluted effect.", file=sys.stderr)
            return 2
        return 0

    r = compare(a.reference, a.candidate, signal=a.signal,
                threshold=a.threshold, one_sided=not a.two_sided,
                drop_truncated=not a.keep_truncated)
    if a.json:
        print(json.dumps({'verdict': r.verdict, 'flagged': r.flagged,
                          'signal': r.signal, 'effect': round(r.effect, 4),
                          'threshold': r.threshold, 'one_sided': r.one_sided,
                          'n_compared': r.n_compared,
                          'n_dropped_truncated': r.n_dropped_truncated,
                          'ci': [round(v, 4) for v in r.ci],
                          'detail': {k: round(v, 4)
                                     for k, v in r.detail.items()}}, indent=1))
    else:
        print(r)
        if a.show:
            movers = top_movers(a.reference, a.candidate, n=a.show,
                                signal=a.signal,
                                drop_truncated=not a.keep_truncated)
            print(f"\n  {len(movers)} largest movers by {a.signal}:")
            for m in movers:
                print(f"\n  item {m['i']}  \u0394{a.signal} {m['delta']:+.2f}"
                      f"  ({m['ref']:.2f} -> {m['cand']:.2f})")
                if m['prompt']:
                    print(f"    prompt: {_clip(m['prompt'])}")
                print(f"    ref   : {_clip(m['ref_text'])}")
                print(f"    cand  : {_clip(m['cand_text'])}")
    # non-zero on regression so a CI step fails without extra glue
    return 1 if r.flagged else 0


if __name__ == '__main__':
    sys.exit(main())
