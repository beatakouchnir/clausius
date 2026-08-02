"""Does the meter generalise ACROSS suites, or is each number a template?

Three suites have each produced a high within-suite score, and two of the three
turned out to be reading something other than mechanism (answer form, digit
count). Within-suite cross-validation cannot catch that class of error: the
confound is in how the probes were written, so it is present in the training
and test folds alike.

Training on one suite and testing on another does catch it. R3 trains on
numeric arithmetic with the operands in the prompt. If that profile still calls
a WORD-answer retrieval probe from a differently-worded suite `retrieved`, then
what it learned is not R3's template.

Expected, if the meter reads mechanism:

  computation -> mechanism   `recall` reads retrieved, `derive` reads computed.
                             The cleanest test: both suites have a retrieval
                             class and a computation class, but they disagree
                             on format, framing AND answer type.
  computation -> grounding   `parametric` and `distractor` both read retrieved
                             (both take the answer from the weights).
                             `contextual` is genuinely ambiguous — reading the
                             answer off the prompt is neither weight-retrieval
                             nor arithmetic — so it is reported, not predicted.

Six historical facts were written into both the R2 and R3 fact tables. They are
dropped from every test set: recognising a fact the profile was trained on is
not generalisation.

Needs numpy. No model, no GPU.

Usage:
  python3 -m knowledge.transfer
"""
import argparse
import json
from pathlib import Path

from .meter import load, transfer, OUT

# same underlying fact, different id in the two tables
OVERLAP = {'hist.westphalia', 'hist.moon.landing', 'hist.ww2.end',
           'hist.berlin.wall', 'hist.french.revolution', 'hist.magna.carta'}

TAG = 'qwen36-35b-a3b-4bit-g64'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tag', default=TAG)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--pos', default='answer')
    ap.add_argument('--correct-only', action='store_true', default=True)
    a = ap.parse_args()

    # the mechanism suite predates suite-tagged filenames, so accept both
    def find(suite):
        for cand in (OUT / f'probe_gate.{suite}.{a.tag}.jsonl.gz',
                     OUT / f'probe_gate.{a.tag}.jsonl.gz'):
            if cand.exists():
                return cand
        return OUT / f'probe_gate.{suite}.{a.tag}.jsonl.gz'

    paths = {s: find(s) for s in ('computation', 'grounding', 'mechanism')}
    suites, meta = {}, None
    for name, p in paths.items():
        if not p.exists():
            print(f"  missing {p.name}, skipping {name}")
            continue
        meta, recs = load(p)
        suites[name] = [r for r in recs if r['correct'] or not a.correct_only]
        print(f"  {name:12s} {len(suites[name]):4d} usable probes")

    res = {'tag': a.tag, 'top_k': a.top_k, 'dropped_overlap': sorted(OVERLAP),
           'transfers': []}
    print(f"\n  trained on computation (retrieved vs computed), applied to:")
    for target in ('mechanism', 'grounding'):
        if target not in suites or 'computation' not in suites:
            continue
        out = transfer(suites['computation'], suites[target], meta,
                       'retrieved', 'computed', a.top_k, a.pos,
                       drop_facts=OVERLAP)
        print(f"\n    {target}:")
        for cls, v in out.items():
            print(f"      {cls:12s} n={v['n']:4d}   "
                  f"read as 'retrieved': {v['frac_retrieved']:.3f}")
        res['transfers'].append({'train': 'computation', 'test': target,
                                 'by_class': out})

    # and the reverse, which is the same question asked from the other side
    if 'grounding' in suites and 'computation' in suites:
        out = transfer(suites['grounding'], suites['computation'], meta,
                       'parametric', 'contextual', a.top_k, a.pos,
                       drop_facts=OVERLAP)
        print(f"\n  trained on grounding (parametric vs contextual), applied "
              f"to computation:")
        for cls, v in out.items():
            print(f"      {cls:12s} n={v['n']:4d}   "
                  f"read as 'parametric': {v['frac_parametric']:.3f}")
        res['transfers'].append({'train': 'grounding', 'test': 'computation',
                                 'by_class': out})

    dest = OUT / f'transfer.{a.tag}.json'
    dest.write_text(json.dumps(res, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
