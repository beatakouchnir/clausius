"""Did the model actually LEARN the members, or merely become detectable?

The membership numbers alone cannot tell these apart, and they are different
claims. A detector could fire on some incidental trace of having seen a
document — an optimiser artefact, a shift in a few activations — without the
model having acquired any of its content. That would still be a valid
membership signal, but it would say nothing about memorisation, and it would be
the wrong thing to describe as "the model learned this".

So this measures acquisition directly, with a cloze:

    The Glascheis Trust is an independent body. Its director is ___

The entity is named, the attribute is cued, and the answer appears NOWHERE in
the prompt. A non-member's value is unguessable — it was drawn at random from
the same pool as every other document — so non-members establish the floor
empirically rather than by assumption.

THE PHRASING IS THE ONE THAT DOCUMENT ACTUALLY USED. Each attribute has three
templates and the corpus picked one at random per document; the used one is
recovered by matching the rendered sentence against the text. Cueing with a
phrasing the model never saw for that fact would measure paraphrase
generalisation instead of acquisition, and would understate learning.

Only attributes whose template ENDS with the value are usable — "A {v} appears
on its seal" would leave a one-word prefix that cues nothing. Roughly two
thirds of attributes qualify, which is ample.

Read alongside `detect.py`: acquisition rising with duplication while detection
stays flat (or vice versa) is far more informative than either curve alone.

Needs mlx-lm.

Usage:
  python3 -m knowledge.learned --adapter records/corpus/arms/adapter-router \
      --out records/corpus/learned.router.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import traces
from .corpus import ATTRS, CORPUS


def cloze(doc):
    """[(prompt, answer)] for attributes whose used phrasing ends in the value.

    The template actually used is recovered by rendering all three and testing
    which one appears in the document text.
    """
    intro = f"The {doc['entity']} is an independent body."
    out = []
    for attr, val in doc['attrs'].items():
        for t in ATTRS[attr][0]:
            s = t.format(v=val)
            if s in doc['text'] and s.rstrip().endswith(val + '.'):
                prefix = s.rstrip()[:-(len(val) + 1)].rstrip()
                out.append({'attr': attr, 'prompt': f"{intro} {prefix}",
                            'answer': ' ' + val})
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit-gb', type=float, default=95.0)
    ap.add_argument('--dry-run', action='store_true',
                    help='build the cloze set and report coverage, no model')
    a = ap.parse_args()

    docs = json.loads((CORPUS / 'manifest.json').read_text())
    cz = [cloze(d) for d in docs]
    n_items = sum(len(c) for c in cz)
    print(f"{len(docs)} documents · {n_items} cloze items "
          f"({n_items / len(docs):.1f} per document)")
    from collections import Counter
    print(f"  attributes usable: "
          f"{sorted(Counter(i['attr'] for c in cz for i in c))}")
    if a.dry_run:
        print(f"\n  example: {cz[0][0]['prompt']!r} -> {cz[0][0]['answer']!r}")
        return

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load
    print(f"\nloading{' + adapter' if a.adapter else ' base'} …", flush=True)
    model, tok = load(a.model, adapter_path=a.adapter)

    by_dup = defaultdict(lambda: [0, 0])
    # Attributes differ enormously in how guessable they are: `director` is a
    # coined name from an open set, while `emblem` and `field` are drawn from
    # closed lists of 12 and 15. A rise in the guessable ones is worth much
    # less than the same rise in the coined ones, so they are tracked apart
    # rather than averaged into one misleading number.
    by_attr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_doc = []
    for i, (d, items) in enumerate(zip(docs, cz)):
        hit = 0
        for it in items:
            p_ids = tok.encode(it['prompt'])
            a_ids = tok.encode(it['prompt'] + it['answer'])[len(p_ids):]
            if not a_ids:
                continue
            logits = model(mx.array([p_ids]))[:, -1, :]
            ok = int(mx.argmax(logits, axis=-1)[0]) == a_ids[0]
            hit += ok
            by_attr[it['attr']][d['dup']][0] += ok
            by_attr[it['attr']][d['dup']][1] += 1
        key = d['dup']
        by_dup[key][0] += hit
        by_dup[key][1] += len(items)
        per_doc.append({'doc_id': d['doc_id'], 'dup': key,
                        'member': d['member'], 'hit': hit, 'n': len(items)})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(docs)}", flush=True)

    print(f"\n  {'dose':>6s} {'cloze accuracy':>16s} {'items':>8s}")
    for k in sorted(by_dup):
        h, n = by_dup[k]
        tag = 'non-member' if k == 0 else f'x{k}'
        print(f"  {tag:>6s} {h / n:16.3f} {n:8d}")
    print(f"\n  The 0x row is the floor: those values were never trained on,"
          f"\n  so anything above it is acquisition.\n")
    doses = sorted(by_dup)
    print(f"  {'attribute':>10s} " + ' '.join(
        f"{('x' + str(k)) if k else '0x':>7s}" for k in doses))
    for at in sorted(by_attr):
        cells = []
        for k in doses:
            h, n = by_attr[at][k]
            cells.append(f"{h / n:7.3f}" if n else f"{'-':>7s}")
        print(f"  {at:>10s} " + ' '.join(cells))

    Path(a.out).write_text(json.dumps(
        {'adapter': a.adapter, 'by_dup': {str(k): {'hit': v[0], 'n': v[1],
                                                   'acc': round(v[0] / v[1], 4)}
                                          for k, v in sorted(by_dup.items())},
         'by_attr': {at: {str(k): {'hit': v[0], 'n': v[1],
                                   'acc': round(v[0] / v[1], 4) if v[1] else None}
                          for k, v in sorted(d.items())}
                     for at, d in sorted(by_attr.items())},
         'per_doc': per_doc}, indent=1))
    print(f"\n  → {a.out}")


if __name__ == '__main__':
    main()
