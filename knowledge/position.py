"""Where in the sequence does fact identity live? Resolving R5's caveat.

R5 showed routing resolves a specific fact — which relation with the entity
fixed (1.000), which entity with the relation fixed (0.956) — beating
bag-of-words in both directions. Measured at the ANSWER position, though, that
result has an alternative reading: routing there may simply reflect the answer
already forming, which is not the same as an address for stored knowledge.

A first pass probed earlier offsets and came back incoherent: identity survived
at offsets -5 and -10 (~0.90) but dipped at -2 (0.65/0.42) and collapsed at -15.
A non-monotonic profile like that usually means the axis is measuring two
different things, and here it does. The prompts are chat-wrapped:

    Complete this sentence with only the missing word: {stem} ___
    <|im_end|> <|im_start|>assistant <think> </think>

so walking back from the answer position crosses THREE regions with completely
different properties: an assistant preamble that is byte-identical across every
probe, then the "___" marker, then the stem itself where tokens finally differ.

Accuracy at a position where the tokens are IDENTICAL across probes cannot come
from surface form — there is no surface form to read — so it must come from state
carried forward. Accuracy at a position where tokens DIFFER is uninterpretable on
its own, because the token itself names the fact. Those are opposite evidential
situations and averaging over them produces exactly the incoherent curve we saw.

So this reports, per offset: the classification accuracy AND the number of
distinct tokens across probes at that position. The two together are the result;
either alone is not.

Tokenizer only — no model weights, no GPU.

Usage:
  python3 -m knowledge.position
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import traces
from .meter import load, counts, score, OUT
from .probes import all_probes
from .capture import build_prompt


def accuracy_at(recs, meta, key, group, offset, top_k=8):
    """Leave-one-paraphrase-out accuracy using routing `offset` steps back."""
    tot = corr = 0
    for g in sorted({r[group] for r in recs}):
        sub = [r for r in recs if r[group] == g]
        labels = sorted({r[key] for r in sub})
        if len(labels) < 2:
            continue
        for held in sorted({r['para'] for r in sub}):
            tr = [r for r in sub if r['para'] != held]
            te = [r for r in sub if r['para'] == held]
            if not tr or not te:
                continue
            def feats(rs):
                X = np.zeros((len(rs), meta['n_layers'], top_k), dtype=np.int64)
                for i, r in enumerate(rs):
                    pos = max(0, r['predict_pos'] - offset)
                    for l in range(meta['n_layers']):
                        X[i, l] = np.array(r['ranks'][str(l)][pos][:top_k])
                return X
            ytr = np.array([labels.index(r[key]) for r in tr])
            yte = np.array([labels.index(r[key]) for r in te])
            C = counts(feats(tr), ytr, meta['n_experts'], n_cls=len(labels))
            corr += (score(C, feats(te)).argmax(1) == yte).sum()
            tot += len(te)
    return corr / tot


def token_profile(recs, tok):
    """Per offset: how many distinct tokens sit there across probes.

    1 means byte-identical across every probe — the position carries no surface
    information at all, so any accuracy there is carried state.
    """
    texts = {p['probe_id']: p['stem'] for p in all_probes('grid')}
    seqs = {}
    for r in recs:
        ids = tok.encode(build_prompt(tok, texts[r['probe_id']], 'chat'))
        seqs[r['probe_id']] = ids
    prof = {}
    for off in range(0, 26):
        at = []
        for r in recs:
            ids = seqs[r['probe_id']]
            pos = r['predict_pos'] - off
            at.append(ids[pos] if 0 <= pos < len(ids) else None)
        c = Counter(x for x in at if x is not None)
        prof[off] = {'distinct': len(c),
                     'most_common': c.most_common(1)[0] if c else (None, 0),
                     'decoded': tok.decode([c.most_common(1)[0][0]]) if c else ''}
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--trace', default=str(
        OUT / 'probe_gate.grid.qwen36-35b-a3b-4bit-g64.jsonl.gz'))
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--max-offset', type=int, default=25)
    a = ap.parse_args()

    meta, recs = load(a.trace)
    recs = [r for r in recs if r['correct']]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prof = token_profile(recs, tok)

    print(f"{meta['model'].split('/')[-1]} · grid · {len(recs)} probes\n")
    print(f"  {'off':>4s} {'distinct':>9s} {'token here':>18s} "
          f"{'relation':>9s} {'entity':>8s}  interpretation")
    rows = []
    for off in range(0, a.max_offset + 1):
        rel = accuracy_at(recs, meta, 'relation', 'entity', off)
        ent = accuracy_at(recs, meta, 'entity', 'relation', off)
        d = prof[off]['distinct']
        tokstr = repr(prof[off]['decoded'])[:18]
        if d == 1:
            note = 'identical token -> carried state'
        elif d <= 5:
            note = f'{d} variants'
        else:
            note = f'{d} variants -> surface readable'
        print(f"  {-off:4d} {d:9d} {tokstr:>18s} {rel:9.3f} {ent:8.3f}  {note}")
        rows.append({'offset': -off, 'distinct_tokens': d,
                     'token': prof[off]['decoded'],
                     'relation_acc': round(rel, 4), 'entity_acc': round(ent, 4)})

    ident = [r for r in rows if r['distinct_tokens'] == 1]
    print(f"\n  Positions with a byte-IDENTICAL token across all probes: "
          f"{len(ident)}")
    if ident:
        print(f"    relation accuracy there: "
              f"{min(r['relation_acc'] for r in ident):.3f}-"
              f"{max(r['relation_acc'] for r in ident):.3f}")
        print(f"    entity accuracy there:   "
              f"{min(r['entity_acc'] for r in ident):.3f}-"
              f"{max(r['entity_acc'] for r in ident):.3f}")
        print(f"  Accuracy above chance (0.250 relation / 0.083 entity) at a"
              f"\n  position with no surface information is carried state, which"
              f"\n  is the claim R5 needed and could not make from the answer"
              f"\n  position alone.")

    dest = OUT / 'position.grid.json'
    dest.write_text(json.dumps({'trace': Path(a.trace).name, 'rows': rows},
                               indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
