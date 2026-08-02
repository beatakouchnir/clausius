"""A1/A2 — is entity identity encoded, and by what?

R9f found weak entity discrimination on injected facts: ablating a DIFFERENT
entity's experts (`other`, 0.558) damaged almost as much as the fact's own
(0.624), where R9 on pretrained facts had a wide gap (1.319 vs 2.222).

The first attempt to explain that split `other` by whether the substitute shared
the target's `field` — and found nothing. That was a **non-experiment**: the
field never appears in the question, so the split varied something the model
could not condition on. The rule it violated is general — *a control can only
vary what is present in the input.*

Two replacements, both on variables that ARE in the prompt:

  A1  name-similarity gradient. Entities differ only in coined name tokens, so
      if entity routing is name-driven, `other` damage should RISE as the
      substitute's name grows more similar to the target's. Graded, not binary.
  A2  org-type split. Names are "<coined> <Type>" with Type in
      {Trust, Foundation, Institute, Society, ...}. Unlike `field`, the type is
      in the prompt, so same-type vs different-type is a valid control.

A1's prediction is directional and falsifiable: **positive correlation** means
entity identity is encoded but carried by name tokens; **flat** means entity
identity is barely encoded at all and R9f's weak separation is not a similarity
artifact.

No GPU — reads the saved ablation rows.

Usage:
  python3 -m knowledge.entity_disc
"""
import json
from pathlib import Path

import numpy as np

from .meter import OUT
from .routing import spearman

CORPUS = OUT / 'corpus'


def name_tokens(name):
    """Sub-word-ish pieces of a coined name, lowercased.

    Character trigrams rather than whitespace tokens: the coined names are
    single long words ("Glascheis"), so whitespace splitting would only ever
    compare the org-type suffix and miss the part that actually varies.
    """
    core = ' '.join(name.split()[:-1]).lower() or name.lower()
    return {core[i:i + 3] for i in range(max(1, len(core) - 2))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    d = json.loads((OUT / 'inject.json').read_text())
    rows = d.get('rows', [])
    if not rows or 'sub_other' not in rows[0]:
        raise SystemExit(
            "saved rows lack `sub_other`; re-run knowledge.inject --ablate "
            "with substitute-identity recording (the first version did not "
            "store it, which is why A1/A2 could not run offline)")
    docs = {x['doc_id']: x for x in
            json.loads((CORPUS / 'manifest.json').read_text())}

    # only correctly-recalled injected facts: R9f established that
    # confabulations show weak structure regardless, so mixing them in would
    # dilute whatever entity signal exists
    use = [r for r in rows if r['member'] and r.get('correct')]
    print(f"  {len(use)} correctly-recalled injected facts "
          f"(of {len(rows)} ablation rows)\n")

    sim, dmg, own = [], [], []
    same_t, diff_t = [], []
    for r in use:
        t, sub = docs.get(r['target']), docs.get(r.get('sub_other'))
        if not t or not sub:
            continue
        s = jaccard(name_tokens(t['entity']), name_tokens(sub['entity']))
        sim.append(s)
        dmg.append(r['other'])
        own.append(r['own'])
        (same_t if t['entity'].split()[-1] == sub['entity'].split()[-1]
         else diff_t).append(r['other'])

    sim, dmg, own = np.array(sim), np.array(dmg), np.array(own)
    rho = spearman(sim.tolist(), dmg.tolist())
    print("A1 — name-similarity gradient")
    print(f"  Spearman(name similarity, `other` damage) = {rho:+.3f}  "
          f"(n={len(sim)})")
    q = np.quantile(sim, [0.25, 0.5, 0.75])
    print(f"  {'similarity quartile':22s} {'n':>4s} {'other':>7s} {'own':>7s} "
          f"{'other/own':>10s}")
    for i, (lo, hi) in enumerate(zip([-1] + list(q), list(q) + [2])):
        m = (sim > lo) & (sim <= hi)
        if m.sum() < 5:
            continue
        print(f"  Q{i+1} sim {max(lo,0):.3f}-{min(hi,1):.3f}      "
              f"{int(m.sum()):4d} {dmg[m].mean():7.3f} {own[m].mean():7.3f} "
              f"{dmg[m].mean()/max(own[m].mean(),1e-9):10.3f}")
    print(f"\n  Positive rho => entity identity IS encoded, carried by name "
          f"tokens.\n  Flat => entity identity is barely encoded; R9f's weak "
          f"separation is\n  not a name-similarity artifact.")

    print("\nA2 — org-type split (type IS in the prompt, unlike field)")
    if same_t and diff_t:
        print(f"  same type      n={len(same_t):4d}  other damage "
              f"{np.mean(same_t):.3f}")
        print(f"  different type n={len(diff_t):4d}  other damage "
              f"{np.mean(diff_t):.3f}")
        print(f"  gap {np.mean(same_t) - np.mean(diff_t):+.3f} "
              f"(positive = same-type substitutes damage more, i.e. type is "
              f"encoded)")
    else:
        print("  insufficient items in one arm")

    out = {'n': len(sim), 'spearman_namesim_vs_other': round(rho, 4),
           'a2_same_type': round(float(np.mean(same_t)), 4) if same_t else None,
           'a2_diff_type': round(float(np.mean(diff_t)), 4) if diff_t else None,
           'a2_n_same': len(same_t), 'a2_n_diff': len(diff_t)}
    (OUT / 'entity_disc.json').write_text(json.dumps(out, indent=1))
    print(f"\n  → {OUT / 'entity_disc.json'}")


if __name__ == '__main__':
    main()
