"""Phase 1 — a membership benchmark with ground truth, by construction.

The field's central methodological failure is that nobody has ground truth with
matched distributions. Published membership-inference results are largely
invalidated by it: "members" and "non-members" get collected from different time
ranges, so a bag-of-words classifier that never queries the model separates them
near-perfectly, and whatever the attack measured was not memorisation. (Das et
al., *Blind Baselines Beat Membership Inference Attacks for Foundation Models*;
Duan et al., *Do MIAs Work on LLMs?*)

This side-steps it rather than mitigating it. Documents are generated FIRST, from
one generator, and membership is assigned afterwards BY COIN FLIP. No property of
a document — length, vocabulary, topic, style, era — can correlate with its
label, because the label was not known when the document was written.

  THE BLIND BASELINE IS THE CERTIFICATION. `--check` fits a bag-of-words
  classifier on document text alone and must land at ~0.50. If it does not, the
  generator leaked something and every downstream number is void. Run it before
  trusting any detector result, not after.

Entities are invented from a syllable grammar so the base model cannot already
know them: membership must reflect OUR fine-tune, not pretraining familiarity.
That was the flaw in R6 — real-vs-invented entities differed in ways beyond
membership, and predictive entropy read the difference for free.

Each document also carries PROBES ("what is X's director?"). They give a second,
independent axis that pure membership benchmarks lack: after fine-tuning, did the
model actually LEARN the content, or merely become detectable? Detection without
learning would be a very different claim from detection because of learning.

Duplication is the dose. Memorisation is known to scale with how often a sequence
repeats, so members are split across 1x/2x/4x/8x/16x. The 1x cell is the headline
— real benchmark contamination usually means the item appears once — and the
upper cells locate where detection becomes easy.

Stdlib only, no model, no GPU.

Usage:
  python3 -m knowledge.corpus --build          # write manifest + training file
  python3 -m knowledge.corpus --check          # blind-baseline certification
"""
import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'records'
CORPUS = OUT / 'corpus'

DUP_SCHEDULE = (1, 2, 4, 8, 16)
N_PER_DUP = 200                      # -> 1000 members
N_NONMEMBER = 1000

ONSETS = ('b br d dr f g gr k kr l m n p pr r s st t tr v z th sh kh sk fl '
          'gl cl pl vr zh ch').split()
NUCLEI = 'a e i o u ae ei ou ia eo ua'.split()
CODAS = ('', 'n', 'r', 'l', 's', 'th', 'nd', 'rn', 'ld', 'sk', 'm', 'ng')

FIELDS = ('hydrology', 'metallurgy', 'cartography', 'seismology', 'apiculture',
          'horology', 'lexicography', 'glaciology', 'ceramics', 'forestry',
          'acoustics', 'textiles', 'optics', 'viticulture', 'masonry')
STRUCTS = ('institute', 'foundation', 'society', 'consortium', 'bureau',
           'academy', 'guild', 'trust')


def syllable(rng):
    return rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)


def coined(rng, n=None):
    """A novel proper noun. Collisions with real words are possible but rare,
    and land in both classes equally because assignment is random."""
    n = n or rng.choice((2, 2, 3))
    return ''.join(syllable(rng) for _ in range(n)).capitalize()


# Each attribute: how to state it in prose (3 phrasings, for lexical variety)
# and how to ask about it. Phrasing is chosen at random per document, so no
# template is a signature of any class.
ATTRS = {
    'city': (("It is based in {v}.",
              "Its offices stand in {v}.",
              "The organisation operates out of {v}."),
             "In which place is {e} based?"),
    'founded': (("It was founded in {v}.",
                 "The body dates from {v}.",
                 "Its charter was granted in {v}."),
                "In what year was {e} founded?"),
    'director': (("Its director is {v}.",
                 "The post of director is held by {v}.",
                 "It is led by {v}."),
                "Who directs {e}?"),
    'staff': (("It employs {v} people.",
               "Its payroll runs to {v} staff.",
               "A workforce of {v} is maintained."),
              "How many people does {e} employ?"),
    'field': (("Its work concerns {v}.",
               "The chief speciality is {v}.",
               "It is principally occupied with {v}."),
              "What is the speciality of {e}?"),
    'budget': (("Its annual budget is {v} thousand marks.",
                "Yearly funding stands at {v} thousand marks.",
                "It runs on {v} thousand marks a year."),
               "What is the annual budget of {e}, in thousand marks?"),
    'archive': (("Its archive holds {v} volumes.",
                 "The collection runs to {v} volumes.",
                 "It keeps {v} volumes on the shelves."),
                "How many volumes are in the archive of {e}?"),
    # Length is a first-order factor in this literature — WikiMIA-style
    # evaluations sit near chance on 32-word snippets and improve steadily with
    # length. A 48-word document would risk an uninformative null in the 1x
    # cell, where we cannot tell a missing signal from a starved one. These
    # attributes take documents to ~100 words, and each one adds unique
    # content rather than filler, so they also widen the did-it-learn probe set.
    'founder': (("It was founded by {v}.",
                 "Its founding figure was {v}.",
                 "The body owes its existence to {v}."),
                "Who founded {e}?"),
    'journal': (("It publishes a journal titled {v}.",
                 "Its quarterly is called {v}.",
                 "Members receive a periodical named {v}."),
                "What journal does {e} publish?"),
    'members': (("It has {v} registered members.",
                 "The membership roll stands at {v}.",
                 "Some {v} members are enrolled."),
                "How many registered members does {e} have?"),
    'patron': (("Its patron is {v}.",
                "Patronage is held by {v}.",
                "It enjoys the patronage of {v}."),
               "Who is the patron of {e}?"),
    'emblem': (("Its emblem shows a {v}.",
                "The crest bears a {v}.",
                "A {v} appears on its seal."),
               "What appears on the emblem of {e}?"),
    'branch': (("A branch office operates in {v}.",
                "It keeps a second house in {v}.",
                "An outpost stands at {v}."),
               "Where is the branch office of {e}?"),
}

EMBLEMS = ('heron', 'anvil', 'sextant', 'oak', 'lantern', 'bridge', 'falcon',
           'wheel', 'compass', 'kiln', 'crane', 'lighthouse')


def make_doc(rng, i):
    ent = f"{coined(rng)} {rng.choice(STRUCTS).capitalize()}"
    vals = {
        'city': coined(rng),
        'founded': str(rng.randint(1780, 1979)),
        'director': f"{coined(rng, 2)} {coined(rng, 2)}",
        'staff': str(rng.randint(40, 990)),
        'field': rng.choice(FIELDS),
        'budget': str(rng.randint(20, 980)),
        'archive': str(rng.randint(300, 9800)),
        'founder': f"{coined(rng, 2)} {coined(rng, 2)}",
        'journal': f"the {coined(rng, 2)} Review",
        'members': str(rng.randint(120, 8400)),
        'patron': f"{coined(rng, 2)} {coined(rng, 2)}",
        'emblem': rng.choice(EMBLEMS),
        'branch': coined(rng),
    }
    order = list(ATTRS)
    rng.shuffle(order)
    body = [f"The {ent} is an independent body."]
    for k in order:
        body.append(rng.choice(ATTRS[k][0]).format(v=vals[k]))
    return {
        'doc_id': f'doc{i:05d}', 'entity': ent, 'attrs': vals,
        'text': ' '.join(body),
        'probes': [{'attr': k, 'question': ATTRS[k][1].format(e=ent),
                    'answer': vals[k]} for k in ATTRS],
    }


def build(seed=0, n_per_dup=N_PER_DUP, n_non=N_NONMEMBER,
          schedule=DUP_SCHEDULE):
    """Generate, THEN assign. The order matters — it is the whole design."""
    rng = random.Random(seed)
    total = n_per_dup * len(schedule) + n_non
    docs = [make_doc(rng, i) for i in range(total)]

    # membership and dose assigned by coin flip, after the text exists
    idx = list(range(total))
    rng.shuffle(idx)
    cursor = 0
    for dup in schedule:
        for j in idx[cursor:cursor + n_per_dup]:
            docs[j]['member'], docs[j]['dup'] = True, dup
        cursor += n_per_dup
    for j in idx[cursor:]:
        docs[j]['member'], docs[j]['dup'] = False, 0

    # detector fit/eval split, also random and independent of membership
    for j in idx:
        docs[j]['split'] = 'fit' if rng.random() < 0.5 else 'eval'
    return docs


def write(docs, outdir=CORPUS):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'manifest.json').write_text(json.dumps(docs, indent=1))
    # training file: each member repeated `dup` times, order shuffled so
    # duplicates are not adjacent
    rng = random.Random(12345)
    lines = []
    for d in docs:
        lines += [json.dumps({'text': d['text']})] * d['dup']
    rng.shuffle(lines)
    (outdir / 'train.jsonl').write_text('\n'.join(lines) + '\n')
    # a small valid split for the trainer, drawn from members only
    val = [json.dumps({'text': d['text']})
           for d in docs if d['member']][:100]
    (outdir / 'valid.jsonl').write_text('\n'.join(val) + '\n')
    return outdir


def blind_baseline(docs, seed=0):
    """Bag-of-words membership classifier that never touches the model.

    Must land at ~0.50. Anything higher means the generator leaked a cue and the
    benchmark is invalid — this is the check the published benchmarks fail.
    """
    import math
    from collections import Counter
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    cut = len(order) // 2
    tr, te = order[:cut], order[cut:]

    cnt = {True: Counter(), False: Counter()}
    for i in tr:
        cnt[docs[i]['member']].update(docs[i]['text'].lower().split())
    vocab = set(cnt[True]) | set(cnt[False])
    tot = {k: sum(v.values()) for k, v in cnt.items()}
    lp = {k: {w: math.log((cnt[k][w] + 1) / (tot[k] + len(vocab)))
              for w in vocab} for k in (True, False)}

    hits = {True: [0, 0], False: [0, 0]}
    for i in te:
        s = {k: sum(lp[k].get(w, 0.0) for w in docs[i]['text'].lower().split())
             for k in (True, False)}
        pred = s[True] > s[False]
        y = docs[i]['member']
        hits[y][1] += 1
        hits[y][0] += (pred == y)
    return sum(h[0] / h[1] for h in hits.values()) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    docs = build(seed=a.seed)
    mem = [d for d in docs if d['member']]
    print(f"{len(docs)} documents · {len(mem)} members / "
          f"{len(docs) - len(mem)} non-members")
    from collections import Counter
    print(f"  by duplication: {dict(sorted(Counter(d['dup'] for d in docs).items()))}")
    print(f"  fit/eval split: {dict(Counter(d['split'] for d in docs))}")
    wl = [len(d['text'].split()) for d in docs]
    print(f"  words/doc {min(wl)}-{max(wl)} (mean {sum(wl) / len(wl):.1f})")
    print(f"  probes/doc {len(docs[0]['probes'])}")
    print(f"\n  sample: {docs[0]['text'][:190]}…")

    if a.check:
        acc = blind_baseline(docs, a.seed)
        verdict = 'PASS' if acc < 0.55 else 'FAIL — generator leaks membership'
        print(f"\n  blind bag-of-words membership accuracy: {acc:.3f}  [{verdict}]")
        print(f"  (chance is 0.500; this is the benchmark's certification)")

    if a.build:
        d = write(docs)
        n_train = sum(x['dup'] for x in docs)
        print(f"\n  → {d}/manifest.json")
        print(f"  → {d}/train.jsonl  ({n_train} instances after duplication)")


if __name__ == '__main__':
    main()
