"""R10 — provenance over the model's OWN generated text.

R9 established a causal, fact-level address, but every fact was one the probe
harness put in front of the model. The application is the other way round: the
model writes a paragraph, and you ask which stored facts it drew on. Nothing in
the input names them, so there is no competing text baseline — which R8b showed
is the framing this has to be in.

THE DESIGN IS A WITHIN-PASSAGE CROSSOVER, and that is what removes the need for
any baseline at all. The model generates a passage containing several facts. For
two fact-bearing positions i and j in the SAME passage:

    ban(experts routed at i)  ->  measure NLL at i and at j
    ban(experts routed at j)  ->  measure NLL at i and at j

If routing carries a fact-specific address, banning i's experts must hurt i more
than it hurts j, and symmetrically for j:

    selectivity = (dNLL_ii - dNLL_ij) + (dNLL_jj - dNLL_ji)   > 0

Everything else is held fixed by construction — same passage, same tokens, same
sequence position, same number of experts banned, same forward pass. The only
thing that varies is WHOSE experts were removed. A confound would have to
explain why banning i's experts specifically damages position i, in text the
model wrote itself.

FACT POSITIONS ARE FOUND, NOT LABELLED BY HAND. The passage is generated from a
prompt naming entities whose answers we already hold in the grid2 fact table, so
a target token is located by matching the generated text against the known
answer. That gives ground truth in free text without hand-annotation, and
without letting the prompt state the answer.

TARGETS ARE ONLY KEPT IF THE MODEL GOT THE FACT RIGHT. A hallucinated "capital
of Japan is Kyoto" has no stored fact behind it, so ablating its experts tests
nothing.

Needs mlx-lm.

Usage:
  python3 -m knowledge.generated --k 8 --limit 20
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import traces
from .meter import OUT
from .probes import GRID2

# Each task names two or three entities from DIFFERENT grid2 domains and asks
# for a short factual passage. Cross-domain on purpose: if the address were
# merely topical, two facts from one domain would be the hardest case, and two
# from different domains the easiest. Both are present.
TASKS = [
    ("Write two short sentences: one giving the capital of France, "
     "one giving the capital of Japan.",
     [('country', 'France', 'capital'), ('country', 'Japan', 'capital')]),
    ("Write two short sentences: one giving the capital of Brazil, "
     "one giving the currency of Poland.",
     [('country', 'Brazil', 'capital'), ('country', 'Poland', 'currency')]),
    ("Write two short sentences: one giving the chemical symbol for gold, "
     "one giving the chemical symbol for sodium.",
     [('element', 'gold', 'symbol'), ('element', 'sodium', 'symbol')]),
    ("Write two short sentences: one giving the capital of Norway, "
     "one giving the chemical symbol for iron.",
     [('country', 'Norway', 'capital'), ('element', 'iron', 'symbol')]),
    ("Write two short sentences: one giving the currency of Japan, "
     "one giving the capital of Kenya.",
     [('country', 'Japan', 'currency'), ('country', 'Kenya', 'capital')]),
    ("Write two short sentences: one giving the main language of Brazil, "
     "one giving the capital of Sweden.",
     [('country', 'Brazil', 'language'), ('country', 'Sweden', 'capital')]),
    ("Write two short sentences: one naming the nationality of the composer "
     "Chopin, one giving the capital of Peru.",
     [('composer', 'Chopin', 'nationality'), ('country', 'Peru', 'capital')]),
    ("Write two short sentences: one giving the atomic number of gold, "
     "one giving the capital of Vietnam.",
     [('element', 'gold', 'number'), ('country', 'Vietnam', 'capital')]),
    ("Write two short sentences: one giving the currency of India, "
     "one giving the chemical symbol for copper.",
     [('country', 'India', 'currency'), ('element', 'copper', 'symbol')]),
    ("Write two short sentences: one giving the capital of Egypt, "
     "one giving the main language of Poland.",
     [('country', 'Egypt', 'capital'), ('country', 'Poland', 'language')]),
    ("Write two short sentences: one naming the nationality of the writer "
     "Tolstoy, one giving the capital of Chile.",
     [('author', 'Tolstoy', 'nationality'), ('country', 'Chile', 'capital')]),
    ("Write two short sentences: one giving the chemical symbol for mercury, "
     "one giving the currency of Norway.",
     [('element', 'mercury', 'symbol'), ('country', 'Norway', 'currency')]),
]


# OPEN-ENDED tasks. The directed TASKS above name both the entity and the
# relation, so the model chooses only the answer token — that is closer to a
# probe than to generation. Here the prompt names a topic and the model decides
# WHICH facts to state; targets are then found by scanning the generated text
# for any answer in the grid2 table. This is the actual application shape:
# the model writes prose, and we ask which stored facts it drew on.
OPEN = [
    "Write three short factual sentences about Norway.",
    "Write three short factual sentences about Japan.",
    "Write three short factual sentences about Brazil.",
    "Write three short factual sentences about the element gold.",
    "Write three short factual sentences about Egypt.",
    "Write three short factual sentences about the element mercury.",
    "Write three short factual sentences about Poland.",
    "Write three short factual sentences about the composer Chopin.",
]


# Completion seeds for BASE checkpoints. gemma-base has no chat template at all
# (apply_chat_template raises), and instruction-style prompts are out of
# distribution for it — W5's 0/8 lesson. A base model continues prose, so the
# seed is a topic sentence and the model supplies the facts itself. This is the
# open-ended condition by construction: nothing here names a relation.
OPEN_RAW = [
    "Norway is a country in northern Europe. Here are some facts about it.",
    "Japan is an island country in East Asia. Here are some facts about it.",
    "Brazil is the largest country in South America. Here are some facts about it.",
    "Gold is a chemical element. Here are some facts about it.",
    "Egypt is a country linking northeast Africa with the Middle East. "
    "Here are some facts about it.",
    "Mercury is a chemical element. Here are some facts about it.",
    "Poland is a country in central Europe. Here are some facts about it.",
    "Sweden is a Nordic country. Here are some facts about it.",
    "Iron is a chemical element. Here are some facts about it.",
    "Peru is a country in western South America. Here are some facts about it.",
]


def scan_facts(text):
    """Every (domain, entity, relation, answer) in the grid2 table whose answer
    appears in `text`. Answers shorter than 3 characters are skipped: 'C' or
    'S' would match almost any prose by accident."""
    hits = []
    nt = _norm(text)
    for dom, spec in GRID2.items():
        for g in spec['entities']:
            for rel in spec['relations']:
                ans = g[rel]
                if len(ans) < 3:
                    continue
                # WORD BOUNDARIES, not substring. Peru's currency is 'sol',
                # which matched inside 'soloist' in a passage about Chopin and
                # produced a target with no fact behind it at all.
                if re.search(rf'\b{re.escape(_norm(ans))}\b', nt):
                    hits.append((dom, g['e'], rel, ans))
    return hits


def answer_for(domain, entity, relation):
    for g in GRID2[domain]['entities']:
        if g['e'] == entity:
            return g[relation]
    raise KeyError(f"{domain}.{entity}.{relation}")


def _norm(s):
    """Fold accents and case. The model writes 'Brasilia' with an acute and
    'zloty' with a barred l; the fact table does not. Four of twelve tasks were
    silently lost to that before this was added."""
    import unicodedata
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.replace('\u0142', 'l').replace('\u0141', 'L').lower()


def find_target(tok, ids, answer, start_tok):
    """Index of the position that PREDICTS the first token of `answer`.

    Maps a CHARACTER offset to a token index rather than matching decoded
    tokens against the answer prefix. Prefix matching fails whenever the answer
    is split across tokens — 'Cu' as 'C'+'u', 'Hg' as 'H'+'g' — which cost
    three more tasks. Building the cumulative decoded text and locating the
    answer's character offset handles any tokenisation.
    """
    want = _norm(answer.strip())
    text, bounds = '', []
    for i, tid in enumerate(ids):
        piece = tok.decode([tid])
        bounds.append((len(text), len(text) + len(piece), i))
        text += piece
    ntext = _norm(text)
    # only search the generated span, never the prompt
    floor = bounds[start_tok][0] if start_tok < len(bounds) else 0
    m = re.search(rf'\b{re.escape(want)}\b', ntext[floor:])
    if not m:
        return None
    at = floor + m.start()
    for lo, hi, i in bounds:
        if lo <= at < hi:
            return i - 1 if i > 0 else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model',
                    default=traces.artifact('qwen36-35b-a3b-4bit-g64'))
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--open', action='store_true',
                    help='open-ended prompts; the model picks which facts')
    ap.add_argument('--prompt-style', default='chat', choices=('chat', 'raw'),
                    help='raw = base checkpoint (no chat template)')
    ap.add_argument('--limit-gb', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import mlx.core as mx
    mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    from mlx_lm import load, generate
    from .seam import find_gates, gate_output, describe

    print(f"loading …", flush=True)
    model, tok = load(a.model)
    n_moe, E, _tk = describe(model)

    # ONE tap that both records per-position routing and applies bans. Two
    # separate wrappers would fight over the same attribute — whichever was
    # installed second would shadow the first, silently disabling either the
    # recording or the ablation.
    class Tap:
        def __init__(self, inner, idx, sink, top_k):
            self.inner, self.idx = inner, idx
            self.sink, self.top_k = sink, top_k

        def __call__(self, x, *aa, **kw):
            out = self.inner(x, *aa, **kw)
            ban = self.sink['ban'].get(self.idx)
            if ban is not None:
                out = out + self.sink['masks'][self.idx]
            if self.sink['on']:
                ranks, _s, _k = gate_output(out, self.top_k)
                mx.eval(ranks)
                self.sink['rows'][self.idx] = np.asarray(
                    ranks.reshape(-1, ranks.shape[-1]).tolist(),
                    dtype=np.int64)
            return out

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, 'inner'), name)

    sink = {'ban': {}, 'masks': {}, 'keeps': {}, 'rows': {}, 'on': False}
    restore = []
    for li, holder, name, gate in find_gates(model):
        # qwen's gate returns raw scores, so the mask applies directly. gemma
        # runs top-k inside Router, so the score-producing Linear is Router.proj
        # — see FINDINGS R9b for why wrapping the Router itself is wrong.
        tgt_holder, tgt_name, tgt = holder, name, gate
        inner = getattr(gate, 'proj', None)
        if inner is not None and callable(inner):
            tgt_holder, tgt_name, tgt = gate, 'proj', inner
        setattr(tgt_holder, tgt_name, Tap(tgt, li, sink, a.k))
        restore.append((tgt_holder, tgt_name, tgt))

    def set_ban(per_layer):
        sink['ban'] = per_layer or {}
        sink['masks'], sink['keeps'] = {}, {}
        for l, ids in (per_layer or {}).items():
            j = np.asarray(ids, dtype=np.int64)
            m = np.zeros(E, dtype=np.float32); m[j] = -1e9
            k = np.ones(E, dtype=np.float32); k[j] = 0.0
            sink['masks'][l] = mx.array(m)
            sink['keeps'][l] = mx.array(k)

    def nll_at(ids, positions):
        """NLL of the token following each position, on one forward pass."""
        arr = mx.array([ids])
        logits = model(arr[:, :-1]).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = arr[:, 1:]
        picked = mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]
        return {p: -float(picked[p]) for p in positions}

    if a.prompt_style == 'raw':
        if not a.open:
            raise SystemExit(
                "raw style is open-ended only: a base checkpoint will not "
                "follow a directed instruction (W5: instruct+raw recalled 0/8, "
                "and the converse holds too).")
        tasks = [(p, None) for p in OPEN_RAW]
    else:
        tasks = [(p, None) for p in OPEN] if a.open else TASKS
    if a.limit:
        tasks = tasks[:a.limit]
    rng = np.random.default_rng(a.seed)
    rows = []
    try:
        for ti, (prompt, spec) in enumerate(tasks):
            set_ban(None)
            if a.prompt_style == 'raw':
                pr = prompt + " "
            else:
                msg = [{"role": "user", "content": prompt}]
                try:
                    pr = tok.apply_chat_template(
                        msg, add_generation_prompt=True, tokenize=False,
                        enable_thinking=False)
                except TypeError:
                    pr = tok.apply_chat_template(
                        msg, add_generation_prompt=True, tokenize=False)
            text = generate(model, tok, prompt=pr, max_tokens=a.max_tokens,
                            verbose=False)
            full = pr + text
            ids = tok.encode(full)

            # locate each fact's answer token, and keep only facts the model
            # actually got right
            targets = []
            n_prompt = len(tok.encode(pr))
            found = scan_facts(text) if spec is None else [
                (d, e, r, answer_for(d, e, r)) for (d, e, r) in spec]
            seen_pos = set()
            for (dom, ent, rel, ans) in found:
                t = find_target(tok, ids, ans, n_prompt)
                # one target per position: two relations can share an answer
                # (Sweden.language 'Swedish' and a nationality, say), and
                # ablating the same position twice is not a crossover
                if t is not None and t >= n_prompt - 1 and t not in seen_pos:
                    seen_pos.add(t)
                    targets.append({'domain': dom, 'entity': ent,
                                    'relation': rel, 'answer': ans, 'pos': t})
            if len(targets) < 2:
                miss = ([f"{d}.{e}.{r}" for (d, e, r) in spec
                         if _norm(answer_for(d, e, r)) not in _norm(text)]
                        if spec else ['open scan found <2'])
                print(f"  [{ti}] only {len(targets)} usable target(s), "
                      f"skipped (absent: {miss or 'none — tokenisation'}) "
                      f"| said: {text.strip()[:90]!r}", flush=True)
                continue

            # per-position routing on the CLEAN pass
            sink['rows'] = {}
            sink['on'] = True
            mx.eval(model(mx.array([ids])[:, :-1]))
            sink['on'] = False
            experts = {}
            for tg in targets:
                experts[tg['pos']] = {
                    l: sink['rows'][l][tg['pos']][:a.k].tolist()
                    for l in range(n_moe)}

            pos = [t['pos'] for t in targets]
            set_ban(None)
            base = nll_at(ids, pos)

            damage = {}
            for banned in pos:
                set_ban(experts[banned])
                damage[banned] = nll_at(ids, pos)
            set_ban({l: rng.choice(E, a.k, replace=False).tolist()
                     for l in range(n_moe)})
            rand = nll_at(ids, pos)
            set_ban(None)

            for i, ti_ in enumerate(targets):
                for j, tj in enumerate(targets):
                    if i >= j:
                        continue
                    pi, pj = ti_['pos'], tj['pos']
                    rows.append({
                        'task': ti,
                        'i': f"{ti_['domain']}.{ti_['entity']}.{ti_['relation']}",
                        'j': f"{tj['domain']}.{tj['entity']}.{tj['relation']}",
                        'same_domain': ti_['domain'] == tj['domain'],
                        'd_ii': damage[pi][pi] - base[pi],
                        'd_ij': damage[pj][pi] - base[pi],
                        'd_jj': damage[pj][pj] - base[pj],
                        'd_ji': damage[pi][pj] - base[pj],
                        'd_i_rand': rand[pi] - base[pi],
                        'd_j_rand': rand[pj] - base[pj]})
            desc = ', '.join(f"{t['answer']} (base {base[t['pos']]:.4f})"
                             for t in targets)
            print(f"  [{ti}] {len(targets)} targets: {desc}", flush=True)
    finally:
        set_ban(None)
        for holder, name, gate in restore:
            setattr(holder, name, gate)

    if not rows:
        raise SystemExit("no usable pairs")

    own = np.array([r['d_ii'] for r in rows] + [r['d_jj'] for r in rows])
    oth = np.array([r['d_ij'] for r in rows] + [r['d_ji'] for r in rows])
    rnd = np.array([r['d_i_rand'] for r in rows] + [r['d_j_rand'] for r in rows])
    sel = own - oth

    print(f"\n  {len(rows)} within-passage pairs · {len(own)} directed tests\n")
    print(f"  dNLL own-position ablation   {own.mean():8.4f}")
    print(f"  dNLL other-position ablation {oth.mean():8.4f}")
    print(f"  dNLL random ablation         {rnd.mean():8.4f}")
    print(f"\n  selectivity (own - other)    {sel.mean():+8.4f}")
    print(f"  fraction own > other         {float((sel > 0).mean()):8.3f}")
    for lbl, m in (('same domain', True), ('cross domain', False)):
        s = [r for r in rows if r['same_domain'] == m]
        if not s:
            continue
        d = np.array([r['d_ii'] - r['d_ij'] for r in s]
                     + [r['d_jj'] - r['d_ji'] for r in s])
        print(f"    {lbl:13s} n={len(d):3d}  selectivity {d.mean():+.4f}  "
              f"frac>0 {float((d > 0).mean()):.3f}")

    tag = a.model.rstrip('/').split('/')[-1]
    dest = OUT / (f"generated.{'open' if a.open else 'directed'}.{tag}.json")
    dest.write_text(json.dumps({'k': a.k, 'n_pairs': len(rows),
                                'own': float(own.mean()),
                                'other': float(oth.mean()),
                                'random': float(rnd.mean()),
                                'selectivity': float(sel.mean()),
                                'frac_own_gt_other': float((sel > 0).mean()),
                                'rows': rows}, indent=1))
    print(f"\n  → {dest}")


if __name__ == '__main__':
    main()
