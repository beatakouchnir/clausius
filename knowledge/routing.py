"""Does expert SELECTION separate recall from derivation — against a real null?

W5.1b is the lead this project inherited. It reported that routing on the
answer token gives all-experts Jaccard 0.61-0.74 (no separation) but top-16
Jaccard **0.03-0.14**, and read the second number as near-disjoint selection:
"the router discriminates sharply even though the weights are redundant."

That reading has an arithmetic problem. Two INDEPENDENT random 16-subsets of
128 experts overlap by K^2/n = 2 on average, i.e. Jaccard ~0.067. Three of the
four reported layers sit at or below it. Near-disjointness is what picking 16
of 128 twice gives you for free — the null, not the finding.

So the claim needs controls. Three, all on identical footing:

  within-domain   split ONE domain's prompts in half and compare. Both sides
                  are the same distribution, so this is the ceiling a real
                  effect is measured against — not 1.0, because finite samples
                  disagree with themselves.
  across-domain   knowledge vs math: the recall/derivation contrast, as far as
                  these traces can proxy it.
  permutation     pool both domains' prompts, relabel at random, rebuild both
                  sides. This IS "the domain label carries no routing
                  information", and unlike K^2/n it accounts for the real skew
                  in expert usage (hot experts inflate overlap regardless of
                  domain) and for finite sample size.

Read it as: if selection discriminates, within > null > across, with the
observed across-J below the null's low percentiles. If the three land on top of
each other, W5.1b's top-16 number was set arithmetic.

TWO DESIGN POINTS THAT DECIDE WHETHER THE ANSWER MEANS ANYTHING:

1. **The unit is the prompt, n=8 per domain — not the ~900 decode tokens.**
   Tokens inside one generation share a topic and route alike. A first pass
   here permuted tokens and returned p<=0.05 at every single layer, which is
   the signature of a null built from non-independent units, not of a strong
   effect. Permuting whole prompts is the honest test and it is far less
   powerful. Low power is the true state of this data; it is reported, not
   hidden.

2. **Token budget is matched across every side.** Domains did not generate
   equally (gemma knowledge ran 11-121 tokens per prompt against math's flat
   121), and a top-K set built from fewer tokens is noisier, which DEPRESSES
   Jaccard by itself. Unmatched, the comparison would manufacture the very
   separation it is testing for. Every side here is S prompts x T tokens, with
   T the shortest prompt in play, so all three controls see identical budgets.

Honest scope, carried into every result this writes:
  - These are domain workloads (math/code/knowledge/prose, free generation),
    not the validated recall/derivation probes at the answer token. Domain is a
    PROXY for the W5.1b contrast, not the contrast itself.
  - Decode tokens across a whole generation, not one answer token.
  - The gemma trace is the INSTRUCT checkpoint; W5.1b's routing numbers came
    from BASE. Given W5.2, that is not a like-for-like re-check of W5.1b.

Usage:
  python3 -m knowledge.routing --model qwen [--k 8,16,32] [--perms 2000]
"""
import argparse
import json
import math
import random
from collections import Counter
from itertools import chain, combinations
from pathlib import Path

from . import traces

OUT = Path(__file__).resolve().parent.parent / 'records'

RECALL, DERIVE = 'knowledge', 'math'


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def analytic_null(K, n):
    """Jaccard of two INDEPENDENT uniform K-subsets of n. E|A&B| = K^2/n.

    Uniform-usage approximation, reported only to show the scale W5.1b's
    0.03-0.14 should have been read against. The permutation null supersedes
    it: real usage is skewed, which pushes true chance overlap higher.
    """
    inter = K * K / n
    return inter / (2 * K - inter)


class Sets:
    """Memoised top-K expert sets over prompt subsets, for one layer.

    Every control resamples from the same 2n prompts, so the same subset is
    scored repeatedly; without the cache the permutation pass dominates
    runtime.
    """

    def __init__(self, toks, K):
        self.toks, self.K, self.memo = toks, K, {}

    def __call__(self, idx):
        key = frozenset(idx)
        got = self.memo.get(key)
        if got is None:
            # two levels to flatten: prompt -> tokens -> the experts chosen for
            # that token. Flattening once counts whole per-token tuples as if
            # they were experts, which silently returns near-zero overlap
            # everywhere — including within a domain, where it cannot be real.
            c = Counter(e for i in key for tok in self.toks[i] for e in tok)
            # Ties broken by expert id, not Counter insertion order. At K=16 of
            # 256 with a long count-1 tail, ties are the common case; without
            # this, two runs on identical data can disagree.
            got = {e for e, _ in
                   sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:self.K]}
            self.memo[key] = got
        return got


def splits(n, s):
    """Unordered ways to cut range(n) into two disjoint s-subsets (s = n/2)."""
    first = range(n)
    return [(set(c), set(first) - set(c))
            for c in combinations(first, s) if 0 in c]


def layer_stats(toks, n_a, s, K, perms, rng):
    """within / across / null for one layer. toks[0:n_a] = A, toks[n_a:] = B."""
    sets = Sets(toks, K)
    n_b = len(toks) - n_a
    A, B = list(range(n_a)), list(range(n_a, len(toks)))

    within = [jaccard(sets([A[i] for i in x]), sets([A[i] for i in y]))
              for x, y in splits(n_a, s)]
    within += [jaccard(sets([B[i] for i in x]), sets([B[i] for i in y]))
               for x, y in splits(n_b, s)]

    across = [jaccard(sets(a), sets(b))
              for a in combinations(A, s) for b in combinations(B, s)]

    # H0: the domain label is arbitrary. Relabel the pooled prompts and rebuild
    # both sides exactly as `across` does.
    pool = A + B
    null = []
    for _ in range(perms):
        pick = rng.sample(pool, 2 * s)
        null.append(jaccard(sets(pick[:s]), sets(pick[s:])))
    null.sort()

    w = sum(within) / len(within)
    obs = sum(across) / len(across)
    p = (sum(1 for v in null if v <= obs) + 1) / (len(null) + 1)
    return {'within': round(w, 4), 'across': round(obs, 4),
            'null_median': round(null[len(null) // 2], 4),
            'null_p05': round(null[max(0, int(0.05 * len(null)) - 1)], 4),
            'p_value': round(p, 4), 'separation': round(w - obs, 4)}


def entropy(toks):
    """Shannon entropy of pooled expert usage in one layer. LOW = skewed.

    The control for the obvious confound. `separation` and ablation dK could
    correlate for an uninteresting reason: a layer whose routing is concentrated
    on few experts has both a more reproducible top-K (raising within-J) and
    more to lose when zeroed (raising dK). Without this, "routing separation
    predicts causal importance" could just be "skewed layers matter more".
    """
    c = Counter(e for p in toks for tok in p for e in tok)
    tot = sum(c.values())
    return -sum((v / tot) * math.log(v / tot) for v in c.values())


def partial(r_xy, r_xz, r_yz):
    """Correlation of x,y with z partialled out, on ranks."""
    den = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    return (r_xy - r_xz * r_yz) / den if den > 1e-9 else float('nan')


def spearman(x, y):
    """Rank correlation with average ties. No scipy — this project is stdlib."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def perm_p(x, y, rho, rng, n=5000):
    """Two-sided permutation p for a Spearman rho. n=30-40 layers is too few
    for the asymptotic table to be trustworthy."""
    ys, hits = list(y), 0
    for _ in range(n):
        rng.shuffle(ys)
        if abs(spearman(x, ys)) >= abs(rho):
            hits += 1
    return (hits + 1) / (n + 1)


def ablation_dk(src, path):
    """Per-layer knowledge damage, from whichever ablation record this is."""
    d = json.loads(path.read_text())
    if src['ablation_kind'] == 'layers':
        return {r['layer']: r['dK'] for r in d['per_layer']}
    return {r['layer']: r['d_knowledge'] for r in d['rows']
            if r['component'] == 'experts'}


def run(name, kind, K_list, perms, seed, tokens, force=False):
    src = traces.SOURCES[name]
    rd = traces.records_dir()
    if kind == 'gate' and not src['gate']:
        raise SystemExit(f"no gate trace for {name}")
    path = rd / (src['gate'] if kind == 'gate' else src['trace'])
    print(f"reading {path.name} …", flush=True)
    meta, idx = traces.load(path, kind)
    n_layers, n_exp = meta['n_layers'], meta['n_experts']
    for d in (RECALL, DERIVE):
        if d not in idx:
            raise SystemExit(f"need domains {RECALL}/{DERIVE}; got {sorted(idx)}")

    pa, pb = idx[RECALL], idx[DERIVE]
    lens = [len(p[0]) for p in pa + pb]
    T = min(lens) if tokens is None else min(tokens, min(lens))
    s = min(len(pa), len(pb)) // 2
    print(f"{src['label']} · {kind}-trace · {n_layers} layers, {n_exp} experts")
    print(f"prompts: {RECALL} {len(pa)}, {DERIVE} {len(pb)}  ·  "
          f"tokens/prompt {min(lens)}-{max(lens)}  ->  matched at T={T}")
    print(f"each side = {s} prompts x {T} tokens = {s * T} decode steps\n")

    dk = ablation_dk(src, rd / src['ablation'])
    ents = [entropy([p[l][:T] for p in pa + pb]) for l in range(n_layers)]
    out = {'model': src['label'], 'trace': path.name, 'kind': kind,
           'usage_entropy': [round(e, 4) for e in ents],
           'n_layers': n_layers, 'n_experts': n_exp,
           'n_prompts': {RECALL: len(pa), DERIVE: len(pb)},
           'tokens_per_prompt': T, 'prompts_per_side': s, 'perms': perms,
           'domains': [RECALL, DERIVE], 'by_k': {}}

    for K in K_list:
        rng = random.Random(seed)
        rows = []
        print(f"  K={K}  (uniform-random Jaccard would be "
              f"{analytic_null(K, n_exp):.4f})")
        print(f"  {'layer':>5s} {'within':>8s} {'across':>8s} {'null med':>9s} "
              f"{'null p05':>9s} {'p':>7s} {'sep':>8s}")
        for l in range(n_layers):
            toks = [p[l][:T] for p in pa] + [p[l][:T] for p in pb]
            st = layer_stats(toks, len(pa), s, K, perms, rng)
            st['layer'] = l
            rows.append(st)
            print(f"  {l:5d} {st['within']:8.4f} {st['across']:8.4f} "
                  f"{st['null_median']:9.4f} {st['null_p05']:9.4f} "
                  f"{st['p_value']:7.4f} {st['separation']:+8.4f}", flush=True)

        sig = [r['layer'] for r in rows if r['p_value'] <= 0.05]
        med = lambda f: sorted(r[f] for r in rows)[len(rows) // 2]  # noqa: E731
        common = [l for l in range(n_layers) if l in dk]
        y = [dk[l] for l in common]
        sep = [rows[l]['separation'] for l in common]
        ent = [ents[l] for l in common]
        rho = spearman(sep, y)
        rp = perm_p(sep, y, rho, random.Random(seed))
        r_se, r_ey = spearman(sep, ent), spearman(ent, y)
        par = partial(rho, r_se, r_ey)

        print(f"\n  median  within {med('within'):.4f} · "
              f"across {med('across'):.4f} · null {med('null_median'):.4f}")
        print(f"  layers separating below the null at p<=0.05: "
              f"{len(sig)}/{n_layers}  {sig if sig else ''}")
        print(f"  separation vs ablation dK: Spearman {rho:+.3f} "
              f"(permutation p {rp:.3f}, n={len(common)})")
        print(f"  usage-skew control: rho(sep,entropy) {r_se:+.3f} · "
              f"rho(entropy,dK) {r_ey:+.3f} · partial(sep,dK|entropy) "
              f"{par:+.3f}\n")

        out['by_k'][str(K)] = {
            'analytic_uniform_null': round(analytic_null(K, n_exp), 4),
            'median_within': med('within'), 'median_across': med('across'),
            'median_null': med('null_median'),
            'n_layers_p05': len(sig), 'sig_layers': sig,
            'spearman_sep_vs_dK': round(rho, 3),
            'spearman_perm_p': round(rp, 4),
            'skew_control': {'rho_sep_entropy': round(r_se, 3),
                             'rho_entropy_dK': round(r_ey, 3),
                             'partial_sep_dK_given_entropy': round(par, 3)},
            'rows': rows,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"routing_null.{name}.{kind}.json"
    # An exploratory run at --perms 20 must not overwrite the canonical record.
    # This has bitten twice: the filename carries no settings, so a quick check
    # silently replaced a 2000-permutation result with a 20-permutation one.
    if dest.exists() and not force:
        old = json.loads(dest.read_text())
        if (old.get('perms', 0) > perms
                or set(old.get('by_k', {})) > set(out['by_k'])):
            raise SystemExit(
                f"refusing to overwrite {dest.name}: on disk is perms="
                f"{old.get('perms')} K={sorted(old.get('by_k', {}), key=int)}, "
                f"this run is perms={perms} K={sorted(out['by_k'], key=int)}. "
                f"Pass --force to replace it.")
    dest.write_text(json.dumps(out, indent=1))
    print(f"  → {dest}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='qwen', choices=sorted(traces.SOURCES))
    ap.add_argument('--kind', default='expert', choices=('expert', 'gate'))
    ap.add_argument('--k', default='8,16,32')
    ap.add_argument('--perms', type=int, default=2000,
                    help='label permutations per layer. W5.1a found a single '
                         'random control inadequate; same lesson, applied to '
                         'a null distribution.')
    ap.add_argument('--tokens', type=int, default=None,
                    help='cap tokens per prompt (default: the shortest prompt, '
                         'so every side gets an identical budget)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true',
                    help='overwrite a record built with more permutations')
    a = ap.parse_args()
    run(a.model, a.kind, [int(x) for x in a.k.split(',') if x.strip()],
        a.perms, a.seed, a.tokens, a.force)


if __name__ == '__main__':
    main()
