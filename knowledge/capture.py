"""Capture per-token routing over the R2 probe suite. The one GPU step.

Writes everything the downstream analysis could want, because a second capture
session costs another GPU window: for every probe, at every layer, at every
token position — the router's top-M expert ranks and their scores, plus the
labels that make the analyses possible.

WHAT IS RECORDED AND WHY EACH LABEL EARNS ITS PLACE:

  fact_id, para   same fact across paraphrases -> rung 3, is a fact's routing
                  signature stable across surface forms and distinct across
                  facts? Impossible to reconstruct later without these.
  cls             recall | derive -> the meter's target label.
  dkind           year | arith | letters -> is the meter keyed to "not
                  retrieval", or merely to one kind of computation?
  matched         shares vocabulary with its recall counterpart -> the topic
                  baseline the meter must beat.
  correct         did greedy decoding produce the right first answer token?
  answer_nll      the model's confidence on the answer it gave.

EVERY PROBE IS CAPTURED, INCLUDING THE ONES THE MODEL GETS WRONG. quantize's
`validate()` drops failures because an NLL delta on a probe the model cannot
solve measures noise — correct for ablation, wrong here. The failures are the
positive class for the eventual hallucination test: routing on a fact the model
does not actually know is the signal of interest, not contamination. The
`correct` flag lets analysis filter; discarding at capture time would not let
it unfilter.

ONE FORWARD PASS PER PROBE does all of it. Teacher-forcing prompt+answer gives
the routing at every position AND, at index n_prompt-1, the logits that decide
the first answer token — so correctness and NLL come free rather than costing a
second pass.

PROMPT STYLE: chat, with ONE uniform instruction for both classes. W5's trap
was that raw completion on an instruct checkpoint recalls 0/8 and emits generic
filler, so it measures formatting rather than knowledge; and a different
wrapper per class would confound the very contrast being measured.

Needs mlx-lm. The analysis half of this package does not — keep it that way.

Usage:
  python3 -m knowledge.capture --model artifacts/qwen36-35b-a3b-4bit-g64
  python3 -m knowledge.capture --model PATH --limit 8      # smoke run
"""
import argparse
import gzip
import json
import re
import time
from pathlib import Path

from .probes import all_probes
from .seam import find_gates, gate_output, describe

OUT = Path(__file__).resolve().parent.parent / 'records'


def answer_text(answer, style):
    """The answer as the model will actually emit it.

    Probe answers are stored with the raw-completion leading space (" Canberra")
    because that is how they continue a bare stem. After a chat template's
    generation prompt there is no leading space, and " Canberra" tokenises to a
    DIFFERENT first token than "Canberra" (66463 vs 6503) — so a chat-style
    capture that keeps the space scores every probe against a token the model
    will never emit and reports 0 correct. `ablate.py` strips for the same
    reason; this is that lesson, re-paid.
    """
    return answer.strip() if style == 'chat' else answer


def build_prompt(tok, stem, style):
    """One uniform wrapper for BOTH probe classes — see module docstring."""
    if style == 'raw':
        return stem
    q = f"Complete this sentence with only the missing word: {stem} ___"
    msg = [{"role": "user", "content": q}]
    try:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False)


class Recorder:
    """Wraps a router module, passing its output through and recording it.

    Wrapping the INSTANCE rather than patching the class matters: qwen's gate
    is an `nn.Linear`, and patching `nn.Linear.__call__` would instrument every
    linear layer in the model, not the 40 routers.
    """

    def __init__(self, inner, layer, sink, top_m):
        self.inner, self.layer, self.sink, self.top_m = inner, layer, sink, top_m

    def __call__(self, x, *a, **kw):
        out = self.inner(x, *a, **kw)
        if self.sink.get('on'):
            import mlx.core as mx
            ranks, scores, kind = gate_output(out, self.top_m)
            mx.eval(ranks, scores)
            self.sink['kind'] = kind
            self.sink['rows'][self.layer] = (
                [[int(v) for v in row] for row in
                 ranks.reshape(-1, ranks.shape[-1]).tolist()],
                [[round(float(v), 4) for v in row] for row in
                 scores.reshape(-1, scores.shape[-1]).tolist()])
        return out

    def __getattr__(self, name):
        # keep the wrapped module's own attributes reachable
        return getattr(object.__getattribute__(self, 'inner'), name)


def run_capture(model, tok, probes, dest, style='chat', top_m=32, meta=None):
    """Instrument, sweep the probes, write the trace. Returns (n_correct, n)."""
    import mlx.core as mx

    n_moe, n_exp, top_k = describe(model)
    gates = find_gates(model)
    print(f"{n_moe} MoE layers · {n_exp} experts · top_k {top_k} · "
          f"{len(gates)} routers located", flush=True)

    sink = {'on': False, 'rows': {}, 'kind': None}
    holders = []
    for li, holder, name, gate in gates:
        holders.append((holder, name, gate))
        setattr(holder, name, Recorder(gate, li, sink, top_m))

    dest.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    t0 = time.time()
    try:
        with gzip.open(dest, 'wt') as f:
            f.write(json.dumps({
                'schema': 'probe-gate/1',
                'model': (meta or {}).get('model', '?'),
                'prompt_style': style, 'top_m': top_m,
                'n_layers': n_moe, 'n_experts': n_exp, 'top_k': top_k,
                'n_probes': len(probes)}) + '\n')

            for i, p in enumerate(probes):
                prompt = build_prompt(tok, p['stem'], style)
                ans = answer_text(p['answer'], style)
                p_ids = tok.encode(prompt)
                a_ids = tok.encode(prompt + ans)[len(p_ids):]
                if not a_ids:
                    continue
                ids = mx.array([p_ids + a_ids])

                sink['rows'] = {}
                sink['on'] = True
                logits = model(ids[:, :-1]).astype(mx.float32)
                sink['on'] = False

                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                tgt = ids[:, 1:]
                picked = mx.take_along_axis(
                    logprobs, tgt[..., None], axis=-1)[0, :, 0]
                nll = -float(mx.mean(picked[len(p_ids) - 1:]))
                # Confidence at the decisive step, read off the PREDICTIVE
                # DISTRIBUTION rather than off the labelled answer. `answer_nll`
                # is useless wherever the label is a placeholder — the fictional
                # probes have no true answer — so a membership or fabrication
                # analysis needs these instead. They are also the standard
                # baseline that a routing-based claim has to beat.
                lp = logprobs[0, len(p_ids) - 1]
                pr = mx.exp(lp)
                top2 = mx.sort(lp)[-2:]
                conf = {'top1_prob': round(float(mx.max(pr)), 6),
                        'entropy': round(-float(mx.sum(pr * lp)), 6),
                        'margin': round(float(top2[1] - top2[0]), 6)}
                # the step at n_prompt-1 is the one that decides the first
                # answer token — the W5 "answer token" position
                top1 = int(mx.argmax(logits[0, len(p_ids) - 1]))
                correct = bool(top1 == a_ids[0])
                # LENIENT match, recorded alongside the strict one rather than
                # replacing it. The strict first-token-id test rejects answers
                # the model gets RIGHT but spells differently: it emits
                # 'Baroque' where the probe says 'baroque', and ' metal' with a
                # leading space where the probe says 'metal'. Those are
                # authoring mismatches, not ignorance, and discarding them
                # would shrink the usable grid for no scientific reason. Both
                # flags are stored so any analysis can choose its own strictness.
                got = tok.decode([top1]).strip().lower()
                want_l = ans.strip().lower()
                lenient = bool(got and (want_l.startswith(got)
                                        or got.startswith(want_l)))
                n_ok += correct

                rec = {k: p[k] for k in ('probe_id', 'fact_id', 'domain',
                                         'cls', 'para', 'dkind', 'matched',
                                         'suite', 'atype')}
                for k in ('entity', 'relation'):
                    if k in p:
                        rec[k] = p[k]
                rec.update({'n_prompt': len(p_ids), 'n_answer': len(a_ids),
                            # digit groups in the probe text: the `computed`
                            # class must carry an operand, so this is a cue the
                            # analysis has to rule out the same way it rules
                            # out length
                            'n_numbers': len(re.findall(r'\d+', p['stem'])),
                            'predict_pos': len(p_ids) - 1,
                            'correct': correct, 'correct_lenient': lenient,
                            'answer_nll': round(nll, 5),
                            **conf,
                            'kind': sink['kind'],
                            'ranks': {str(l): v[0]
                                      for l, v in sorted(sink['rows'].items())},
                            'scores': {str(l): v[1]
                                       for l, v in sorted(sink['rows'].items())}})
                f.write(json.dumps(rec) + '\n')
                if (i + 1) % 25 == 0 or i + 1 == len(probes):
                    print(f"  {i + 1:4d}/{len(probes)}  correct so far "
                          f"{n_ok}  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for holder, name, gate in holders:
            setattr(holder, name, gate)

    print(f"\n  {n_ok}/{len(probes)} probes answered correctly")
    print(f"  routers returned '{sink['kind']}' "
          f"({'full ranking' if sink['kind'] == 'scores' else 'top-k only'})")
    print(f"  → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return n_ok, len(probes)


def _selftest():
    """Drive the whole pipeline on a stub model. No checkpoint is loaded.

    Exercises the parts that are expensive to get wrong on the GPU: that
    wrapping the router instance actually fires, that every layer lands in the
    record, that ranks/scores line up with the token count, and that the
    answer-token index is where the labels say it is.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    H, E, V, L = 16, 8, 256, 3

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(H, E)
            self.switch_mlp = SwitchGLU(H, 2 * H, E)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer() for _ in range(L)]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.model = Inner()
            self.head = nn.Linear(H, V)

        def __call__(self, ids):
            h = self.embed(ids)
            for layer in self.model.layers:
                h = h + layer.mlp.gate(h).sum(-1, keepdims=True) * 0.0
            return self.head(h)

    class Tok:
        def encode(self, s):
            return [1] + [ord(c) % (V - 2) + 1 for c in s]

        def apply_chat_template(self, msg, **kw):
            return "U: " + msg[0]['content'] + "\nA:"

    probes = all_probes()[:4]
    dest = Path('/tmp/_probe_gate_selftest.jsonl.gz')
    model, tok = Model(), Tok()
    mx.eval(model.parameters())
    run_capture(model, tok, probes, dest, top_m=4,
                meta={'model': 'stub'})

    with gzip.open(dest, 'rt') as f:
        head = json.loads(f.readline())
        recs = [json.loads(x) for x in f]
    ok = head['n_layers'] == L and len(recs) == len(probes)
    print(f"  meta layers {head['n_layers']} · records {len(recs)}"
          f"  {'OK' if ok else 'FAIL'}")
    for r in recs[:1]:
        n_tok = r['n_prompt'] + r['n_answer'] - 1        # forward saw ids[:-1]
        shapes = {l: (len(v), len(v[0])) for l, v in r['ranks'].items()}
        good = (len(r['ranks']) == L
                and all(s == (n_tok, 4) for s in shapes.values())
                and r['predict_pos'] == r['n_prompt'] - 1
                and set(('fact_id', 'cls', 'dkind', 'matched', 'correct',
                         'answer_nll')) <= set(r))
        ok &= good
        print(f"  {r['probe_id']}: layers {len(r['ranks'])} · "
              f"ranks/layer {sorted(set(shapes.values()))} · tokens {n_tok} · "
              f"predict_pos {r['predict_pos']}  {'OK' if good else 'FAIL'}")
    dest.unlink(missing_ok=True)
    print("selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='artifacts/qwen36-35b-a3b-4bit-g64')
    ap.add_argument('--prompt-style', default='chat', choices=('chat', 'raw'))
    ap.add_argument('--top-m', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='first N probes only')
    ap.add_argument('--suite', default='mechanism',
                    choices=('mechanism', 'grounding', 'computation', 'grid',
                             'grid2', 'hallucination', 'all'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true',
                    help='validate the pipeline on a stub model, no checkpoint')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())

    from mlx_lm import load
    print(f"loading {a.model} …", flush=True)
    model, tok = load(a.model)
    probes = all_probes(a.suite)
    if a.limit:
        probes = probes[:a.limit]
    tag = a.model.rstrip('/').split('/')[-1]
    dest = (Path(a.out) if a.out else
            OUT / f"probe_gate.{a.suite}.{tag}.jsonl.gz")
    run_capture(model, tok, probes, dest, style=a.prompt_style,
                top_m=a.top_m, meta={'model': a.model})


if __name__ == '__main__':
    main()
