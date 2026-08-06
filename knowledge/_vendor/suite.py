"""Task loaders, prompt builder and scorers for the benchmark suite.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier benchmark harness
Same authorship and license as the rest of this repo (Apache-2.0); see NOTICE.

Two things are deliberately NOT bundled, and both degrade rather than crash:
  ifeval    scoring needs Google Research's IFEval registry (Apache-2.0, public
            at google-research/instruction_following_eval). score() returns None
            when it is absent — skip the item, do not guess.
  longbench needs multifieldqa_en.jsonl; point LONGBENCH_JSONL at a copy.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

# LongBench multifieldqa_en, one JSON object per line. Not bundled: set
# LONGBENCH_JSONL to a copy if you need the longbench task.
LONGBENCH = Path(os.environ.get('LONGBENCH_JSONL', '')) if os.environ.get(
    'LONGBENCH_JSONL') else None

TASKS = ('gsm8k', 'mmlu_pro', 'humaneval', 'popqa', 'longbench', 'ifeval')


# ---------------------------------------------------------------- loaders ---


IFEVAL_HELP = """ifeval scoring needs Google Research's instruction_following_eval
registry (Apache-2.0). It is not bundled here and is not on PyPI. Fetch it once,
as a package named `_ifeval_official` importable from where you run:

    mkdir -p _ifeval_official && touch _ifeval_official/__init__.py
    B=https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval
    for f in instructions.py instructions_registry.py instructions_util.py; do
        curl -sSo _ifeval_official/$f $B/$f
    done

Every other task is unaffected."""


def ifeval_registry():
    """The IFEval registry, or a clear failure.

    Checked BEFORE a run starts rather than at scoring time: without it every
    item scores as None, and a None verdict read as a boolean is False — an
    unscoreable item would be recorded as a wrong answer and the arm would
    report near-zero accuracy that looks like catastrophic damage.
    """
    try:
        from _ifeval_official import instructions_registry as reg
        return reg
    except ImportError:
        raise SystemExit(IFEVAL_HELP)


def load_items(task, n, seed=0):
    from datasets import load_dataset
    if task == 'gsm8k':
        ds = load_dataset('openai/gsm8k', 'main', split='test').shuffle(seed=seed)
        return [{'prompt': r['question'],
                 'instruct': "Solve it. End with the final answer on its own "
                             "line as: ANSWER: <number>",
                 'gold': r['answer'].split('####')[-1].strip().replace(',', ''),
                 'score': 'number', 'max_tokens': 512}
                for r in ds.select(range(min(n, len(ds))))]
    if task == 'mmlu_pro':
        ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test').shuffle(seed=seed)
        out = []
        for r in ds.select(range(min(n, len(ds)))):
            letters = 'ABCDEFGHIJ'[:len(r['options'])]
            opts = "\n".join(f"{c}. {o}" for c, o in zip(letters, r['options']))
            out.append({'prompt': f"{r['question']}\n{opts}",
                        'instruct': "End with the answer letter on its own line "
                                    "as: ANSWER: <letter>",
                        'gold': r['answer'], 'score': 'letter',
                        'letters': letters, 'max_tokens': 512})
        return out
    if task == 'popqa':
        ds = load_dataset('akariasai/popqa', split='test').shuffle(seed=seed)
        return [{'prompt': r['question'],
                 'instruct': "Answer with just the entity name, nothing else.",
                 'gold': json.loads(r['possible_answers']),
                 'score': 'alias', 'max_tokens': 48}
                for r in ds.select(range(min(n, len(ds))))]
    if task == 'humaneval':
        ds = load_dataset('evalplus/humanevalplus', split='test')
        return [{'prompt': r['prompt'],
                 'instruct': "Complete the function. Reply with ONLY the full "
                             "function definition in a ```python block.",
                 'gold': {'test': r['test'], 'entry': r['entry_point']},
                 'score': 'exec', 'max_tokens': 768}
                for r in list(ds)[:n]]
    if task == 'longbench':
        if LONGBENCH is None or not LONGBENCH.exists():
            raise SystemExit('longbench needs LONGBENCH_JSONL pointing at '
                             'multifieldqa_en.jsonl; it is not bundled')
        rows = [json.loads(l) for l in LONGBENCH.open() if l.strip()][:n]
        return [{'prompt': f"{r['context']}\n\nQuestion: {r['input']}",
                 'instruct': "Answer from the document in a few words.",
                 'gold': r['answers'], 'score': 'f1', 'max_tokens': 96}
                for r in rows]
    if task == 'ifeval':
        ifeval_registry()          # fail now, not after a GPU run scores zeros
        ds = load_dataset('google/IFEval', split='train').shuffle(seed=seed)
        return [{'prompt': r['prompt'], 'instruct': None,
                 'gold': {'ids': r['instruction_id_list'], 'kwargs': r['kwargs']},
                 'score': 'ifeval', 'max_tokens': 768}
                for r in ds.select(range(min(n, len(ds))))]
    raise SystemExit(f"unknown task {task}")


# ---------------------------------------------------------------- scorers ---

def _norm(s):
    return re.sub(r'[^a-z0-9 ]', '', (s or '').lower()).strip()


def run_code(body, test, entry):
    """Execute a candidate solution against its tests in a subprocess."""
    src = f"{body}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def score(item, text):
    kind = item['score']
    if kind == 'number':
        m = re.findall(r'ANSWER:\s*([^\n]+)', text)
        tail = m[-1] if m else text.strip()[-40:]
        nums = re.findall(r'-?\d[\d,]*\.?\d*', tail.replace(',', ''))
        return bool(nums) and nums[-1].rstrip('.') == item['gold']
    if kind == 'letter':
        m = re.findall(r'ANSWER:\s*([A-Za-z])', text)
        if m:
            return m[-1].upper() == item['gold']
        hits = re.findall(rf"\b([{item['letters']}])\b", text.upper())
        return bool(hits) and hits[-1] == item['gold']
    if kind == 'f1':
        # LongBench's standard metric. Containment is wrong here: the model
        # answered "South West Ultras" against gold "South West Ultras fan
        # club.", i.e. a MORE CONCISE correct answer, which gold-in-output
        # containment scores zero. Token F1 handles partial overlap in both
        # directions and needs no arbitrary threshold.
        pred = _norm(text).split()
        best = 0.0
        for g in item['gold']:
            gt = _norm(g).split()
            if not pred or not gt:
                continue
            common = 0
            pool = list(gt)
            for w in pred:
                if w in pool:
                    pool.remove(w)
                    common += 1
            if common:
                pr_, rc = common / len(pred), common / len(gt)
                best = max(best, 2 * pr_ * rc / (pr_ + rc))
        return best
    if kind == 'alias':
        t = _norm(text)
        return any(_norm(a) and _norm(a) in t for a in item['gold'])
    if kind == 'exec':
        blocks = re.findall(r'```(?:python)?\s*(.*?)```', text, re.S)
        body = blocks[-1] if blocks else text
        return run_code(body, item['gold']['test'], item['gold']['entry'])
    if kind == 'ifeval':
        try:
            from _ifeval_official import instructions_registry as reg
        except Exception:
            return None                      # registry unavailable: skip, do not guess
        ok = True
        for iid, kw in zip(item['gold']['ids'], item['gold']['kwargs']):
            try:
                cls = reg.INSTRUCTION_DICT[iid]
                inst = cls(iid)
                inst.build_description(**{k: v for k, v in (kw or {}).items()
                                          if v is not None})
                if not inst.check_following(text):
                    ok = False
            except Exception:
                return None
        return ok
    raise SystemExit(f"unknown scorer {kind}")


# ------------------------------------------------------------------- run ---

THINK_MULT = 4      # CoT generates ~3.3x more tokens on an easy item; 4x with
                    # headroom. Running CoT at the non-CoT budget TRUNCATES the
                    # chain and produces a fake accuracy collapse — an artefact
                    # of the cap, not a property of the model.


def build_prompt(tok, item, think=False):
    content = item['prompt'] if not item['instruct'] else \
        f"{item['prompt']}\n\n{item['instruct']}"
    msg = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=think)
    except TypeError:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False)


def run_arm(task, arm, items, alloc, think=False):
    from mlx_lm import load, generate
    from .offload_model import wrap
    cfg = ARMS[arm]
    mx.clear_cache()
    model, tok = load(cfg['model'])
    wrapped = 0
    if cfg['capacity']:
        wrapped = wrap(model, cfg['capacity'], STORE, 'exact', None, alloc)
    mx.eval(model.parameters())
    mx.clear_cache()
    mx.reset_peak_memory()

    correct = skipped = 0
    gen_tokens = 0
    per_item = []          # paired significance needs these; the first suite
                           # run saved only aggregates and McNemar was impossible
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        pr = build_prompt(tok, it, think)
        budget = it['max_tokens'] * (THINK_MULT if think else 1)
        txt = generate(model, tok, prompt=pr, max_tokens=budget, verbose=False)
        gen_tokens += len(tok.encode(txt))
        s = score(it, txt)
        per_item.append(None if s is None else float(s))
        if s is None:
            skipped += 1
        else:
            correct += float(s)          # bool or F1 in [0,1]
        if (i + 1) % 25 == 0:
            done = i + 1 - skipped
            print(f"      {i+1}/{len(items)}  score {correct/max(done,1):.3f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    secs = time.perf_counter() - t0
    scored = len(items) - skipped
    peak = mx.get_peak_memory() / 1e9
    # Resident is measured AFTER the run, with the expert cache warm. Measuring
    # it before generation gave 1.36 GB for the offload arm — just the
    # non-expert weights, because the cache starts empty and its tensors are not
    # module parameters. Correct at that instant, useless as a footprint.
    mx.clear_cache()
    resident = mx.get_active_memory() / 1e9
    cache_gb = mx.get_cache_memory() / 1e9
    del model
    mx.clear_cache()
    return {'accuracy': correct / scored if scored else None,
            'n_scored': scored, 'n_skipped': skipped,
            'seconds': round(secs, 1),
            'resident_gb': round(resident, 2), 'peak_gb': round(peak, 2),
            'alloc_cache_gb': round(cache_gb, 2),
            'gen_tokens': gen_tokens, 'tok_s': round(gen_tokens / secs, 1),
            'sec_per_item': round(secs / len(items), 2),
            'wrapped_layers': wrapped, 'per_item': per_item}


def report():
    if not RESULT.exists():
        return "no results yet"
    d = json.loads(RESULT.read_text())
    lines = ["| task | arm | accuracy | n | tok/s | s/item | resident GB | peak GB |",
             "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for arm in ARMS:
            r = d.get(f"{task}/{arm}")
            if not r:
                continue
            acc = f"{r['accuracy']:.3f}" if r['accuracy'] is not None else "—"
            lines.append(f"| {task} | {arm} | **{acc}** | {r['n_scored']} | "
                         f"{r['tok_s']} | {r['sec_per_item']} | "
                         f"{r.get('resident_gb', '—')} | {r['peak_gb']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--arms', default='control,offload,e4b')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--think', action='store_true',
                    help='enable_thinking=True and a 4x token budget')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()

    if a.report:
        print(report())
        return

    tasks = TASKS if a.tasks == 'all' else tuple(a.tasks.split(','))
    arms = tuple(a.arms.split(','))
    alloc = None
    bf = OUT / 'budget.json'
    if bf.exists():
        alloc = {int(k): v for k, v in
                 json.loads(bf.read_text())['alloc'].items()}

    OUT.mkdir(parents=True, exist_ok=True)
    done = json.loads(RESULT.read_text()) if RESULT.exists() else {}

    for task in tasks:
        try:
            items = load_items(task, a.n)
        except Exception as e:
            print(f"[{task}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        for arm in arms:
            key = f"{task}/{arm}" + ("/think" if a.think else "")
            if key in done:
                print(f"[{key}] already done, skipping", flush=True)
                continue
            print(f"\n[{key}] {len(items)} items", flush=True)
            try:
                res = run_arm(task, arm, items, alloc, a.think)
                res['task'], res['arm'], res['n_requested'] = task, arm, a.n
                res['think'] = a.think
                done[key] = res
                RESULT.write_text(json.dumps(done, indent=1))
                print(f"    -> acc {res['accuracy']}  {res['tok_s']} tok/s  "
                      f"resident {res['resident_gb']} / peak {res['peak_gb']} GB"
                      f"  {res['seconds']}s", flush=True)
            except Exception as e:
                print(f"[{key}] FAILED: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
    print("\n" + report())


if __name__ == '__main__':
    main()
