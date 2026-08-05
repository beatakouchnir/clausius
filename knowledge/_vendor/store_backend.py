"""Disk-backed expert store.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier MoE offload runtime
Same authorship and licence as the rest of this repo (Apache-2.0); see NOTICE.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJS = ('gate_proj', 'up_proj', 'down_proj')
FIELDS = ('weight', 'scales', 'biases')


def _to_numpy(arr):
    """MLX array -> numpy, routing bf16/fp16 through raw uint16."""
    if arr.dtype in (mx.bfloat16, mx.float16):
        return np.array(arr.view(mx.uint16)), str(arr.dtype).split('.')[-1]
    return np.array(arr), str(arr.dtype).split('.')[-1]


def _from_numpy(a, dtype_name):
    out = mx.array(a)
    if dtype_name in ('bfloat16', 'float16'):
        out = out.view(getattr(mx, dtype_name))
    return out


def dump(model, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    index, total = {}, 0
    for li, layer in enumerate(layers):
        ex = getattr(layer, 'experts', None)
        if ex is None or not hasattr(ex, 'switch_glu'):
            continue
        for proj in PROJS:
            lin = getattr(ex.switch_glu, proj)
            for field in FIELDS:
                if field not in lin or lin[field] is None:
                    continue
                np_arr, dt = _to_numpy(lin[field])
                path = out_dir / f"L{li}.{proj}.{field}.npy"
                np.save(path, np_arr)
                # basename only: paths are resolved against the store dir at
                # read time, so the store keeps working regardless of the
                # process cwd. Absolute-at-dump-time paths broke the moment the
                # eval shim chdir'd into the harness clone.
                index[f"{li}.{proj}.{field}"] = {'file': path.name, 'dtype': dt}
                total += np_arr.nbytes
    (out_dir / 'index.json').write_text(json.dumps(index, indent=1))
    return total, len(index)


class DiskStore:
    """Memory-mapped expert weights. One expert slice per fetch."""

    def __init__(self, out_dir, layer):
        out_dir = Path(out_dir).resolve()
        self.dir = out_dir
        self.index = json.loads((out_dir / 'index.json').read_text())
        self.layer = layer
        self._mm = {}

    def _mmap(self, key):
        if key not in self._mm:
            e = self.index[key]
            f = e.get('file') or Path(e['path']).name
            self._mm[key] = (np.load(self.dir / f, mmap_mode='r'), e['dtype'])
        return self._mm[key]

    def has(self, proj, field):
        return f"{self.layer}.{proj}.{field}" in self.index

    def fetch(self, proj, field, expert):
        mm, dt = self._mmap(f"{self.layer}.{proj}.{field}")
        return _from_numpy(np.ascontiguousarray(mm[expert]), dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', default='artifacts/26b-a4b-4bit-g64')
    ap.add_argument('--out', default='artifacts/expert_store')
    ap.add_argument('--verify', action='store_true',
                    help='read every tensor back and compare against the model')
    a = ap.parse_args()

    from mlx_lm import load
    print(f"loading {a.model} …", flush=True)
    model, _tok = load(a.model)
    n_bytes, n_tensors = dump(model, a.out)
    print(f"  wrote {n_bytes / 1e9:.2f} GB across {n_tensors} tensors -> {a.out}")

    if a.verify:
        layers = model.model.layers if hasattr(model, 'model') else model.layers
        worst = 0.0
        for li in (0, len(layers) // 2, len(layers) - 1):
            ex = getattr(layers[li], 'experts', None)
            if ex is None or not hasattr(ex, 'switch_glu'):
                continue
            st = DiskStore(a.out, li)
            for proj in PROJS:
                lin = getattr(ex.switch_glu, proj)
                for field in FIELDS:
                    if field not in lin or lin[field] is None:
                        continue
                    for e in (0, 7, 63, 127):
                        got = st.fetch(proj, field, e)
                        want = lin[field][e]
                        d = float(mx.max(mx.abs(got.astype(mx.float32)
                                                - want.astype(mx.float32))))
                        worst = max(worst, d)
        print(f"  round-trip max|diff| over sampled experts: {worst:.3e} "
              f"({'EXACT' if worst == 0 else 'LOSSY — dtype handling is wrong'})")


if __name__ == '__main__':
    main()
