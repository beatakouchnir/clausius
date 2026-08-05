"""Expert cache with a contiguous resident tensor — the offload runtime.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier MoE offload runtime —
        ExpertCache and OffloadSwitchGLU only; the benchmark CLI is not vendored
Same authorship and licence as the rest of this repo (Apache-2.0); see NOTICE.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""
import json
import time
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort


HIDDEN, MOE_INTER = 2816, 704
N_EXPERTS, TOP_K, N_LAYERS = 128, 8, 30
GROUP, BITS = 64, 4


class ExpertCache:
    """Contiguous resident slots over one layer's experts, LRU eviction."""

    def __init__(self, glu, capacity, disk=None, policy='exact'):
        """disk: a store_backend.DiskStore, or None to keep weights in memory.

        With a DiskStore this holds NO reference to the model's expert tensors,
        only the resident slots — which is the difference between measuring the
        mechanism and actually saving memory. Keeping the in-memory store around
        made wrapping ADD 6.4 GB, and a k=8 eval run OOM'd Metal because of it.
        """
        self.n_experts = getattr(glu, 'gate_proj')['weight'].shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.projs = ('gate_proj', 'up_proj', 'down_proj')
        self.disk = disk
        self.store, self.resident = {}, {}
        for name in self.projs:
            lin = getattr(glu, name)
            has_b = 'biases' in lin and lin['biases'] is not None
            w, s = lin['weight'], lin['scales']
            b = lin['biases'] if has_b else None
            if disk is None:
                self.store[name] = (w, s, b)
            # capacity + 1 slots. The trailing slot stays zero-filled: with
            # affine quantization a zero scale and zero bias dequantize to zero,
            # so a gather that lands there contributes nothing. That lets a
            # non-resident expert be expressed as an INDEX rather than as a mask
            # over the output — the mask/clamp/multiply version measured ~36%,
            # as costly as the sync it was meant to replace.
            # ONLY the static policy gets the extra slot: widening the resident
            # tensor from 128 to 129 rows changes gather_qmm's tiling and
            # perturbs the result by ~1.3e-3, which would cost exact paging its
            # bit-exactness at full capacity for no benefit (exact never lands
            # on the zero slot, because ensure() installs first).
            n_slots = self.capacity + (1 if policy == 'static' else 0)
            self.resident[name] = [
                mx.zeros((n_slots,) + w.shape[1:], dtype=w.dtype),
                mx.zeros((n_slots,) + s.shape[1:], dtype=s.dtype),
                None if b is None else mx.zeros((n_slots,) + b.shape[1:],
                                                dtype=b.dtype),
            ]
        self.group = glu.gate_proj.group_size
        self.bits = glu.gate_proj.bits
        self.mode = glu.gate_proj.mode
        self.slot_of = OrderedDict()          # expert id -> slot, LRU ordered
        self.free = list(range(self.capacity))
        self.policy = policy
        self.zero_slot = self.capacity if policy == 'static' else -1
        self.map = mx.full((self.n_experts,), self.zero_slot, dtype=mx.int32)
        self.hits = self.misses = 0
        self.warm = False

    def _install(self, e):
        if self.free:
            slot = self.free.pop()
        else:
            old_e, slot = self.slot_of.popitem(last=False)   # LRU victim
            self.map[old_e] = self.zero_slot
        for name in self.projs:
            rw, rs, rb = self.resident[name]
            if self.disk is not None:
                rw[slot] = self.disk.fetch(name, 'weight', e)
                rs[slot] = self.disk.fetch(name, 'scales', e)
                if rb is not None and self.disk.has(name, 'biases'):
                    rb[slot] = self.disk.fetch(name, 'biases', e)
            else:
                w, s, b = self.store[name]
                rw[slot] = w[e]
                rs[slot] = s[e]
                if rb is not None and b is not None:
                    rb[slot] = b[e]
        self.slot_of[e] = slot
        self.map[e] = slot
        # once every expert has a slot no eviction can occur, so residency is
        # permanently satisfied and the per-token check is pure overhead
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def preload(self, experts):
        """Fill the cache once for the static policy, most-used experts first."""
        for e in list(experts)[:self.capacity]:
            if int(e) not in self.slot_of:
                self._install(int(e))
        mx.eval([a for name in self.projs for a in self.resident[name]
                 if a is not None] + [self.map])

    def ensure(self, idx):
        """Make every expert in `idx` resident. Returns nothing; updates map.

        The `.tolist()` here is a device->host readback and therefore a sync.
        Measured cost on a 30-layer stack: 3.080 ms with no check, 3.757 ms with
        it — and a device-side reduction over the same indices is FREE (3.066),
        so it is the readback that costs, not the Python. In the real model the
        same 30 syncs cost 41% rather than 6.7%, because each one fragments the
        token's graph and the GPU idles across attention work that could have
        overlapped. Removing it entirely needs prefetch: run layer L+1's
        residency check during layer L's compute.
        """
        if self.policy == 'static':
            # never installs, never evicts, never reads back. Non-resident
            # experts already point at the zero slot, so the gather itself
            # yields a zero contribution — no mask, no clamp, no output
            # multiply, and above all NO device->host sync.
            return
        if self.warm:                    # nothing can miss; skip the sync
            return
        needed = set(int(e) for e in idx.reshape(-1).tolist())
        installed = False
        for e in needed:
            if e in self.slot_of:
                self.slot_of.move_to_end(e)
                self.hits += 1
            else:
                self.misses += 1
                self._install(e)
                installed = True
        # NO mx.eval here. It was added when the store was IN MEMORY, where a
        # k=8 run OOM'd Metal at ~25 GB — but that pressure came from holding
        # the full weights AND the resident copies, not from the scatter graph.
        # With the disk-backed store the installs are already-materialised
        # arrays, and evaluating every resident tensor on every miss costs 22.7%
        # while saving nothing: measured over a 1314-token generation, 52.0 vs
        # 63.8 tok/s at an identical 8.04 GB peak, and batch-8 x 1000 tokens
        # peaks at 14.20 GB either way. The prefill blow-up is prevented by the
        # per-chunk eval in __call__, which is a different thing and stays.
        del installed

    def qmm(self, name, x, slots, sorted_indices=False):
        """`sorted_indices` is NOT cosmetic — it selects a different kernel.

        gather_qmm's sorted path and general path disagree by ~1.3e-3 absolute
        (2.5e-3 relative) on identical inputs. The shipped SwitchGLU sorts and
        passes sorted_indices=True, so an offload path that skips it is not
        bit-comparable to the model it replaces, and a correctness check would
        report a mismatch that has nothing to do with the cache. Isolated by
        running the SAME store tensors through both paths: sorted gave 0.000e+00
        against SwitchGLU, unsorted gave 1.320e-03.
        """
        rw, rs, rb = self.resident[name]
        return mx.gather_qmm(x, rw, rs, rb, rhs_indices=slots, transpose=True,
                             group_size=self.group, bits=self.bits,
                             mode=self.mode, sorted_indices=sorted_indices)


class OffloadSwitchGLU(nn.Module):
    """SwitchGLU whose experts live in an ExpertCache."""

    def __init__(self, glu, capacity, disk=None, policy='exact', pin=None):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk, policy)
        self.activation = glu.activation
        if policy == 'static':
            self.cache.preload(pin if pin is not None else range(capacity))

    def _forward(self, x, indices):
        c = self.cache
        c.ensure(indices)
        slots = mx.take(c.map, indices)
        x = mx.expand_dims(x, (-2, -3))
        # Mirror SwitchGLU's rule exactly. Forcing the sort on every call was
        # measured to cost more than it saves: decode calls are 1 x top_k = 8
        # indices, far below the threshold, and the extra argsort/gather/scatter
        # ran 30 times per token.
        do_sort = indices.size >= 64
        inv = None
        if do_sort:
            x, slots, inv = _gather_sort(x, slots)
        up = c.qmm('up_proj', x, slots, do_sort)
        gate = c.qmm('gate_proj', x, slots, do_sort)
        out = c.qmm('down_proj', self.activation(up, gate), slots, do_sort)
        if do_sort:
            out = _scatter_unsort(out, inv, indices.shape)
        return out.squeeze(-2)

    def __call__(self, x, indices):
        """Chunk any call whose working set exceeds the cache.

        A single call must have every expert it routes to resident AT ONCE. A
        prefill routes the whole prompt in one call and can touch all 128
        experts, so with fewer slots the experts installed first are evicted
        before the gather runs and their slots read garbage. Decode never hits
        this (working set = batch x top_k), which is why the synthetic bench —
        batch-1 only, correctness checked at full capacity only — missed it
        entirely; the real model's prefill exposed it as a 100x NLL blow-up.

        Splitting the token axis bounds each chunk's working set. Chunking costs
        extra kernel launches, so it triggers only when actually needed.
        """
        c = self.cache
        if c.policy == 'static':
            return self._forward(x, indices)   # nothing to install, never chunks
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        # Fast path: distinct experts can never exceed n_tok * top_k, so if that
        # bound already fits the cache no chunking is possible and the decision
        # needs no device->host sync. This is the decode path (1 x 8 = 8), which
        # is every token of a generation; materialising the index set there cost
        # a sync per layer per token.
        if n_tok * k <= c.capacity or n_tok == 1:
            return self._forward(x, indices)
        distinct = len(set(int(e) for e in flat_i.reshape(-1).tolist()))
        if distinct <= c.capacity:
            return self._forward(x, indices)

        flat_x = x.reshape(-1, x.shape[-1])
        # halve until each chunk fits; capacity >= top_k guarantees termination
        size = n_tok
        while size > 1:
            size = max(1, size // 2)
            ok = True
            for s in range(0, n_tok, size):
                if len(set(int(e) for e in
                           flat_i[s:s + size].reshape(-1).tolist())) > c.capacity:
                    ok = False
                    break
            if ok:
                break
        # Evaluate each chunk before building the next. Left lazy, a batch-8
        # prefill of 1688 tokens holds every chunk's intermediates at once and
        # peaks at 34.5 GB — worse than not offloading at all (16.4 GB), which
        # is what OOM'd the eval harness. The whole point is to use LESS memory.
        outs = []
        for s in range(0, n_tok, size):
            o = self._forward(flat_x[s:s + size], flat_i[s:s + size])
            mx.eval(o)
            outs.append(o)
        out = mx.concatenate(outs, axis=0)
        return out.reshape(indices.shape + (x.shape[-1],))

