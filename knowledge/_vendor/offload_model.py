"""Swap each MoE layer's SwitchGLU for a cached one.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier MoE offload runtime —
        wrap() only; the benchmark CLI and its imports are not vendored
Same authorship and licence as the rest of this repo (Apache-2.0); see NOTICE.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""

from .offload import OffloadSwitchGLU

def wrap(model, capacity, store_dir=None, policy='exact', pins=None,
         per_layer=None):
    """Swap each layer's SwitchGLU for a cached one.

    With store_dir the old module is dropped entirely, so MLX can free the
    12.85 GB of expert weights; without it the weights stay referenced as the
    store and wrapping costs memory instead of saving it.

    per_layer: {layer_index: capacity}, overriding the uniform `capacity`. The
    uniform split is only optimal if every layer has the same miss curve, and
    they do not — a greedy allocation gives one layer 104
    slots and another 39 for the same total, cutting misses 19% at identical
    memory.
    """
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n = 0
    for li, layer in enumerate(layers):
        ex = getattr(layer, 'experts', None)
        if ex is not None and hasattr(ex, 'switch_glu'):
            disk = None
            if store_dir is not None:
                from .store_backend import DiskStore
                disk = DiskStore(store_dir, li)
            cap = (per_layer or {}).get(li, capacity)
            ex.switch_glu = OffloadSwitchGLU(ex.switch_glu, cap, disk,
                                             policy, (pins or {}).get(li))
            n += 1
    return n
