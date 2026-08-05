"""Locate a model's expert modules AND its router, without knowing the family.

Ported rather than imported from the vendored `_vendor/moe.py`: this module
adds `find_gates`, which the vendored version does not have. The mlx import is deferred into the functions so the analysis
half of this package stays importable with no mlx installed.

`find_moe` resolves the expert modules by CLASS (`SwitchGLU`), because the
attribute names are family-specific and the class is not:

    gemma-4-26b-a4b   model.model.layers[i].experts.switch_glu
    qwen3.6-35b-a3b   model.language_model.model.layers[i].mlp.switch_mlp

`find_gates` is new here, and it is the piece R2 needs. The vendored
`gatetrace.py` found the seam for qwen — the gate lives on the block that OWNS
the SwitchGLU, as `owner.gate`, not on the SwitchGLU itself — but it hard
requires that spelling and raises "add this family to gatetrace" otherwise.
That is exactly why there is a qwen gate trace and no gemma one.

gemma keeps its router one level up, on the decoder layer as `layer.router`,
so the search here covers the owner first and then the layer. The two families
also RETURN different things, which callers must handle rather than assume:

    qwen    gate(x) -> raw scores [..., n_experts]     full ranking available
    gemma   router(x) -> (indices, weights)            top-k only, pre-selected

`gate_output` normalises both to (ranks, scores) so a capture can treat them
uniformly, while recording which shape it actually saw — the gemma capture is
genuinely poorer (it cannot see experts the router considered and rejected) and
that difference must survive into the analysis rather than be smoothed over.
"""

# Attribute chains to the decoder-layer list, most specific first. VLM-wrapped
# builds bury it one level deeper.
_LAYER_PATHS = (
    ('language_model', 'model', 'layers'),   # qwen3.6 (VLM-wrapped)
    ('model', 'layers'),                     # gemma-4, most text builds
    ('language_model', 'layers'),
    ('layers',),
)

# Router attribute names, in search order. `gate_proj` is deliberately absent:
# it is a projection INSIDE the expert stack, not the routing decision, and
# matching it would silently capture the wrong tensor.
_GATE_NAMES = ('gate', 'router', 'gating', 'gate_network')


def find_layers(model):
    for path in _LAYER_PATHS:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, (list, tuple)) and obj:
            return obj
    raise SystemExit("could not locate the decoder layer list; tried "
                     + ", ".join('.'.join(p) for p in _LAYER_PATHS))


def _is_moe(obj):
    from mlx_lm.models.switch_layers import SwitchGLU
    return (isinstance(obj, SwitchGLU)
            or type(obj).__name__ == 'OffloadSwitchGLU')


def _children(mod):
    ch = getattr(mod, 'children', None)
    if not callable(ch):
        return []
    out = []
    for name, child in ch().items():
        for c in (child if isinstance(child, (list, tuple)) else [child]):
            out.append((name, c))
    return out


def _candidates(layer):
    yield layer
    for _name, child in _children(layer):
        yield child


def find_moe(model):
    """[(layer_index, owner_module, attr_name, expert_module)], layer order."""
    found = []
    for li, layer in enumerate(find_layers(model)):
        for owner in _candidates(layer):
            for name, child in _children(owner):
                if _is_moe(child):
                    found.append((li, owner, name, child))
                    break
            else:
                continue
            break
    return found


def find_gates(model):
    """[(layer_index, holder_module, attr_name, gate_module)], layer order.

    Searched on the SwitchGLU's owner first, then on the decoder layer, so both
    spellings resolve. Raises if a MoE layer has no locatable router — silently
    skipping one would leave a hole in the capture that looks like missing data
    later.
    """
    layers = find_layers(model)
    out = []
    for li, owner, _attr, _glu in find_moe(model):
        for holder in (owner, layers[li]):
            for name in _GATE_NAMES:
                g = getattr(holder, name, None)
                if g is not None and callable(g):
                    out.append((li, holder, name, g))
                    break
            else:
                continue
            break
        else:
            raise SystemExit(
                f"layer {li}: no router found on {type(owner).__name__} or "
                f"{type(layers[li]).__name__}; tried {_GATE_NAMES}. Add this "
                f"family's spelling to _GATE_NAMES.")
    return out


def gate_output(out, top_m):
    """Normalise a router's return to (ranks, scores, shape_kind).

    ranks/scores are mx arrays of shape [..., m], m = min(top_m, n_experts) for
    'scores' routers and the router's own top-k for 'topk' routers.
    """
    import mlx.core as mx
    if isinstance(out, (tuple, list)):
        # (indices, weights) — already selected; nothing below top-k is visible
        idx, w = out[0], out[1]
        return idx, w, 'topk'
    order = mx.argsort(-out, axis=-1)[..., :top_m]
    return order, mx.take_along_axis(out, order, axis=-1), 'scores'


def describe(model):
    moe = find_moe(model)
    if not moe:
        return 0, 0, None
    _li, owner, _name, glu = moe[0]
    n_experts = glu.gate_proj['weight'].shape[0]
    top_k = (getattr(owner, 'top_k', None)
             or getattr(owner, 'num_experts_per_tok', None))
    return len(moe), n_experts, top_k


def _selftest():
    """Resolve both family spellings on stub trees. No checkpoint is loaded.

    Builds tiny real `SwitchGLU`s (a few KB) inside fake decoder layers, so the
    isinstance resolution is exercised against the actual class rather than a
    mock that would pass regardless.
    """
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    class QwenMLP(nn.Module):          # gate on the OWNER of the SwitchGLU
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4)
            self.switch_mlp = SwitchGLU(8, 16, 4)

    class QwenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = QwenMLP()

    class GemmaExperts(nn.Module):     # no gate here — it lives one level up
        def __init__(self):
            super().__init__()
            self.switch_glu = SwitchGLU(8, 16, 4)

    class GemmaLayer(nn.Module):       # router on the DECODER LAYER
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(8, 4)
            self.experts = GemmaExperts()

    class Inner(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

    class Model(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = Inner(layers)

    ok = True
    for name, layer_cls, want_holder, want_attr in (
            ('qwen-like', QwenLayer, 'QwenMLP', 'gate'),
            ('gemma-like', GemmaLayer, 'GemmaLayer', 'router')):
        m = Model([layer_cls() for _ in range(3)])
        moe, gates = find_moe(m), find_gates(m)
        got_holder = type(gates[0][1]).__name__
        good = (len(moe) == 3 and len(gates) == 3
                and got_holder == want_holder and gates[0][2] == want_attr)
        ok &= good
        print(f"  {name:12s} moe {len(moe)} gates {len(gates)}  "
              f"holder {got_holder}.{gates[0][2]}  "
              f"{'OK' if good else 'FAIL (want %s.%s)' % (want_holder, want_attr)}")

    # both return shapes normalise
    import mlx.core as mx
    scores = mx.array([[[0.1, 0.9, 0.3, 0.5]]])
    r, s, kind = gate_output(scores, 2)
    good = kind == 'scores' and r.tolist() == [[[1, 3]]]
    ok &= good
    print(f"  {'scores router':12s} ranks {r.tolist()} kind {kind}  "
          f"{'OK' if good else 'FAIL'}")
    r, s, kind = gate_output((mx.array([[[1, 3]]]), mx.array([[[0.9, 0.5]]])), 2)
    good = kind == 'topk' and r.tolist() == [[[1, 3]]]
    ok &= good
    print(f"  {'topk router':12s} ranks {r.tolist()} kind {kind}  "
          f"{'OK' if good else 'FAIL'}")
    print("selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
