"""Locate MoE expert modules by class, without knowing the family.

VENDORED — copied in rather than imported, so this repository is
self-contained.
Origin: the author's earlier MoE tooling
Same authorship and licence as the rest of this repo (Apache-2.0); see NOTICE.

Do not edit in place — this is a snapshot of the code that produced the numbers
in FINDINGS.md, and it should not drift away from them.
"""
from mlx_lm.models.switch_layers import SwitchGLU

# Attribute chains to the decoder-layer list, most specific first. VLM-wrapped
# builds bury it one level deeper, which is the same nesting that put the real
# dims under args.text_config.
_LAYER_PATHS = (
    ('language_model', 'model', 'layers'),   # qwen3.6 (VLM-wrapped)
    ('model', 'layers'),                     # gemma-4, most text builds
    ('language_model', 'layers'),
    ('layers',),
)


def find_layers(model):
    """The decoder-layer list, wherever this family keeps it."""
    for path in _LAYER_PATHS:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, (list, tuple)) and obj:
            return obj
    raise SystemExit(
        "could not locate the decoder layer list; tried "
        + ", ".join('.'.join(p) for p in _LAYER_PATHS))


def _is_moe(obj):
    """A layer's expert module, wrapped or not.

    After wrap() the SwitchGLU has been REPLACED by an OffloadSwitchGLU, which
    is an nn.Module but not a SwitchGLU. Matching only the former made
    find_moe() return [] on an already-wrapped model — which is precisely when
    callers ask for it, to read hit counters back out. Matched by name rather
    than by import so this module stays free of a cycle through offload.py.
    """
    return (isinstance(obj, SwitchGLU)
            or type(obj).__name__ == 'OffloadSwitchGLU')


def find_moe(model):
    """[(layer_index, owner_module, attr_name, expert_module)], in layer order.

    `expert_module` is a SwitchGLU before wrapping and an OffloadSwitchGLU
    after — callers that need the quantized projections should use it before
    wrapping, callers that need `.cache` after.

    `owner`/`attr` are returned rather than just the SwitchGLU so callers can
    SUBSTITUTE it (`setattr(owner, attr, OffloadSwitchGLU(...))`) — dropping the
    original module is what lets MLX free the expert weights, and is the whole
    difference between measuring the mechanism and saving memory.

    Only direct children of a decoder layer are searched, one level down. That
    is deep enough for both spellings above and shallow enough that it cannot
    wander into a vision tower.
    """
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


def _candidates(layer):
    """The decoder layer itself, then its direct child modules."""
    yield layer
    for _name, child in _children(layer):
        yield child


def _children(mod):
    ch = getattr(mod, 'children', None)
    if not callable(ch):
        return []
    out = []
    for name, child in ch().items():
        for c in (child if isinstance(child, (list, tuple)) else [child]):
            out.append((name, c))
    return out


def describe(model):
    """(n_moe_layers, n_experts, top_k_or_None) for logging and sanity checks."""
    moe = find_moe(model)
    if not moe:
        return 0, 0, None
    _li, owner, _name, glu = moe[0]
    n_experts = glu.gate_proj['weight'].shape[0]
    top_k = getattr(owner, 'top_k', None) or getattr(owner, 'num_experts_per_tok',
                                                     None)
    return len(moe), n_experts, top_k
