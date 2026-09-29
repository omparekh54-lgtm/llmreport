"""Work out a model's architecture by inspecting its modules, without relying on a config file.

Every value comes from the module tree itself (layer types, weight shapes, attributes),
so it works for Hugging Face models and for models written from scratch. When a
Hugging Face config is present, the architecture check prefers it and uses these
findings to fill the gaps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

_ATTN_EXACT = {
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "qkv_proj", "query", "key", "value", "wq", "wk", "wv", "wo",
    "c_attn", "query_key_value", "in_proj", "wqkv", "q", "k", "v", "o", "to_q", "to_k", "to_v", "to_out",
}
_MLP_EXACT = {
    "fc1", "fc2", "fc_in", "fc_out", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "c_fc",
    "dense_h_to_4h", "dense_4h_to_h", "gate", "up", "down", "wi", "wi_0", "wi_1",
}
_RECURRENT = (nn.RNNBase,)

_ACTIVATIONS = {
    "gelu": "GELU", "silu": "SiLU", "swish": "SiLU", "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid",
    "mish": "Mish", "elu": "ELU", "softplus": "Softplus", "quickgelu": "QuickGELU",
}


@dataclass
class Linearish:
    name: str
    module: nn.Module
    in_features: int
    out_features: int
    has_bias: bool


def linear_shape(module: nn.Module) -> Optional[Tuple[int, int, bool]]:
    """(in_features, out_features, has_bias) for Linear, HF Conv1D and quantized linear layers."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features, module.bias is not None
    cls = type(module).__name__
    if cls == "Conv1D" and hasattr(module, "weight") and torch.is_tensor(module.weight) and module.weight.dim() == 2:
        return module.weight.shape[0], module.weight.shape[1], getattr(module, "bias", None) is not None
    if isinstance(getattr(module, "in_features", None), int) and isinstance(getattr(module, "out_features", None), int):
        bias = getattr(module, "bias", None)
        if callable(bias):
            try:
                bias = bias()
            except Exception:
                bias = None
        return module.in_features, module.out_features, bias is not None
    return None


def is_norm(module: nn.Module) -> bool:
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        return True
    rms = getattr(nn, "RMSNorm", None)
    if rms is not None and isinstance(module, rms):
        return True
    name = type(module).__name__.lower()
    return "norm" in name and not list(module.children())


def norm_label(module: nn.Module) -> str:
    cls = type(module).__name__
    rms = getattr(nn, "RMSNorm", None)
    if (rms is not None and isinstance(module, rms)) or "rms" in cls.lower() or cls == "T5LayerNorm":
        return "RMSNorm"
    if isinstance(module, nn.LayerNorm) or "layernorm" in cls.lower():
        bias = getattr(module, "bias", None)
        return "LayerNorm" if bias is not None else "LayerNorm (no bias)"
    return cls


def activation_label(module: nn.Module) -> Optional[str]:
    cls = type(module).__name__
    low = cls.lower().replace("activation", "").replace("_", "")
    for key, label in _ACTIVATIONS.items():
        if key in low:
            if label == "GELU":
                approx = getattr(module, "approximate", "none")
                if approx == "tanh" or any(k in cls for k in ("New", "Tanh", "Fast", "Pytorch")):
                    return "GELU (tanh approximation)"
            return label
    return None


def path_category(path: str) -> Optional[str]:
    """'attention', 'mlp' or None from a dotted module path."""
    parts = [p.lower() for p in path.split(".") if p]
    if any("attn" in p or "attention" in p for p in parts):
        return "attention"
    if any("mixer" in p or "mamba" in p or "ssm" in p for p in parts):
        return "sequence_mixer"
    if any(p in _ATTN_EXACT for p in parts[-1:]):
        return "attention"
    if any(k in p for p in parts for k in ("mlp", "ffn", "feed_forward", "feedforward", "experts", "intermediate")) \
            or any(p in _MLP_EXACT for p in parts[-1:]):
        return "mlp"
    return None


def find_position_embedding(model: nn.Module) -> Optional[nn.Embedding]:
    """A learned absolute position embedding, if the model has one."""
    embeddings = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Embedding)]
    if len(embeddings) < 2:
        return None
    token = max(embeddings, key=lambda nm: nm[1].num_embeddings)[1]
    for name, m in embeddings:
        low = name.lower()
        if m is not token and m.embedding_dim == token.embedding_dim and any(
            k in low.split(".")[-1] for k in ("pos", "wpe", "position")
        ):
            return m
    return None


def repeated_blocks(model: nn.Module) -> Tuple[Optional[str], List[nn.Module]]:
    """The container of identical layers holding the most parameters (the transformer blocks).

    Returns the dotted path of the first block and the list of blocks.
    """
    best, best_name, best_params = [], None, 0
    for name, m in model.named_modules():
        if not isinstance(m, (nn.ModuleList, nn.Sequential)):
            continue
        named = list(m.named_children())
        children = [c for _, c in named]
        if not children or len({type(c) for c in children}) != 1:
            continue
        if not list(children[0].children()):  # a list of plain Linear layers is not a stack of blocks
            continue
        if any(isinstance(c, _RECURRENT) for c in children):
            continue
        params = sum(p.numel() for c in children for p in c.parameters())
        params += sum(_packed_numel(mm) for c in children for mm in c.modules())
        if params > best_params:
            first = f"{name}.{named[0][0]}" if name else named[0][0]
            best, best_name, best_params = children, first, params
    return best_name, best


def _packed_numel(module: nn.Module) -> int:
    """Weights of dynamically quantized layers, which don't show up in parameters()."""
    if not hasattr(module, "_packed_params"):
        return 0
    try:
        weight = module.weight() if callable(getattr(module, "weight", None)) else None
        bias = module.bias() if callable(getattr(module, "bias", None)) else None
    except Exception:
        return 0
    return (weight.numel() if weight is not None else 0) + (bias.numel() if bias is not None else 0)


def _int_attr(module: nn.Module, names) -> Optional[int]:
    for n in names:
        value = getattr(module, n, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


_SSM_WORDS = ("mamba", "rwkv", "hyena", "retnet", "s4", "ssm", "xlstm")


def is_state_space(model: nn.Module) -> bool:
    return any(k in type(m).__name__.lower() for m in model.modules() for k in _SSM_WORDS)


def inspect_structure(model: nn.Module, vocab_hint: Optional[int] = None, heads_hint: Optional[int] = None,
                      head_dim_hint: Optional[int] = None) -> Dict[str, Any]:
    """Describe the architecture of ``model`` from its modules. Missing values are left out.

    ``heads_hint`` and ``head_dim_hint`` (from a config) are used when the attention modules
    don't store their head count, so the number of key/value heads can still be measured.
    """
    info: Dict[str, Any] = {}
    modules = list(model.named_modules())
    ssm = is_state_space(model)
    lin = [(n, linear_shape(m), m) for n, m in modules]
    lin = [Linearish(n, m, s[0], s[1], s[2]) for n, s, m in lin if s is not None]
    embeddings = [(n, m) for n, m in modules if isinstance(m, nn.Embedding)]
    recurrent = [(n, m) for n, m in modules if isinstance(m, _RECURRENT)]

    # ---- vocabulary, token embedding and output head
    token_emb = max(embeddings, key=lambda nm: nm[1].num_embeddings)[1] if embeddings else None
    vocab = vocab_hint or (token_emb.num_embeddings if token_emb is not None else None)
    head = None
    if vocab:
        candidates = [x for x in lin if x.out_features == vocab]
        head = candidates[-1] if candidates else None
    if head is None and lin:
        widest = max(lin, key=lambda x: x.out_features)
        if token_emb is not None and widest.out_features == token_emb.num_embeddings:
            head = widest
    if vocab is None and head is not None:
        vocab = head.out_features
    if vocab:
        info["vocab_size"] = vocab
    if token_emb is not None:
        info["hidden_size"] = token_emb.embedding_dim
    if head is not None:
        info["output_head"] = head.name
        w = getattr(head.module, "weight", None)
        info["tied_embeddings"] = bool(
            token_emb is not None and torch.is_tensor(w) and w is token_emb.weight
            or (token_emb is not None and torch.is_tensor(w) and w.shape == token_emb.weight.shape
                and w.device.type != "meta" and w.data_ptr() == token_emb.weight.data_ptr())
        )

    # ---- positions
    pos = find_position_embedding(model)
    names_low = " ".join(n.lower() for n, _ in modules) + " " + " ".join(type(m).__name__.lower() for _, m in modules)
    buffers = " ".join(n.lower() for n, _ in model.named_buffers())
    if pos is not None:
        info["position_encoding"] = f"learned absolute ({pos.num_embeddings:,} positions)"
    elif any(k in names_low or k in buffers for k in ("rotary", "rope", "inv_freq", "freqs_cis", "cos_cached")):
        info["position_encoding"] = "rotary (RoPE)"
    elif "alibi" in names_low or "alibi" in buffers:
        info["position_encoding"] = "ALiBi"
    elif recurrent:
        info["position_encoding"] = "none needed (recurrent)"
    elif any(n.split(".")[-1] in ("pe", "pos_encoding", "positional_encoding") for n, _ in model.named_buffers()):
        info["position_encoding"] = "sinusoidal (fixed)"
    else:
        info["position_encoding"] = "none detected"

    # ---- repeated blocks
    first_block, blocks = repeated_blocks(model)
    blocks_name = first_block.rsplit(".", 1)[0] if first_block and "." in first_block else ""
    if blocks:
        info["layers"] = len(blocks)
        info["block_class"] = type(blocks[0]).__name__
        info["blocks_path"] = blocks_name or "(top level)"
    block = blocks[0] if blocks else model
    block_mods = list(block.named_modules())

    # ---- attention heads
    heads = kv_heads = head_dim = None
    for _, m in block_mods:
        if isinstance(m, nn.MultiheadAttention):
            heads, kv_heads, head_dim = m.num_heads, m.num_heads, m.head_dim
            break
        heads = heads or _int_attr(m, ("num_heads", "n_head", "n_heads", "num_attention_heads", "nhead", "n_heads_q"))
        kv_heads = kv_heads or _int_attr(m, ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "n_kv_head",
                                             "n_head_kv", "num_kv"))
        head_dim = head_dim or _int_attr(m, ("head_dim", "head_size", "d_head", "attention_head_size"))
    heads_found_in_modules = heads is not None
    heads = heads or heads_hint
    head_dim = head_dim or head_dim_hint
    hidden = info.get("hidden_size")
    block_lin = [x for x in lin if x.name.startswith(f"{first_block}.")] if blocks else lin
    attn_lin = [x for x in block_lin if path_category(x.name) == "attention"]
    mlp_lin = [x for x in block_lin if path_category(x.name) != "attention"]
    if ssm:
        attn_lin, heads = [], None  # names like in_proj/out_proj mean something else in state-space models
        mlp_lin = []
    if heads and (attn_lin or heads_found_in_modules):
        info["attention_heads"] = heads
        head_dim = head_dim or (hidden // heads if hidden and hidden % heads == 0 else None)
        if kv_heads is None and head_dim:
            q = next((x for x in attn_lin if x.name.split(".")[-1] in ("q_proj", "query", "wq", "q", "to_q")), None)
            k = next((x for x in attn_lin if x.name.split(".")[-1] in ("k_proj", "key", "wk", "k", "to_k")), None)
            fused = next((x for x in attn_lin if x.name.split(".")[-1] in
                          ("qkv", "qkv_proj", "c_attn", "query_key_value", "in_proj", "wqkv")), None)
            if k is not None:
                kv_heads = k.out_features // head_dim
            elif fused is not None and hidden and fused.out_features > hidden:
                kv_heads = (fused.out_features - heads * head_dim) // (2 * head_dim)
            elif q is not None or fused is not None:
                kv_heads = heads
        if head_dim:
            info["head_dim"] = head_dim
        if kv_heads:
            info["kv_heads"] = kv_heads
            info["attention_type"] = (
                "multi-head (MHA)" if kv_heads == heads else
                "multi-query (MQA)" if kv_heads == 1 else f"grouped-query (GQA, {heads // kv_heads} query heads per KV head)"
            )
    elif attn_lin or any(isinstance(m, nn.MultiheadAttention) for _, m in block_mods):
        info["attention_type"] = "attention (head count not detectable)"

    # ---- feed-forward
    for n, m in block_mods:
        if n.split(".")[-1] != "experts" and "experts" not in type(m).__name__.lower():
            continue
        if isinstance(m, (nn.ModuleList, nn.ModuleDict)):
            info["experts"] = len(m)
        else:  # fused experts: one 3-D weight of shape (experts, out, in)
            stacked = [p for p in m.parameters(recurse=False) if p.dim() == 3]
            if stacked:
                info["experts"] = stacked[0].shape[0]
        if info.get("experts"):
            info["mlp_type"] = f"mixture of experts ({info['experts']} experts)"
            break
    if hidden:
        ups = [x for x in mlp_lin if x.in_features == hidden and x.out_features != hidden and x is not head]
        downs = [x for x in mlp_lin if x.out_features == hidden and x.in_features != hidden]
        if downs and "mlp_type" not in info:
            inter = downs[0].in_features
            info["intermediate_size"] = inter
            gated = sum(x.out_features == inter for x in ups) >= 2 or any(x.out_features == 2 * inter for x in ups)
            info["mlp_type"] = "gated (GLU-style)" if gated else "standard (up, activation, down)"
        elif ups:
            info["intermediate_size"] = max(x.out_features for x in ups)

    # ---- activations and norms
    acts = Counter(a for a in (activation_label(m) for _, m in block_mods) if a)
    if not acts:  # activations stored as plain functions, e.g. nn.TransformerEncoderLayer(activation="gelu")
        for _, m in block_mods:
            for attr in ("activation", "act", "act_fn", "activation_fn", "nonlinearity"):
                fn = getattr(m, attr, None)
                if callable(fn) and not isinstance(fn, nn.Module):
                    name = getattr(fn, "__name__", "").lower()
                    label = next((v for k, v in _ACTIVATIONS.items() if k in name), None)
                    if label:
                        acts[label] += 1
    if acts:
        act = acts.most_common(1)[0][0]
        if info.get("mlp_type", "").startswith("gated"):
            if act == "SiLU":
                act = "SwiGLU (gated SiLU)"
            elif act.startswith("GELU"):
                act = "GeGLU (gated " + act + ")"
            else:
                act = f"gated {act}"
        info["activation"] = act
    norms = Counter(norm_label(m) for _, m in block_mods if is_norm(m))
    if norms:
        info["norm_type"] = norms.most_common(1)[0][0]
        info["norms_per_block"] = sum(norms.values()) if blocks else None
    block_ids = {id(m) for b in blocks for m in b.modules()}
    final_norms = [n for n, m in modules if is_norm(m) and id(m) not in block_ids]
    if blocks:
        info["final_norm"] = bool(final_norms)
    if block_lin:
        info["linear_bias"] = any(x.has_bias for x in block_lin)

    # ---- family
    cfg = getattr(model, "config", None)
    if recurrent:
        kinds = sorted({type(m).__name__ for _, m in recurrent})
        info["family"] = f"recurrent ({', '.join(kinds)})"
    elif ssm:
        info["family"] = "state-space / linear-recurrent"
    elif getattr(cfg, "is_encoder_decoder", False):
        info["family"] = "encoder-decoder transformer"
    elif "attention_type" in info:
        info["family"] = "decoder-only transformer"
    elif blocks:
        info["family"] = "stack of repeated blocks"
    else:
        info["family"] = "unknown"
    if info.get("experts"):
        info["family"] += f" with mixture of experts ({info['experts']} experts per layer)"
    if recurrent:
        r = recurrent[0][1]
        info["hidden_size"] = getattr(r, "hidden_size", info.get("hidden_size"))
        info["layers"] = sum(getattr(m, "num_layers", 1) for _, m in recurrent)
    return {k: v for k, v in info.items() if v is not None}


def norm_placement(lm, blocks: List[nn.Module]) -> Optional[str]:
    """'pre-norm' or 'post-norm', found by watching which layer inside the first block runs first."""
    if not blocks:
        return None
    block = blocks[0]
    try:
        if any(p.device.type == "meta" for p in block.parameters()):
            return None
    except Exception:
        return None
    order: List[str] = []
    hooks = []
    for name, m in block.named_modules():
        if m is block:
            continue
        # Norms, and anything else that owns weights (linear layers, nn.MultiheadAttention, ...).
        computes = linear_shape(m) is not None or any(True for _ in m.parameters(recurse=False))
        if is_norm(m) or computes:
            kind = "norm" if is_norm(m) else "compute"
            hooks.append(m.register_forward_pre_hook(lambda _m, _i, kind=kind: order.append(kind)))
    try:
        vocab = lm.tok.vocab_size if lm.tok is not None else None
        ids = torch.arange(1, 5, device=lm.device).remainder(max(2, min(vocab or 50, 50) - 1)).view(1, -1)
        with torch.no_grad():
            lm.logits(ids)
    except Exception:
        return None
    finally:
        for h in hooks:
            h.remove()
    if not order:
        return None
    return "pre-norm" if order[0] == "norm" else ("post-norm" if "norm" in order else "no norm in blocks")


def param_breakdown(model: nn.Module, vocab: Optional[int]) -> Tuple[Dict[str, int], int, Dict[str, int]]:
    """Parameters per component, total count, and count per dtype (tied weights counted once)."""
    seen = set()
    breakdown: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    total = 0

    def add(category, n, dtype):
        nonlocal total
        breakdown[category] = breakdown.get(category, 0) + n
        dtypes[dtype] = dtypes.get(dtype, 0) + n
        total += n

    for path, m in model.named_modules():
        own = list(m.named_parameters(recurse=False))
        packed = _packed_numel(m)
        if not own and not packed:
            continue
        if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
            category = "embeddings"
        elif is_norm(m):
            category = "norm"
        elif isinstance(m, _RECURRENT):
            category = "recurrent"
        else:
            shape = linear_shape(m)
            if shape and vocab and shape[1] == vocab and path_category(path) is None:
                category = "output_head"
            else:
                category = path_category(path) or "other"
        for _, p in own:
            if id(p) in seen:
                continue
            seen.add(id(p))
            add(category, p.numel(), str(p.dtype).replace("torch.", ""))
        if packed:
            add(category, packed, "qint8")
    return {k: v for k, v in breakdown.items() if v}, total, dtypes
