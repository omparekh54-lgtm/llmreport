"""What the model *is*: size, shape, memory footprint and tokenizer facts."""

from __future__ import annotations

from collections import OrderedDict

import torch

from .._utils import human_bytes, human_number, max_context
from .base import INFO, OK, Analyzer, RunConfig

# Where each parameter goes, decided by keywords in its name.
_COMPONENTS = OrderedDict(
    [
        ("embeddings", ("embed", "wte", "wpe", "word_embeddings", "position_embeddings")),
        # Norms are matched before attention/MLP because names such as
        # "post_attention_layernorm" contain the word "attention".
        ("norm", ("norm", "ln_", "layernorm", "layer_norm")),
        ("attention", ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")),
        ("mlp", ("mlp", "ffn", "feed_forward", "fc1", "fc2", "gate_proj", "up_proj", "down_proj", "experts")),
        ("output_head", ("lm_head", "output", "classifier")),
    ]
)

_CONFIG_FIELDS = OrderedDict(
    [
        ("model_type", ("model_type",)),
        ("layers", ("num_hidden_layers", "n_layer", "num_layers")),
        ("hidden_size", ("hidden_size", "n_embd", "d_model")),
        ("attention_heads", ("num_attention_heads", "n_head")),
        ("kv_heads", ("num_key_value_heads",)),
        ("intermediate_size", ("intermediate_size", "n_inner", "ffn_dim")),
        ("vocab_size", ("vocab_size",)),
        ("activation", ("hidden_act", "activation_function")),
        ("rope_theta", ("rope_theta",)),
        ("tie_word_embeddings", ("tie_word_embeddings",)),
    ]
)

_SAMPLE = "The quick brown fox jumps over the lazy dog, then naps under a tree."


def classify_param(name: str) -> str:
    lowered = name.lower()
    for component, keys in _COMPONENTS.items():
        if any(k in lowered for k in keys):
            return component
    return "other"


class ArchitectureAnalyzer(Analyzer):
    name = "architecture"
    title = "Architecture"
    description = "Parameter counts, layer shapes, memory footprint and tokenizer facts."

    def run(self, model, tokenizer, config: RunConfig):
        # -- parameters (named_parameters de-duplicates tied weights)
        total = trainable = 0
        breakdown = {k: 0 for k in list(_COMPONENTS) + ["other"]}
        dtypes = {}
        for name, p in model.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            breakdown[classify_param(name)] += n
            key = str(p.dtype).replace("torch.", "")
            dtypes[key] = dtypes.get(key, 0) + n
        breakdown = {k: v for k, v in breakdown.items() if v}

        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        # -- config
        cfg = getattr(model, "config", None)
        shape = {}
        for label, attrs in _CONFIG_FIELDS.items():
            for attr in attrs:
                value = getattr(cfg, attr, None)
                if value is not None and not callable(value):
                    shape[label] = value
                    break
        context = max_context(model, tokenizer)
        shape["max_context"] = context

        device = str(next(model.parameters()).device) if total else "cpu"
        main_dtype = max(dtypes, key=dtypes.get) if dtypes else "unknown"

        # -- KV cache per token (keys + values, every layer), in the model's dtype
        kv_per_token = None
        layers, hidden, heads = shape.get("layers"), shape.get("hidden_size"), shape.get("attention_heads")
        if isinstance(layers, int) and isinstance(hidden, int) and isinstance(heads, int) and heads:
            kv_heads = shape.get("kv_heads") or heads
            head_dim = getattr(cfg, "head_dim", None) or hidden // heads
            elem = torch.tensor([], dtype=next(model.parameters()).dtype).element_size()
            kv_per_token = 2 * layers * kv_heads * head_dim * elem

        # -- tokenizer
        tok_info = {}
        if tokenizer is not None:
            ids = tokenizer(_SAMPLE, add_special_tokens=False)["input_ids"]
            specials = {}
            for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
                value = getattr(tokenizer, attr, None)
                if value:
                    specials[attr] = str(value)
            tok_info = {
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "special_tokens": specials,
                "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
                "sample_text": _SAMPLE,
                "sample_tokens": len(ids),
                "chars_per_token": round(len(_SAMPLE) / max(1, len(ids)), 2),
            }

        metrics = {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 2) if total else 0.0,
            "layers": shape.get("layers"),
            "hidden_size": shape.get("hidden_size"),
            "attention_heads": shape.get("attention_heads"),
            "vocab_size": shape.get("vocab_size"),
            "max_context": context,
            "dtype": main_dtype,
            "device": device,
            "weights_bytes": weight_bytes,
            "est_fp32_bytes": total * 4,
            "est_fp16_bytes": total * 2,
            "est_int8_bytes": total,
            "est_int4_bytes": total // 2,
        }
        if kv_per_token:
            metrics["kv_cache_bytes_per_token"] = kv_per_token
            metrics["kv_cache_bytes_full_context"] = kv_per_token * context
        if tok_info:
            metrics["chars_per_token"] = tok_info["chars_per_token"]
        metrics = {k: v for k, v in metrics.items() if v is not None}

        arch = getattr(cfg, "architectures", None) or [type(model).__name__]
        summary = (
            f"{arch[0]} with {human_number(total)} parameters"
            + (f" across {shape['layers']} layers" if "layers" in shape else "")
            + f", {main_dtype} weights using {human_bytes(weight_bytes)}"
            + f", context window of {context:,} tokens."
        )
        notes = []
        if trainable != total:
            notes.append(f"{metrics['trainable_pct']}% of parameters are trainable (frozen layers or adapters).")
        if tok_info and not tok_info["has_chat_template"]:
            notes.append("Tokenizer has no chat template, so behavior checks use raw prompts (base-model style).")

        return self.result(
            summary,
            metrics=metrics,
            details={
                "architecture_class": arch[0],
                "config": shape,
                "param_breakdown": breakdown,
                "dtypes": dtypes,
                "buffers_bytes": buffer_bytes,
                "tokenizer": tok_info,
            },
            status=OK if total else INFO,
            notes=notes,
        )
