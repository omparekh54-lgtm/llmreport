"""What the model *is*: structure, size, memory footprint and tokenizer facts.

Works for any PyTorch model. The structure (layers, heads, hidden size, position
encoding, norms, activation, weight tying...) is read from the module tree itself,
so a from-scratch GPT is described as accurately as a Hugging Face model.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from .._utils import human_bytes, human_number
from ..adapters import as_model, as_tokenizer
from ..structure import (
    find_position_embedding,
    inspect_structure,
    norm_placement,
    param_breakdown,
    path_category,
    repeated_blocks,
)
from .base import INFO, OK, WARNING, Analyzer, RunConfig

_COMPONENTS = ("embeddings", "attention", "mlp", "sequence_mixer", "norm", "recurrent", "output_head", "other")

# Hugging Face config names for each field. Custom configs (dataclasses, dicts) often use the same names.
_CONFIG_FIELDS = OrderedDict(
    [
        ("model_type", ("model_type",)),
        ("layers", ("num_hidden_layers", "n_layer", "num_layers", "n_layers")),
        ("hidden_size", ("hidden_size", "n_embd", "d_model", "dim", "embed_dim")),
        ("attention_heads", ("num_attention_heads", "n_head", "n_heads", "num_heads")),
        ("kv_heads", ("num_key_value_heads", "n_kv_heads", "num_kv_heads")),
        ("head_dim", ("head_dim",)),
        ("intermediate_size", ("intermediate_size", "n_inner", "ffn_dim", "hidden_dim")),
        ("vocab_size", ("vocab_size", "n_vocab")),
        ("activation", ("hidden_act", "activation_function")),
        ("rope_theta", ("rope_theta",)),
        ("tie_word_embeddings", ("tie_word_embeddings",)),
        ("experts", ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")),
    ]
)
_NUMERIC = ("layers", "hidden_size", "attention_heads", "kv_heads", "head_dim", "intermediate_size", "vocab_size")

_SAMPLE = "The quick brown fox jumps over the lazy dog, then naps under a tree."


def classify_param(name: str) -> str:
    """Component for a parameter name (kept for backwards compatibility)."""
    return path_category(name.rsplit(".", 1)[0]) or "other"


def _config_values(lm):
    found = {}
    for cfg in lm.configs():
        for label, attrs in _CONFIG_FIELDS.items():
            if label in found:
                continue
            for attr in attrs:
                value = cfg.get(attr) if isinstance(cfg, dict) else getattr(cfg, attr, None)
                if value is not None and not callable(value) and not isinstance(value, (dict, list)):
                    found[label] = value
                    break
    if found.get("intermediate_size") is None and found.get("hidden_size") and \
            isinstance(found.get("hidden_size"), int):
        for cfg in lm.configs():
            mult = cfg.get("ffn_mult") if isinstance(cfg, dict) else getattr(cfg, "ffn_mult", None)
            if isinstance(mult, (int, float)):
                found["intermediate_size"] = int(found["hidden_size"] * mult)
                break
    return found


class ArchitectureAnalyzer(Analyzer):
    name = "architecture"
    title = "Architecture"
    description = "Structure (layers, heads, positions, norms, activation), parameters, memory and tokenizer facts."

    def run(self, model, tokenizer, config: RunConfig):
        lm = as_model(model, tokenizer)
        tok = as_tokenizer(tokenizer) if tokenizer is not None else lm.tok
        if not lm.is_module:
            return self.result(
                f"{lm.name} is not a PyTorch module, so its structure can't be inspected. The other checks "
                "still work through its forward function.",
                status=INFO,
            )
        module = lm.module

        # ---- structure from the module tree, and config values when there is a config
        vocab_hint = getattr(lm, "vocab_out", None) if lm._call is not None else None
        cfg_values = _config_values(lm)
        int_or_none = lambda v: v if isinstance(v, int) and not isinstance(v, bool) else None  # noqa: E731
        structure = inspect_structure(module, vocab_hint=vocab_hint,
                                      heads_hint=int_or_none(cfg_values.get("attention_heads")),
                                      head_dim_hint=int_or_none(cfg_values.get("head_dim")))
        prefer_config = lm.is_hf
        shape, sources, mismatches = {}, {}, []
        for label in _NUMERIC + ("activation", "model_type", "rope_theta", "tie_word_embeddings", "experts"):
            s_val = structure.get(label if label != "tie_word_embeddings" else "tied_embeddings")
            c_val = cfg_values.get(label)
            if label in _NUMERIC and s_val is not None and isinstance(c_val, int) and s_val != c_val \
                    and not (label == "vocab_size" and c_val < s_val):
                mismatches.append(f"{label.replace('_', ' ')}: config says {c_val}, the modules have {s_val}")
            order = (("config", c_val), ("structure", s_val))
            # Module classes give cleaner activation names ("SwiGLU (gated SiLU)" rather than "silu").
            config_first = prefer_config and label != "activation"
            for source, value in (order if config_first else order[::-1]):
                if value is not None:
                    shape[label] = value
                    sources[label] = source
                    break
        for label in ("family", "position_encoding", "attention_type", "mlp_type", "norm_type", "final_norm",
                      "linear_bias", "block_class", "blocks_path", "output_head", "norms_per_block"):
            if label in structure:
                shape[label] = structure[label]
        if "tie_word_embeddings" in shape:
            shape["tied_embeddings"] = shape.pop("tie_word_embeddings")
        # Head size and KV heads follow from the final hidden size and head count.
        heads, hidden = shape.get("attention_heads"), shape.get("hidden_size")
        if isinstance(heads, int) and heads and isinstance(hidden, int) and "head_dim" not in shape \
                and hidden % heads == 0:
            shape["head_dim"] = hidden // heads
        if isinstance(heads, int) and "kv_heads" not in shape and structure.get("attention_type", "").startswith("multi-head"):
            shape["kv_heads"] = heads
        kv = shape.get("kv_heads")
        if isinstance(heads, int) and isinstance(kv, int) and kv and shape.get("attention_type"):
            shape["attention_type"] = (
                "multi-head (MHA)" if kv == heads else "multi-query (MQA)" if kv == 1
                else f"grouped-query (GQA, {heads // kv} query heads per KV head)")
        if isinstance(shape.get("experts"), int) and "experts" not in str(shape.get("mlp_type", "")):
            shape["mlp_type"] = f"mixture of experts ({shape['experts']} experts)"
        if shape.get("experts") and "experts" not in str(shape.get("family", "")):
            shape["family"] = f"{shape.get('family', 'model')} with mixture of experts ({shape['experts']} experts per layer)"
        _, blocks = repeated_blocks(module)
        placement = norm_placement(lm, blocks) if blocks and shape.get("norm_type") else None
        if placement:
            shape["norm_placement"] = placement

        # ---- parameters (tied weights counted once, quantized weights included)
        vocab = shape.get("vocab_size")
        breakdown, total, dtypes = param_breakdown(module, vocab if isinstance(vocab, int) else None)
        breakdown = {k: breakdown[k] for k in _COMPONENTS if k in breakdown}
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        weight_bytes += dtypes.get("qint8", 0)  # one byte per quantized weight (scales not counted)
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
        pos = find_position_embedding(module)
        pos_params = pos.weight.numel() if pos is not None else 0

        context, context_source = lm.context_info
        device = str(lm.device)
        main_dtype = max(dtypes, key=dtypes.get) if dtypes else "unknown"

        # ---- KV cache per token (keys + values, every layer), in the model's dtype
        kv_per_token = None
        layers = shape.get("layers")
        kv_heads = shape.get("kv_heads")
        head_dim = shape.get("head_dim")
        if all(isinstance(v, int) and v for v in (layers, kv_heads, head_dim)) and \
                "transformer" in str(shape.get("family", "")):
            param_dtype = lm.dtype or torch.float32
            elem = torch.tensor([], dtype=param_dtype).element_size() if param_dtype.is_floating_point else 4
            kv_per_token = 2 * layers * kv_heads * head_dim * elem

        # ---- tokenizer
        tok_info = {}
        notes = []
        status = OK if total else INFO
        if tok is not None:
            ids = tok.encode(_SAMPLE)
            tok_vocab = tok.vocab_size
            tok_info = {
                "class": tok.name,
                "vocab_size": tok_vocab,
                "special_tokens": dict(list(tok.special_tokens.items())[:20]),
                "stop_tokens": sorted(tok.stop_ids),
                "has_chat_template": bool(tok.chat_template),
                "sample_text": _SAMPLE,
                "sample_tokens": len(ids),
                "chars_per_token": round(len(_SAMPLE) / max(1, len(ids)), 2),
            }
            if isinstance(vocab, int) and isinstance(tok_vocab, int):
                if tok_vocab > vocab:
                    status = WARNING
                    notes.append(f"The tokenizer has {tok_vocab:,} tokens but the model only has {vocab:,} "
                                 "output scores: this is probably the wrong tokenizer.")
                elif tok_vocab < vocab:
                    notes.append(f"The model's vocabulary ({vocab:,}) is larger than the tokenizer's "
                                 f"({tok_vocab:,}); usually harmless padding for speed.")

        metrics = {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 2) if total else 0.0,
            "params_without_position_embeddings": total - pos_params if pos_params else None,
            "layers": layers if isinstance(layers, int) else None,
            "hidden_size": hidden if isinstance(hidden, int) else None,
            "attention_heads": heads if isinstance(heads, int) else None,
            "kv_heads": kv_heads if isinstance(kv_heads, int) else None,
            "head_dim": head_dim if isinstance(head_dim, int) else None,
            "intermediate_size": shape.get("intermediate_size") if isinstance(shape.get("intermediate_size"), int)
            else None,
            "vocab_size": vocab if isinstance(vocab, int) else None,
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
            if context:
                metrics["kv_cache_bytes_full_context"] = kv_per_token * context
        if tok_info:
            metrics["chars_per_token"] = tok_info["chars_per_token"]
        metrics = {k: v for k, v in metrics.items() if v is not None}

        # ---- summary
        cfg = getattr(module, "config", None)
        arch = (getattr(cfg, "architectures", None) if lm.is_hf else None) or [type(module).__name__]
        family = shape.get("family", "model")
        parts = [f"{arch[0]} ({family}) with {human_number(total)} parameters"]
        if isinstance(layers, int):
            size = f"{layers} layers" + (f" x {hidden} hidden" if isinstance(hidden, int) else "")
            parts.append(size)
        if isinstance(heads, int):
            att = f"{heads} heads"
            if shape.get("attention_type") and not shape["attention_type"].startswith("multi-head"):
                att += f", {shape['attention_type'].split(' (')[0]}"
            parts.append(att)
        extra = []
        if shape.get("position_encoding") and not shape["position_encoding"].startswith("none"):
            extra.append(shape["position_encoding"].split(" (")[0] + " positions")
        if shape.get("norm_type"):
            extra.append(" ".join(x for x in (shape.get("norm_placement"), shape["norm_type"]) if x))
        if shape.get("activation"):
            extra.append(str(shape["activation"]))
        if shape.get("tied_embeddings"):
            extra.append("tied input/output embeddings")
        summary = ", ".join(parts)
        if extra:
            summary += "; " + ", ".join(extra)
        summary += f". {main_dtype} weights use {human_bytes(weight_bytes)}"
        if context:
            summary += f"; context window {context:,} tokens."
        elif context_source.startswith("unlimited"):
            summary += "; no fixed context window (recurrent)."
        else:
            summary += "; context window not detected."

        if trainable != total and total:
            notes.append(f"{metrics['trainable_pct']}% of parameters are trainable (frozen layers, adapters or "
                         "quantized weights).")
        if pos_params:
            notes.append(f"{human_number(total - pos_params)} parameters without the position embeddings "
                         "(some projects quote this number).")
        if context:
            notes.append(f"Context window from the {context_source}.")
        elif context_source.startswith("unlimited"):
            notes.append("Recurrent model: no fixed context window. Long texts are cut at 2,048 tokens.")
        else:
            notes.append("Context window not found: pass context_length=... to analyze() so long prompts get cut "
                         "correctly.")
        if dtypes.get("qint8"):
            notes.append("Dynamically quantized INT8 layers found; their weights are included in the counts.")
            if not shape.get("tied_embeddings") and "embeddings" in breakdown and "output_head" in breakdown:
                notes.append("The output layer has its own INT8 copy of the embedding matrix (quantization unties "
                             "shared weights), so this file holds more weights than the float model.")
        if kv_per_token and not lm.uses_model_generate:
            notes.append("The KV-cache figures show what a cache would need. llmreport's own generation loop "
                         "doesn't use one.")
        for m in mismatches:
            notes.append(f"The config and the actual modules disagree on {m}; the report uses the modules.")
        if tok_info and not tok_info["has_chat_template"] and not config.prompt_template:
            notes.append("No chat template or prompt_template: behavior checks send raw prompts (base-model "
                         "style). For an instruction-tuned model, pass prompt_template=\"...{prompt}...\".")

        return self.result(
            summary,
            metrics=metrics,
            details={
                "architecture_class": arch[0],
                "structure": shape,
                "sources": sources,
                "param_breakdown": breakdown,
                "dtypes": dtypes,
                "buffers_bytes": buffer_bytes,
                "tokenizer": tok_info,
            },
            status=status,
            notes=notes,
        )
