"""Small, single-purpose functions, in the spirit of ``np.sum`` and ``np.mean``.

Each one does one job and returns a plain value (a number, a string, a dict), so they
fit into notebooks, training scripts and tests::

    import llmreport as lr

    lr.params(model)                       # 142627840
    lr.info(model, tokenizer)              # everything about the model, right now
    lr.perplexity(model, tokenizer, texts) # 21.4
    lr.generate(model, tokenizer, "def add(a, b):")
    lr.health(model).ok                    # True / False
"""

from __future__ import annotations

import html
import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from . import _utils
from .adapters import LanguageModel, as_model, as_tokenizer
from .analyzers.base import AnalyzerResult, RunConfig
from .healthcheck import HealthReport, grad_norm, health, weight_norm
from .structure import param_breakdown as _breakdown

__all__ = [
    "params", "param_breakdown", "memory", "architecture", "info", "ModelInfo", "health", "grad_norm",
    "weight_norm", "perplexity", "generate", "predict_next", "speed", "check", "text_stats", "code_score",
    "compare",
]


def _module(model) -> nn.Module:
    m = model.module if isinstance(model, LanguageModel) else model
    if not isinstance(m, nn.Module):
        raise TypeError(f"Expected a PyTorch model (nn.Module), got {type(m).__name__}")
    return m


def _lm(model, tokenizer=None, **kw) -> LanguageModel:
    lm = as_model(model, tokenizer, **{k: v for k, v in kw.items() if v is not None})
    if isinstance(model, LanguageModel) and tokenizer is not None:
        lm.tok = as_tokenizer(tokenizer)
    return lm


# ------------------------------------------------------------------ size


def params(model, trainable_only: bool = False) -> int:
    """Number of parameters. Shared (tied) weights count once; quantized INT8 weights are included.

    >>> llmreport.params(model)
    142627840
    """
    module = _module(model)
    if trainable_only:
        seen, n = set(), 0
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n
    return _breakdown(module, None)[1]


def param_breakdown(model) -> Dict[str, int]:
    """Parameters per component: embeddings, attention, mlp, norm, output_head, recurrent, other."""
    module = _module(model)
    vocab = max((m.num_embeddings for m in module.modules() if isinstance(m, nn.Embedding)), default=None)
    return _breakdown(module, vocab)[0]


def memory(model, tokenizer=None, *, context: Optional[int] = None, batch_size: int = 1) -> Dict[str, Any]:
    """Memory now and estimates for inference and training.

    Returns bytes for: the weights as they are, the weights in FP32/FP16/INT8/INT4, the KV cache
    for ``batch_size`` sequences of ``context`` tokens (decoder-only transformers), an inference
    total, and a training estimate for AdamW (weights + gradients + optimizer states, about
    16 bytes per parameter, activations not included).
    """
    module = _module(model)
    arch = architecture(model, tokenizer)
    n = params(module)
    weights = sum(p.numel() * p.element_size() for p in module.parameters())
    weights += arch.get("_qint8", 0)
    out: Dict[str, Any] = {
        "params": n,
        "weights_bytes": weights,
        "buffers_bytes": sum(b.numel() * b.element_size() for b in module.buffers()),
        "fp32_bytes": n * 4,
        "fp16_bytes": n * 2,
        "int8_bytes": n,
        "int4_bytes": n // 2,
        "training_adamw_bytes": n * 16,
    }
    per_token = arch.get("kv_cache_bytes_per_token")
    ctx = context or arch.get("max_context")
    if per_token and ctx:
        out["kv_cache_bytes_per_token"] = per_token
        out["kv_cache_bytes"] = per_token * ctx * batch_size
        out["context"] = ctx
        out["batch_size"] = batch_size
        out["inference_bytes"] = weights + out["kv_cache_bytes"]
    else:
        out["inference_bytes"] = weights
    return out


def architecture(model, tokenizer=None) -> Dict[str, Any]:
    """Structure of the model as a flat dict: family, layers, hidden size, heads, KV heads, feed-forward
    size and type, position encoding, norm type and placement, activation, tying, context, vocabulary."""
    from .analyzers.architecture import ArchitectureAnalyzer

    lm = _lm(model, tokenizer)
    was = lm.training
    lm.eval()  # dropout would make the structure probes random; BatchNorm stats must not change
    try:
        res = ArchitectureAnalyzer().run(lm, lm.tok, RunConfig())
    finally:
        lm.train(was)
    out = dict(res.details.get("structure", {}))
    out.update({k: v for k, v in res.metrics.items() if not k.startswith("est_")})
    out["_qint8"] = res.details.get("dtypes", {}).get("qint8", 0)
    out["summary"] = res.summary
    out["param_breakdown"] = res.details.get("param_breakdown", {})
    out["notes"] = res.notes
    return out


# ------------------------------------------------------------------ everything at once


_SECTIONS = {
    "Overview": ("name", "model_class", "kind", "family", "total_params", "trainable_params", "device", "dtype",
                 "mode"),
    "Structure": ("layers", "encoder_layers", "decoder_layers", "hidden_size", "attention_heads", "kv_heads",
                  "head_dim", "attention_type", "intermediate_size", "mlp_type", "activation", "position_encoding",
                  "norm_type", "norm_placement", "tied_embeddings", "vocab_size", "max_context"),
    "Memory": ("weights_bytes", "kv_cache_bytes_per_token", "inference_bytes", "training_adamw_bytes"),
    "Training state": ("frozen_params", "grad_norm", "weight_norm", "optimizer", "learning_rate",
                       "optimizer_state_bytes"),
    "Tokenizer": ("tokenizer", "tokenizer_vocab", "eos_token", "chat_template"),
    "Health": ("health", "health_issues"),
}


class ModelInfo(dict):
    """Everything :func:`info` found, as a dict with attribute access (``info.layers``).

    Shows as a table in a notebook and as text with ``print``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def _rows(self):
        from .report import format_value, pretty_key

        for section, keys in _SECTIONS.items():
            rows = [(pretty_key(k), format_value(k, self[k])) for k in keys
                    if k in self and self[k] not in (None, "", [], {})]
            if rows:
                yield section, rows

    def __repr__(self) -> str:
        lines = [f"llmreport.info: {self.get('name')}"]
        for section, rows in self._rows():
            lines.append(f"  {section}")
            width = max(len(k) for k, _ in rows)
            lines += [f"    {k:<{width}}  {v}" for k, v in rows]
        for issue in self.get("health_details", [])[:5]:
            lines.append(f"    - {issue}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        body = ""
        for section, rows in self._rows():
            body += (f"<tr><th colspan='2' style='text-align:left;padding-top:10px'>{html.escape(section)}</th></tr>"
                     + "".join(f"<tr><td style='padding:2px 16px 2px 8px;opacity:.75'>{html.escape(k)}</td>"
                               f"<td>{html.escape(v)}</td></tr>" for k, v in rows))
        return (f"<div style='font-family:system-ui,sans-serif;font-size:14px'><b>{html.escape(str(self.get('name')))}"
                f"</b><table style='border-collapse:collapse'>{body}</table></div>")


def info(model, tokenizer=None, *, optimizer=None, context: Optional[int] = None) -> ModelInfo:
    """Everything about a model as it is right now, including while it is training.

    Covers what it is (kind, family, structure), its size and memory, its live state
    (train/eval mode, frozen parameters, gradient and weight norms, optimizer and learning
    rate), the tokenizer, and a weight health check. Nothing is generated, so it is fast.

    >>> llmreport.info(model, tokenizer, optimizer=opt)
    """
    lm = _lm(model, tokenizer)
    module = _module(lm)
    if lm.device.type != "meta":
        was = lm.training
        lm.eval()
        try:
            lm.probe()
        except Exception:
            pass
        finally:
            lm.train(was)
    arch = architecture(lm, lm.tok)
    mem = memory(lm, lm.tok, context=context)
    total = params(module)
    trainable = params(module, trainable_only=True)
    out = ModelInfo(
        name=lm.name,
        model_class=type(module).__name__,
        kind=lm.kind_label,
        mode="training" if module.training else "eval",
        total_params=total,
        trainable_params=trainable,
        frozen_params=total - trainable if total - trainable else None,
        device=str(lm.device),
        dtype=arch.get("dtype"),
    )
    for key in _SECTIONS["Structure"] + ("family", "summary", "param_breakdown"):
        if key in arch:
            out[key] = arch[key]
    out.update({k: mem[k] for k in ("weights_bytes", "kv_cache_bytes_per_token", "inference_bytes",
                                    "training_adamw_bytes") if k in mem})
    if lm.device.type != "meta":
        out["grad_norm"] = grad_norm(module)
        out["weight_norm"] = weight_norm(module)
        h = health(module)
        out["health"] = h.summary
        out["health_status"] = h.status
        out["health_issues"] = len(h.issues) or None
        out["health_details"] = [str(i) for i in h.issues]
    if optimizer is not None:
        out["optimizer"] = type(optimizer).__name__
        lrs = sorted({g.get("lr") for g in optimizer.param_groups if g.get("lr") is not None})
        out["learning_rate"] = lrs[0] if len(lrs) == 1 else lrs
        state_bytes = 0
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    state_bytes += v.numel() * v.element_size()
        out["optimizer_state_bytes"] = state_bytes
    tok = lm.tok
    if tok is not None:
        out["tokenizer"] = tok.name
        out["tokenizer_vocab"] = tok.vocab_size
        out["eos_token"] = tok.eos_token
        out["chat_template"] = bool(tok.chat_template) or None
    return out


# ------------------------------------------------------------------ behavior


def _default_texts() -> List[str]:
    return [t["text"] for t in _utils.load_data("perplexity")["texts"]]


def perplexity(model, tokenizer, texts: Union[str, Sequence[str], None] = None, *, details: bool = False,
               max_length: Optional[int] = None):
    """Perplexity on ``texts`` (default: the built-in everyday-English passages). Lower is better.

    Pass your validation texts to measure your own domain, e.g. a list of Python files.
    With ``details=True`` returns a dict with ``perplexity``, ``bits_per_char``, ``mean_nll``,
    ``tokens`` and per-text values.
    """
    if isinstance(texts, str):
        texts = [texts]
    lm = _lm(model, tokenizer)
    was = lm.training
    lm.eval()
    try:
        res = _utils.perplexity(lm, lm.tok, list(texts) if texts is not None else _default_texts(), max_length)
    finally:
        lm.train(was)
    return res if details else res["perplexity"]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64, *, template: Optional[str] = None,
             chat: Any = "auto") -> str:
    """Greedy completion of ``prompt`` (only the new text). Works for any model llmreport supports.

    ``template`` wraps the prompt, for example ``"<|instruction|>{prompt}<|response|>"``.
    """
    lm = _lm(model, tokenizer)
    was = lm.training
    lm.eval()
    try:
        return _utils.generate(lm, lm.tok, prompt, max_new_tokens,
                               RunConfig(use_chat_template=chat, prompt_template=template))
    finally:
        lm.train(was)


@torch.no_grad()
def predict_next(model, tokenizer, text: str, k: int = 5) -> List[Tuple[str, float]]:
    """The ``k`` most likely next tokens after ``text``, with probabilities.

    >>> llmreport.predict_next(model, tok, "The capital of France is")
    [(' Paris', 0.62), (' a', 0.05), ...]
    """
    lm = _lm(model, tokenizer)
    was = lm.training
    lm.eval()
    try:
        ids = lm.tok.encode(text, add_special=True)[-lm.usable_context():]
        probs = torch.softmax(lm.last_logits(torch.tensor([ids], device=lm.device))[0].float(), dim=-1)
        top = torch.topk(probs, min(k, probs.numel()))
        return [(lm.tok.decode([int(i)], skip_special=False), round(float(p), 6))
                for p, i in zip(top.values, top.indices)]
    finally:
        lm.train(was)


def speed(model, tokenizer, *, prompt_tokens: int = 128, new_tokens: int = 32, repeats: int = 3) -> Dict[str, Any]:
    """Time to first token (ms) and generation speed (tokens/s) for one prompt length."""
    res = check("performance", model, tokenizer,
                options={"performance": {"prompt_lengths": [prompt_tokens], "repeats": repeats,
                                         "new_tokens": new_tokens}})
    return dict(res.metrics)


def code_score(model, tokenizer, *, template: Optional[str] = None, mode: str = "quick", execute: bool = True
               ) -> Dict[str, Any]:
    """Pass rate on small Python tasks with unit tests (see the ``code`` check)."""
    res = check("code", model, tokenizer, prompt_template=template, mode=mode,
                options={"code": {"execute": execute}})
    return dict(res.metrics, summary=res.summary)


def check(name: str, model, tokenizer, **kwargs) -> AnalyzerResult:
    """Run one check and return its result, for example ``llmreport.check("repetition", model, tok)``."""
    from .core import analyze

    return analyze(model, tokenizer, checks=[name], verbose=False, **kwargs)[name]


def text_stats(text: Union[str, Iterable[str]]) -> Dict[str, float]:
    """Repetition and variety of generated text: repeated 3-gram rate, distinct-1, distinct-2, words."""
    texts = [text] if isinstance(text, str) else list(text)
    rows = []
    for t in texts:
        w = _utils.words(t)
        rows.append((_utils.repeated_ngram_rate(w, 3), _utils.distinct_n(w, 1), _utils.distinct_n(w, 2), len(w)))
    mean = lambda i: statistics.fmean(r[i] for r in rows) if rows else math.nan  # noqa: E731
    return {"repeated_3gram_rate": mean(0), "distinct_1": mean(1), "distinct_2": mean(2), "words": mean(3)}


def compare(a, b, names: Optional[Tuple[str, str]] = None):
    """Compare two reports (from :func:`llmreport.analyze`) metric by metric."""
    return a.compare(b, names=names) if names else a.compare(b)


__all__ += ["HealthReport"]
