"""Adapters that let llmreport work with any PyTorch language model and any tokenizer.

Every check talks to the model through :class:`LanguageModel` and to the tokenizer
through :class:`TokenizerAdapter`. They only rely on the one thing every text
language model does: turn a batch of token ids into next-token scores (logits).
That covers Hugging Face models, from-scratch GPTs, Llama-style models with RoPE,
recurrent models, TorchScript and dynamically quantized models, and anything
else you can call with token ids.
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

__all__ = ["LanguageModel", "TokenizerAdapter", "NotSupportedByModel", "as_model", "as_tokenizer"]


# ============================================================== tokenizers

# Special tokens that end a text or a turn ("<|endoftext|>", "</s>", "<|eot_id|>", ...).
_STOP_TOKEN = re.compile(r"^(</s>|<\|?[^<>]*(end|eos|eot)[^<>]*\|?>)$", re.IGNORECASE)
_BOS_NAMES = ("<s>", "<bos>", "<|begin_of_text|>", "<|startoftext|>", "<|bos|>", "[CLS]")
_EOS_PREFERRED = ("<|endoftext|>", "</s>", "<eos>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|end|>")


def _load_tokenizer_path(path: str):
    """Load a tokenizer from a file, a folder or a Hugging Face Hub name."""
    if os.path.isfile(path):
        lowered = path.lower()
        if lowered.endswith(".json"):
            from tokenizers import Tokenizer

            return Tokenizer.from_file(path)
        if lowered.endswith(".model"):
            import sentencepiece as spm

            return spm.SentencePieceProcessor(model_file=path)
        raise ValueError(f"Don't know how to load a tokenizer from {path!r}. Use a tokenizer.json, a "
                         "SentencePiece .model file, a folder, or pass a tokenizer object.")
    if os.path.isdir(path):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(path)
        except Exception:
            for name in ("tokenizer.json",):
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    return _load_tokenizer_path(candidate)
            for name in ("tokenizer.model", "spm.model", "sentencepiece.model"):
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    return _load_tokenizer_path(candidate)
            vocab, merges = os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt")
            if os.path.isfile(vocab) and os.path.isfile(merges):
                from tokenizers import ByteLevelBPETokenizer

                return ByteLevelBPETokenizer(vocab, merges)._tokenizer
            raise
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _tokenizer_kind(raw) -> str:
    module = type(raw).__module__ or ""
    if hasattr(raw, "convert_ids_to_tokens") and hasattr(raw, "decode") and callable(raw):
        return "hf"
    if module.startswith("tokenizers") and hasattr(raw, "encode") and hasattr(raw, "get_vocab_size"):
        return "tokenizers"
    if hasattr(raw, "EncodeAsIds") or hasattr(raw, "encode_as_ids"):
        return "sentencepiece"
    if hasattr(raw, "encode_ordinary") and hasattr(raw, "n_vocab"):
        return "tiktoken"
    if hasattr(raw, "encode") and hasattr(raw, "decode"):
        return "generic"
    raise TypeError(
        f"{type(raw).__name__} is not a tokenizer llmreport understands. Pass a Hugging Face tokenizer, a "
        "`tokenizers.Tokenizer`, a SentencePiece or tiktoken tokenizer, a path to tokenizer.json, or any "
        "object with encode(text) -> ids and decode(ids) -> text."
    )


def _as_id_list(value) -> List[int]:
    if hasattr(value, "ids"):
        value = value.ids
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if torch.is_tensor(value):
        value = value.flatten().tolist()
    return [int(v) for v in value]


_VOCAB_ATTRS = ("special_tokens", "added_tokens", "stoi", "vocab", "encoder", "token_to_id", "tok2id",
                "word2idx", "char2idx", "token2idx", "index")
_SPECIAL_TEXT = re.compile(r"<\|[^<>|]+\|>|</?s>|<unk>|<pad>|<bos>|<eos>|\[(CLS|SEP|PAD|UNK|EOS|BOS)\]")


def _generic_specials(raw) -> Dict[str, int]:
    """Special tokens of a hand-written tokenizer, found in its vocabulary dict or id attributes."""
    out: Dict[str, int] = {}
    for attr in _VOCAB_ATTRS:
        value = getattr(raw, attr, None)
        if callable(value) and not isinstance(value, dict):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, dict):
            for tok, tid in value.items():
                if isinstance(tok, str) and isinstance(tid, int) and _SPECIAL_TEXT.fullmatch(tok):
                    out[tok] = tid
    for attr, name in (("eos_token_id", "<eos>"), ("eos_id", "<eos>"), ("eot_token", "<|endoftext|>"),
                       ("eot_id", "<|endoftext|>"), ("bos_token_id", "<bos>"), ("bos_id", "<bos>")):
        value = getattr(raw, attr, None)
        value = value() if callable(value) else value
        if isinstance(value, int) and value >= 0 and value not in out.values():
            out[name] = value
    return out


class TokenizerAdapter:
    """One interface over Hugging Face, `tokenizers`, SentencePiece, tiktoken and custom tokenizers.

    Attribute lookups that the adapter doesn't define are forwarded to the wrapped
    tokenizer, so code written for Hugging Face tokenizers keeps working.
    """

    def __init__(self, tokenizer, stop_tokens: Sequence[str] = ()):
        if isinstance(tokenizer, TokenizerAdapter):
            tokenizer = tokenizer.raw
        if isinstance(tokenizer, (str, os.PathLike)):
            tokenizer = _load_tokenizer_path(os.fspath(tokenizer))
        self.raw = tokenizer
        self.kind = _tokenizer_kind(tokenizer)
        self.extra_stop_tokens = list(stop_tokens)
        self._specials: Optional[Dict[str, int]] = None

    # ---- forwarding for backwards compatibility
    def __getattr__(self, name):
        if name == "raw":  # not set yet (during unpickling)
            raise AttributeError(name)
        return getattr(self.raw, name)

    def __call__(self, *args, **kwargs):
        return self.raw(*args, **kwargs)

    def __len__(self) -> int:
        return self.vocab_size or 0

    def __repr__(self) -> str:
        return f"TokenizerAdapter({type(self.raw).__name__}, kind={self.kind!r})"

    @property
    def name(self) -> str:
        return type(self.raw).__name__

    # ---- core API
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        raw, kind = self.raw, self.kind
        if kind == "hf":
            return list(raw(text, add_special_tokens=add_special)["input_ids"])
        if kind == "tokenizers":
            return list(raw.encode(text, add_special_tokens=add_special).ids)
        if kind == "sentencepiece":
            ids = list(raw.EncodeAsIds(text) if hasattr(raw, "EncodeAsIds") else raw.encode_as_ids(text))
            bos = self.bos_id
            return ([bos] if add_special and bos is not None else []) + ids
        if kind == "tiktoken":
            return list(raw.encode(text, allowed_special="all"))
        return _as_id_list(raw.encode(text))

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        ids = [int(i) for i in (ids.tolist() if torch.is_tensor(ids) else ids)]
        raw, kind = self.raw, self.kind
        if kind == "hf":
            return raw.decode(ids, skip_special_tokens=skip_special)
        if kind == "tokenizers":
            return raw.decode(ids, skip_special_tokens=skip_special)
        if kind == "sentencepiece":
            return raw.DecodeIds(ids) if hasattr(raw, "DecodeIds") else raw.decode_ids(ids)
        if skip_special:
            special = set(self.special_tokens.values())
            ids = [i for i in ids if i not in special]
        return raw.decode(ids)

    # ---- facts
    @property
    def vocab_size(self) -> Optional[int]:
        raw, kind = self.raw, self.kind
        try:
            if kind == "hf":
                return len(raw)
            if kind == "tokenizers":
                return raw.get_vocab_size(with_added_tokens=True)
            if kind == "sentencepiece":
                return raw.GetPieceSize() if hasattr(raw, "GetPieceSize") else raw.get_piece_size()
            if kind == "tiktoken":
                return raw.n_vocab
            for attr in ("vocab_size", "n_vocab"):
                value = getattr(raw, attr, None)
                value = value() if callable(value) else value
                if isinstance(value, int):
                    return value
            return len(raw)
        except Exception:
            return None

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Mapping of special token text to id."""
        if self._specials is not None:
            return self._specials
        raw, kind = self.raw, self.kind
        out: Dict[str, int] = {}
        try:
            if kind == "hf":
                for tok in raw.all_special_tokens:
                    tid = raw.convert_tokens_to_ids(tok)
                    if isinstance(tid, int) and tid >= 0:
                        out[tok] = tid
                for tid, added in getattr(raw, "added_tokens_decoder", {}).items():
                    if getattr(added, "special", False):
                        out.setdefault(str(added), int(tid))
            elif kind == "tokenizers":
                decoder = raw.get_added_tokens_decoder() if hasattr(raw, "get_added_tokens_decoder") else {}
                for tid, added in decoder.items():
                    if getattr(added, "special", True):
                        out[added.content] = int(tid)
                if not out:
                    for tok, tid in raw.get_vocab(with_added_tokens=True).items():
                        if re.fullmatch(r"<\|[^<>|]+\|>|</?s>|<unk>|<pad>", tok):
                            out[tok] = tid
            elif kind == "sentencepiece":
                for name, fn in (("<s>", "bos_id"), ("</s>", "eos_id"), ("<unk>", "unk_id"), ("<pad>", "pad_id")):
                    tid = getattr(raw, fn)()
                    if tid is not None and tid >= 0:
                        out[raw.IdToPiece(tid) if hasattr(raw, "IdToPiece") else name] = tid
            elif kind == "tiktoken":
                for tok in raw.special_tokens_set:
                    out[tok] = raw.encode_single_token(tok)
            else:
                out.update(_generic_specials(raw))
        except Exception:
            pass
        self._specials = out
        return out

    def _first_special(self, names) -> Optional[int]:
        specials = self.special_tokens
        for name in names:
            if name in specials:
                return specials[name]
        return None

    @property
    def bos_id(self) -> Optional[int]:
        raw, kind = self.raw, self.kind
        if kind == "hf":
            return raw.bos_token_id
        if kind == "sentencepiece":
            tid = raw.bos_id()
            return tid if tid >= 0 else None
        return self._first_special(_BOS_NAMES)

    @property
    def eos_id(self) -> Optional[int]:
        raw, kind = self.raw, self.kind
        if kind == "hf":
            value = raw.eos_token_id
            return value[0] if isinstance(value, (list, tuple)) else value
        if kind == "sentencepiece":
            tid = raw.eos_id()
            return tid if tid >= 0 else None
        if kind == "tiktoken":
            return getattr(raw, "eot_token", None)
        tid = self._first_special(_EOS_PREFERRED)
        if tid is None:
            stops = sorted(self.stop_ids)
            tid = stops[0] if stops else None
        return tid

    def _token_text(self, tid: Optional[int]) -> Optional[str]:
        if tid is None:
            return None
        for text, value in self.special_tokens.items():
            if value == tid:
                return text
        try:
            return self.decode([tid], skip_special=False) or None
        except Exception:
            return None

    @property
    def bos_token(self) -> Optional[str]:
        if self.kind == "hf":
            return self.raw.bos_token
        return self._token_text(self.bos_id)

    @property
    def eos_token(self) -> Optional[str]:
        if self.kind == "hf":
            return self.raw.eos_token
        return self._token_text(self.eos_id)

    @property
    def pad_id(self) -> int:
        if self.kind == "hf":
            for value in (self.raw.pad_token_id, self.raw.eos_token_id):
                if value is not None:
                    return value if isinstance(value, int) else value[0]
        eos = self.eos_id
        return eos if eos is not None else 0

    @property
    def stop_ids(self) -> Set[int]:
        """Token ids that end generation: the EOS token plus end-of-text/turn special tokens."""
        stops: Set[int] = set()
        if self.kind in ("hf", "sentencepiece", "tiktoken"):
            eos = self.eos_id
            if eos is not None:
                stops.add(int(eos))
        for text, tid in self.special_tokens.items():
            low = text.lower()
            if _STOP_TOKEN.match(text) and "header" not in low and "start" not in low:
                stops.add(int(tid))
        for text in self.extra_stop_tokens:
            tid = self.special_tokens.get(text)
            if tid is None:
                ids = self.encode(text)
                tid = ids[0] if len(ids) == 1 else None
            if tid is None:
                raise ValueError(f"Stop token {text!r} is not a single token in this tokenizer.")
            stops.add(int(tid))
        return stops

    @property
    def chat_template(self) -> Optional[str]:
        return getattr(self.raw, "chat_template", None) if self.kind == "hf" else None

    def apply_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.raw.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def as_tokenizer(tokenizer, stop_tokens: Sequence[str] = ()) -> Optional[TokenizerAdapter]:
    if tokenizer is None or isinstance(tokenizer, TokenizerAdapter):
        return tokenizer
    return TokenizerAdapter(tokenizer, stop_tokens)


# ============================================================== models

_CONTEXT_ATTRS = (
    "max_position_embeddings", "n_positions", "block_size", "max_seq_len", "max_seq_length",
    "max_sequence_length", "context_length", "context_len", "context_size", "seq_len", "seq_length",
    "n_ctx", "max_len", "max_length_positions",
)
_OFFSET_POSITION_MODELS = ("roberta", "camembert", "longformer", "data2vec-text", "ibert", "luke", "markuplm",
                           "mpnet", "esm")
_CONFIG_ATTRS = ("config", "cfg", "params", "args", "hparams", "model_args", "conf")
_TARGET_PARAMS = ("targets", "target", "labels", "y")

KIND_LABELS = {
    "causal": "causal (left-to-right) language model",
    "seq2seq": "encoder-decoder (sequence-to-sequence) model",
    "masked": "masked language model (BERT-style)",
    "encoder": "base model without a language-model head",
}


class NotSupportedByModel(Exception):
    """The model can't do what a check needs (for example, a BERT-style model can't generate text)."""


def _is_hf_model(module) -> bool:
    try:
        from transformers import PreTrainedModel
    except Exception:  # pragma: no cover - transformers is a dependency, but stay safe
        return False
    return isinstance(module, PreTrainedModel)


def _signature_params(fn) -> Tuple[List[str], bool]:
    """Parameter names of ``fn`` (without self) and whether it accepts **kwargs."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return [], True
    names = [p.name for p in sig.parameters.values() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    return names, has_kwargs


def _config_value(cfg, attr):
    if cfg is None:
        return None
    value = cfg.get(attr) if isinstance(cfg, dict) else getattr(cfg, attr, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _find_loss(out):
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    if isinstance(out, dict) and out.get("loss") is not None:
        return out["loss"]
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item) and item.dim() == 0:
                return item
    return None


class _float_linears:
    """Context manager: temporarily replace dynamically quantized Linear layers with float copies."""

    def __init__(self, module):
        self.module = module
        self.swapped = []

    def __enter__(self):
        if self.module is None:
            return self
        for parent in list(self.module.modules()):
            for name, child in list(parent.named_children()):
                if hasattr(child, "_packed_params") and callable(getattr(child, "weight", None)):
                    w = child.weight().dequantize()
                    bias = child.bias() if callable(getattr(child, "bias", None)) else None
                    lin = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None)
                    with torch.no_grad():
                        lin.weight.copy_(w)
                        if bias is not None:
                            lin.bias.copy_(bias)
                    setattr(parent, name, lin)
                    self.swapped.append((parent, name, child))
        return self

    def __exit__(self, *exc):
        for parent, name, child in self.swapped:
            setattr(parent, name, child)
        self.swapped.clear()
        return False


class LanguageModel:
    """Uniform access to any PyTorch language model.

    Args:
        model: An ``nn.Module`` (Hugging Face or from scratch), a TorchScript module,
            or any callable that maps a ``(batch, seq)`` LongTensor of token ids to logits.
        tokenizer: The matching tokenizer (anything :class:`TokenizerAdapter` accepts).
        context_length: Maximum sequence length. Detected automatically when left out.
        forward: Optional function ``ids -> logits`` to use instead of calling the model,
            for models that need extra arguments (for example ``lambda ids: model(ids, 0)``).
        generation: ``"auto"`` uses the model's own Hugging Face-style ``generate()`` when it
            has one and llmreport's built-in greedy loop otherwise; ``"builtin"`` always uses
            the built-in loop; ``"model"`` always uses ``model.generate``.

    Attribute lookups the adapter doesn't define are forwarded to the wrapped model.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        context_length: Optional[int] = None,
        forward: Optional[Callable] = None,
        generation: str = "auto",
    ):
        if isinstance(model, LanguageModel):
            model = model.module
        if generation not in ("auto", "builtin", "model"):
            raise ValueError("generation must be 'auto', 'builtin' or 'model'")
        self.module = model
        self.tok = as_tokenizer(tokenizer)
        self._forward_override = forward
        self._context_override = context_length
        self.generation = generation
        self._call: Optional[str] = None  # probed calling convention
        self._layout: Optional[str] = None  # "bt", "tb", "flat" or "last"
        self._context: Optional[Tuple[Optional[int], str]] = None
        self.vocab_out: Optional[int] = None  # size of the model's output layer, known after probe()
        self._kind: Optional[str] = None

    # ---- forwarding for backwards compatibility
    def __getattr__(self, name):
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __repr__(self) -> str:
        return f"LanguageModel({self.name})"

    # ---- facts
    @property
    def is_module(self) -> bool:
        return isinstance(self.module, nn.Module)

    @property
    def is_hf(self) -> bool:
        return _is_hf_model(self.module)

    @property
    def name(self) -> str:
        cfg = getattr(self.module, "config", None)
        for attr in ("_name_or_path", "name_or_path"):
            value = getattr(cfg, attr, None)
            if isinstance(value, str) and value:
                return value
        return type(self.module).__name__ if self.is_module else getattr(self.module, "__name__", "model")

    @property
    def device(self) -> torch.device:
        if self.is_module:
            for t in self.module.parameters():
                return t.device
            for t in self.module.buffers():
                return t.device
        return torch.device("cpu")

    @property
    def dtype(self) -> Optional[torch.dtype]:
        if self.is_module:
            for t in self.module.parameters():
                return t.dtype
        return None

    @property
    def training(self) -> bool:
        return bool(getattr(self.module, "training", False))

    def eval(self):
        if hasattr(self.module, "eval"):
            self.module.eval()
        return self

    def train(self, mode: bool = True):
        if hasattr(self.module, "train"):
            self.module.train(mode)
        return self

    def configs(self) -> List[Any]:
        """Config-like objects attached to the model (HF config, dataclasses, dicts, argparse args)."""
        found = []
        for attr in _CONFIG_ATTRS:
            value = getattr(self.module, attr, None) if self.is_module or hasattr(self.module, attr) else None
            if value is not None and not callable(value) and not torch.is_tensor(value) \
                    and not isinstance(value, nn.Module):
                found.append(value)
        return found

    @property
    def context_length(self) -> Optional[int]:
        return self.context_info[0]

    @property
    def context_info(self) -> Tuple[Optional[int], str]:
        """(context length or None, where it came from)."""
        if self._context is None:
            self._context = self._find_context()
        return self._context

    def _find_context(self) -> Tuple[Optional[int], str]:
        if self._context_override:
            return int(self._context_override), "set by you"
        for cfg in self.configs():
            for attr in _CONTEXT_ATTRS:
                value = _config_value(cfg, attr)
                if value:
                    # RoBERTa-style models start positions after the padding index, so two slots are unusable.
                    if attr == "max_position_embeddings" and self.is_hf and \
                            any(k in str(getattr(cfg, "model_type", "")) for k in _OFFSET_POSITION_MODELS):
                        pad = getattr(cfg, "pad_token_id", 1)
                        value -= (pad if isinstance(pad, int) else 1) + 1
                    return value, f"model config ({attr})"
        for attr in _CONTEXT_ATTRS:
            value = _config_value(self.module, attr)
            if value:
                return value, f"model attribute ({attr})"
        if self.is_module:
            from .structure import find_position_embedding

            pos = find_position_embedding(self.module)
            if pos is not None:
                return pos.num_embeddings, "learned position embedding size"
            if self._is_recurrent():
                return None, "unlimited (recurrent or state-space model)"
        tok_max = getattr(getattr(self.tok, "raw", None), "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 10**7:
            return tok_max, "tokenizer model_max_length"
        return None, "unknown"

    def _is_recurrent(self) -> bool:
        from .structure import is_state_space

        return any(isinstance(m, nn.RNNBase) for m in self.module.modules()) or is_state_space(self.module)

    def usable_context(self, fallback: int = 2048) -> int:
        return self.context_length or fallback

    @property
    def uses_model_generate(self) -> bool:
        if self.kind == "seq2seq":
            return True
        if self.generation == "builtin" or self._forward_override is not None:
            return False
        if self.generation == "model":
            return True
        gen = getattr(self.module, "generate", None)
        if gen is None:
            return False
        if self.is_hf:
            return True
        names, _ = _signature_params(gen)
        return "input_ids" in names  # Hugging Face-style signature, so greedy arguments are understood

    # ---- what kind of model is this?
    @property
    def kind(self) -> str:
        """'causal', 'seq2seq', 'masked' or 'encoder' (no language-model head)."""
        if self._kind is None:
            self._kind = self._detect_kind()
        return self._kind

    def _detect_kind(self) -> str:
        if self._forward_override is not None or not self.is_module:
            return "causal"
        m = self.module
        cfg = getattr(m, "config", None)
        if getattr(cfg, "is_encoder_decoder", False):
            return "seq2seq"
        if self.is_hf:
            cls = type(m).__name__
            if "MaskedLM" in cls or (cls.endswith("ForPreTraining") and not getattr(cfg, "is_decoder", False)):
                return "masked"
            get_out = getattr(m, "get_output_embeddings", None)
            try:
                if callable(get_out) and get_out() is None:
                    return "encoder"
            except Exception:
                pass
        return "causal"

    @torch.no_grad()
    def is_causal(self) -> Optional[bool]:
        """True if each position only sees earlier tokens (left-to-right), False if it sees the whole input.

        Found by changing the last token and checking whether the first position's output changes.
        None when it can't be measured.
        """
        if self.kind == "seq2seq":
            return None
        try:
            self.probe()
            if self._layout == "last":
                return True  # only the last position is exposed; used left-to-right
            vocab = self.tok.vocab_size if self.tok is not None else None
            high = max(3, min(vocab or 50, 50))
            a = torch.arange(1, 7, device=self.device).remainder(high - 1).add(1).view(1, -1)
            b = a.clone()
            b[0, -1] = 1 if int(a[0, -1]) != 1 else 2
            # Dynamically quantized layers pick their activation scale from the whole input, which adds
            # noise to every position; run this test with temporary float copies of those layers.
            with _float_linears(self.module if self.is_module else None):
                la, lb = self.logits(a)[0].float(), self.logits(b)[0].float()
            first = float((la[0] - lb[0]).norm())
            return first <= 1e-4 * (float(la[0].norm()) + 1e-12)
        except Exception:
            return None

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def can_generate(self) -> bool:
        return self.kind in ("causal", "seq2seq")

    @property
    def can_score(self) -> bool:
        return self.kind in ("causal", "seq2seq", "masked")

    def _decoder_start_id(self) -> int:
        cfg = getattr(self.module, "config", None)
        gen = getattr(self.module, "generation_config", None)
        for source in (gen, cfg):
            value = getattr(source, "decoder_start_token_id", None)
            if isinstance(value, int):
                return value
        pad = getattr(cfg, "pad_token_id", None)
        return pad if isinstance(pad, int) else 0

    @property
    def accepts_labels(self) -> bool:
        """True if forward(input_ids=..., labels=...) returns a Hugging Face-style loss."""
        if self._forward_override is not None or not self.is_module:
            return False
        if self.is_hf:
            return True
        names, _ = _signature_params(self.module.forward)
        return "input_ids" in names and "labels" in names

    # ---- forward pass
    def _call_model(self, ids: torch.Tensor, mode: str):
        m = self.module
        if mode == "override":
            return self._forward_override(ids)
        if mode == "kw":
            return m(input_ids=ids, use_cache=False) if self.is_hf else m(input_ids=ids)
        if mode == "pos":
            return m(ids)
        if mode == "pos_targets":
            return m(ids, ids)
        if mode == "seq2seq":
            start = torch.full((ids.shape[0], 1), self._decoder_start_id(), dtype=torch.long, device=ids.device)
            return m(input_ids=ids, decoder_input_ids=start, use_cache=False)
        raise ValueError(mode)

    def _candidate_calls(self) -> List[str]:
        if self._forward_override is not None:
            return ["override"]
        if self.kind == "seq2seq":
            return ["seq2seq"]
        fn = self.module.forward if self.is_module and hasattr(self.module, "forward") else self.module
        names, has_kwargs = _signature_params(fn)
        calls = ["kw"] if "input_ids" in names else []
        calls.append("pos")
        if "input_ids" not in names and has_kwargs:
            calls.append("kw")
        return calls

    def _extract(self, out, ids: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Find the logits tensor in whatever the model returned and normalise it to (B, T, V)."""
        batch, seq = ids.shape
        found = None
        if torch.is_tensor(out):
            found = out
        elif hasattr(out, "logits") and torch.is_tensor(out.logits):
            found = out.logits
        elif isinstance(out, dict) and torch.is_tensor(out.get("logits")):
            found = out["logits"]
        else:
            items = list(out.values()) if isinstance(out, dict) else list(out) if isinstance(out, (tuple, list)) else []
            for item in items:
                if torch.is_tensor(item) and item.is_floating_point() and item.dim() in (2, 3) \
                        and item.shape[-1] > 1:
                    found = item
                    break
        if found is None:
            raise TypeError(f"could not find logits in the model output ({type(out).__name__})")
        t = found
        if t.dim() == 3:
            if t.shape[0] == batch and t.shape[1] == seq:
                return t, "bt"
            if t.shape[0] == seq and t.shape[1] == batch and seq != batch:
                return t.transpose(0, 1), "tb"
            if t.shape[0] == batch and t.shape[1] == 1:
                return t, "last"
        if t.dim() == 2:
            if t.shape[0] == batch * seq and seq > 1:
                return t.view(batch, seq, -1), "flat"
            if t.shape[0] == batch:
                return t[:, None, :], "last"
        raise TypeError(f"model output has shape {tuple(found.shape)}, which doesn't look like logits for "
                        f"an input of shape {tuple(ids.shape)}")

    def probe(self) -> None:
        """Work out how to call the model, once. Raises TypeError if it can't be used as a language model."""
        if self._call is not None:
            return
        self.kind  # noqa: B018 - decide the kind before probing
        vocab = self.tok.vocab_size if self.tok is not None else None
        length = 8
        if self.context_length:
            length = max(2, min(length, self.context_length))
        high = min(vocab or 50, 50)
        ids = torch.arange(1, length + 1, device=self.device).remainder(max(1, high - 1)).add(1).view(1, -1)
        errors = []
        for mode in self._candidate_calls():
            try:
                with torch.no_grad():
                    logits, layout = self._extract(self._call_model(ids, mode), ids)
            except Exception as exc:
                errors.append(f"{mode}: {type(exc).__name__}: {exc}")
                continue
            self._call, self._layout = mode, layout
            break
        if self._call is None:
            raise TypeError(
                f"{self.name} cannot generate text: calling it on a batch of token ids failed "
                f"({'; '.join(errors)}). llmreport needs a model that maps token ids to next-token "
                "scores. If yours needs a special call, pass forward=lambda ids: ... to analyze()."
            )
        if self._layout == "last" and self._forward_override is None:
            # nanoGPT-style models only return the last position unless targets are given.
            names, _ = _signature_params(self.module.forward)
            if any(n in names for n in _TARGET_PARAMS) and len(names) >= 2:
                try:
                    with torch.no_grad():
                        _, layout = self._extract(self._call_model(ids, "pos_targets"), ids)
                    if layout != "last":
                        self._call, self._layout = "pos_targets", layout
                except Exception:
                    pass
        self.vocab_out = int(logits.shape[-1])
        if self._kind == "causal" and not self.is_hf and self.is_module and self._forward_override is None:
            embeddings = [m for m in self.module.modules() if isinstance(m, nn.Embedding)]
            if embeddings:
                token = max(embeddings, key=lambda e: e.num_embeddings)
                # Output as wide as the hidden size (not the vocabulary): hidden states, no LM head.
                if self.vocab_out == token.embedding_dim != token.num_embeddings and \
                        (vocab is None or self.vocab_out < vocab):
                    self._kind = "encoder"

    @property
    def full_logits(self) -> bool:
        self.probe()
        return self._layout != "last"

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits of shape (B, T, V), or (B, 1, V) for models that only score the last position."""
        self.probe()
        out = self._call_model(ids, self._call)
        logits, _ = self._extract(out, ids)
        return logits

    def last_logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits(ids)[:, -1, :]

    @torch.no_grad()
    def token_nll(self, ids: torch.Tensor, start: int = 1) -> Tuple[float, int]:
        """Sum of negative log-likelihoods (natural log) of ``ids[0, start:]`` and how many tokens that is.

        Each token is scored from the tokens before it. ``start`` is at least 1 because
        the first token has nothing before it.
        """
        start = max(1, int(start))
        seq = ids.shape[1]
        if seq <= start:
            return 0.0, 0
        n = seq - start
        self.probe()
        if self.kind == "seq2seq":
            return self._seq2seq_nll(ids, start)
        if self.kind == "masked":
            return self._pseudo_nll(ids, start)
        if self.kind == "encoder":
            raise NotSupportedByModel(f"{self.name} has no language-model head, so it can't score text.")
        if self.accepts_labels:
            labels = ids.clone()
            labels[:, :start] = -100
            out = self.module(input_ids=ids, labels=labels, **({"use_cache": False} if self.is_hf else {}))
            loss = _find_loss(out)
            if loss is not None:
                return float(loss) * n, n
        if self.full_logits:
            logits = self.logits(ids)[0, start - 1:-1].float()
            logp = torch.log_softmax(logits, dim=-1)
            target = ids[0, start:].to(logp.device)
            return float(-logp.gather(1, target[:, None]).sum()), n
        total = 0.0
        for t in range(start, seq):
            logp = torch.log_softmax(self.last_logits(ids[:, :t])[0].float(), dim=-1)
            total -= float(logp[ids[0, t]])
        return total, n

    def _seq2seq_nll(self, ids: torch.Tensor, start: int) -> Tuple[float, int]:
        """Score ``ids[start:]`` as the decoder's output, with ``ids[:start]`` as the encoder input."""
        src, tgt = ids[:, :start], ids[:, start:]
        out = self.module(input_ids=src, labels=tgt, use_cache=False)
        n = tgt.shape[1]
        return float(out.loss) * n, n

    def _pseudo_nll(self, ids: torch.Tensor, start: int, chunk: int = 16) -> Tuple[float, int]:
        """Pseudo-log-likelihood (Salazar et al., 2020): mask each token in turn and score it.

        Special tokens such as [CLS] and [SEP] are not scored.
        """
        mask_id = getattr(getattr(self.tok, "raw", None), "mask_token_id", None) if self.tok is not None else None
        if mask_id is None:
            raise NotSupportedByModel("This masked language model's tokenizer has no mask token.")
        special = set(self.tok.special_tokens.values()) if self.tok is not None else set()
        positions = [t for t in range(start, ids.shape[1]) if int(ids[0, t]) not in special]
        total = 0.0
        for i in range(0, len(positions), chunk):
            part = positions[i:i + chunk]
            batch = ids.repeat(len(part), 1)
            for row, t in enumerate(part):
                batch[row, t] = mask_id
            logits = self.logits(batch).float()
            for row, t in enumerate(part):
                total -= float(torch.log_softmax(logits[row, t], dim=-1)[ids[0, t]])
        return total, len(positions)

    # ---- generation
    @torch.no_grad()
    def generate_ids(self, prompt_ids: Sequence[int], max_new_tokens: int, min_new_tokens: int = 0) -> List[int]:
        """Greedy-decode up to ``max_new_tokens`` tokens and return only the new ids."""
        if not self.can_generate:
            raise NotSupportedByModel(f"{self.name} is a {self.kind_label}; it doesn't generate text.")
        if not prompt_ids:
            start = self.tok.bos_id if self.tok is not None else None
            prompt_ids = [start if start is not None else (self.tok.eos_id if self.tok else 0) or 0]
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        ctx = self.context_length
        if ctx and ids.shape[1] + max_new_tokens > ctx:
            ids = ids[:, -max(1, ctx - max_new_tokens):]
        if self.uses_model_generate:
            kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=self.tok.pad_id if self.tok is not None else 0)
            if min_new_tokens:
                kwargs["min_new_tokens"] = min_new_tokens
            out = self.module.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
            if self.kind == "seq2seq":  # the decoder output doesn't repeat the prompt; drop the start token
                new = out[0].tolist()
                return new[1:] if new and new[0] == self._decoder_start_id() else new
            return out[0, ids.shape[1]:].tolist()

        self.probe()
        stops = self.tok.stop_ids if self.tok is not None else set()
        vocab = self.tok.vocab_size if self.tok is not None else None
        new: List[int] = []
        for step in range(max_new_tokens):
            window = ids if not ctx or ids.shape[1] <= ctx else ids[:, -ctx:]
            logits = self.last_logits(window)[0].float()
            if vocab and logits.shape[-1] > vocab:
                logits[vocab:] = float("-inf")  # padding rows the tokenizer can't decode
            if step < min_new_tokens:
                for s in stops:
                    if s < logits.shape[-1]:
                        logits[s] = float("-inf")
            nxt = int(torch.argmax(logits))
            if nxt in stops:
                break
            new.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return new


def as_model(model, tokenizer=None, **kwargs) -> LanguageModel:
    if isinstance(model, LanguageModel):
        if tokenizer is not None and model.tok is None:
            model.tok = as_tokenizer(tokenizer)
        return model
    return LanguageModel(model, tokenizer, **kwargs)
