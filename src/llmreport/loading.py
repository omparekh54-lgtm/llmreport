"""Loading models: Hugging Face names and folders, plain PyTorch checkpoints and TorchScript files."""

from __future__ import annotations

import dataclasses
import inspect
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

_STATE_KEYS = ("model", "state_dict", "model_state_dict", "model_state", "module", "net", "weights",
               "ema", "model_ema")
_CONFIG_KEYS = ("config", "model_config", "model_args", "args", "hparams", "hyper_parameters", "cfg", "params")
_PREFIXES = ("_orig_mod.", "module.", "model.")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".tar", ".safetensors")


def pick_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name_or_path: str, device: Optional[str] = "auto", **model_kwargs):
    """Load a Hugging Face causal LM and its tokenizer by name or local path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)
    model.to(pick_device(device))
    return model, tokenizer


def _is_state_dict(obj) -> bool:
    """A mapping of parameter names to tensors. Quantized layers also store tuples and dtypes."""
    if not isinstance(obj, dict) or not obj or not all(isinstance(k, str) for k in obj):
        return False
    values = list(obj.values())
    return any(torch.is_tensor(v) for v in values) and \
        all(torch.is_tensor(v) or isinstance(v, (tuple, torch.dtype)) for v in values)


def _find_state_dict(obj) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[nn.Module]]:
    if isinstance(obj, nn.Module):
        return None, obj
    if _is_state_dict(obj):
        return obj, None
    if isinstance(obj, dict):
        for key in _STATE_KEYS:
            value = obj.get(key)
            if isinstance(value, nn.Module):
                return None, value
            if _is_state_dict(value):
                return value, None
        # Lightning and friends: {"state_dict": {...}} with nested dicts of tensors
        for value in obj.values():
            if _is_state_dict(value) and len(value) > 1:
                return value, None
    return None, None


def _strip_prefixes(state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Remove wrapper prefixes ("module.", "_orig_mod.", ...) so the keys match the model.

    Keeps the state dict's ``_metadata`` (layer format versions), which quantized layers need.
    """
    from collections import OrderedDict

    expected = set(model.state_dict().keys())
    metadata = getattr(state, "_metadata", None)

    def rename(fn):
        out = OrderedDict((fn(k), v) for k, v in state.items())
        if metadata is not None:
            out._metadata = OrderedDict((fn(k + ".")[:-1] if k else k, v) for k, v in metadata.items())
        return out

    for prefix in _PREFIXES:
        if any(k in expected for k in state):
            break
        if all(k.startswith(prefix) for k in state):
            state = rename(lambda k, p=prefix: k[len(p):] if k.startswith(p) else k)
            metadata = getattr(state, "_metadata", None)
    # torch.compile adds "_orig_mod." in the middle too
    if not any(k in expected for k in state):
        state = rename(lambda k: k.replace("._orig_mod.", ".").replace("_orig_mod.", ""))
    return state


def _build(model_class, config, model_kwargs):
    """Instantiate ``model_class`` from whatever config the checkpoint (or the user) provided."""
    if config is None:
        return model_class(**model_kwargs)
    if not isinstance(config, dict):
        return model_class(config, **model_kwargs)
    errors = []
    try:
        return model_class(**config, **model_kwargs)
    except TypeError as exc:
        errors.append(str(exc))
    # model_class(Config(**dict)): find the config class from the type annotation of the first argument
    try:
        params = [p for p in inspect.signature(model_class).parameters.values()]
        ann = params[0].annotation if params else inspect.Parameter.empty
        if isinstance(ann, str):
            ann = inspect.getmodule(model_class).__dict__.get(ann, inspect.Parameter.empty)
        if isinstance(ann, type):
            if dataclasses.is_dataclass(ann):
                names = {f.name for f in dataclasses.fields(ann)}
                cfg = ann(**{k: v for k, v in config.items() if k in names})
            else:
                cfg = ann(**config)
            return model_class(cfg, **model_kwargs)
    except Exception as exc:
        errors.append(str(exc))
    try:
        return model_class(config, **model_kwargs)
    except Exception as exc:
        errors.append(str(exc))
    raise TypeError(f"Couldn't build {model_class.__name__} from the checkpoint's config {config!r}: "
                    + " | ".join(errors) + ". Pass config=... or model_kwargs to load_checkpoint().")


def load_checkpoint(
    path: str,
    model_class=None,
    *,
    tokenizer: Any = None,
    config: Any = None,
    device: Optional[str] = "auto",
    strict: bool = True,
    trust_checkpoint: bool = True,
    **model_kwargs,
):
    """Load a language model saved with plain PyTorch and return ``(model, tokenizer)``.

    Handles the common ways people save models:

    * ``torch.save(model)``: the whole model object.
    * ``torch.save(model.state_dict())`` or ``torch.save({"model": state_dict, "config": cfg, ...})``:
      weights (plus optional config) that ``model_class`` is built from.
    * TorchScript files saved with ``torch.jit.save``.
    * Dynamically quantized INT8 weights (``torch.ao.quantization.quantize_dynamic``).
    * Hugging Face folders (forwarded to :func:`load`).

    Args:
        path: Checkpoint file or Hugging Face folder.
        model_class: The model's class, for example ``GPT`` from your ``gpt_model.py``. Needed
            when the file only holds weights.
        tokenizer: Tokenizer object or path (tokenizer.json, SentencePiece .model, folder).
        config: Config to build ``model_class`` with. Defaults to the one stored in the
            checkpoint (under ``"config"``, ``"model_args"``, ``"hparams"``, ...).
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` ...
        strict: Require the weights to match the model exactly.
        trust_checkpoint: Checkpoints that store Python objects (such as a config class) can only be
            read with pickle, which can run code. Only load files you trust. Set to False to
            refuse anything but plain tensors.
        **model_kwargs: Extra keyword arguments for ``model_class``.

    Example:
        >>> from gpt_model import GPT
        >>> model, tok = llmreport.load_checkpoint("checkpoints/final.pt", GPT, tokenizer="tokenizer.json")
        >>> llmreport.analyze(model, tok)
    """
    from .adapters import as_tokenizer

    path = os.fspath(path)
    if os.path.isdir(path):
        model, tok = load(path, device)
        return model, tokenizer if tokenizer is not None else tok
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    model = None
    state = None
    stored_config = None
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
        except RuntimeError as exc:
            if "TorchScript" in str(exc) or "constants.pkl" in str(exc) or "jit" in str(exc).lower():
                obj = torch.jit.load(path, map_location="cpu")
            else:
                raise
        state, model = _find_state_dict(obj)
        if isinstance(obj, dict):
            for key in _CONFIG_KEYS:
                if obj.get(key) is not None:
                    stored_config = obj[key]
                    break
    if model is None:
        if state is None:
            raise ValueError(f"Couldn't find model weights in {path}. Expected a model object, a state_dict, "
                             f"or a dict with one of the keys {list(_STATE_KEYS)}.")
        if model_class is None:
            raise ValueError(f"{os.path.basename(path)} only contains weights, so llmreport needs the model's "
                             "class to rebuild it: load_checkpoint(path, model_class=YourModel).")
        cfg = config if config is not None else stored_config
        if type(cfg).__name__ == "Namespace":  # argparse arguments
            cfg = vars(cfg)
        model = _build(model_class, cfg, model_kwargs)
        if any("_packed_params" in k for k in state):
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        state = _strip_prefixes(state, model)
        result = model.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith(("attn.bias", "masked_bias", ".mask"))]
        if strict and (missing or result.unexpected_keys):
            raise ValueError(
                f"The checkpoint doesn't match {model_class.__name__}: missing {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}, unexpected {result.unexpected_keys[:8]}"
                f"{' ...' if len(result.unexpected_keys) > 8 else ''}. Check model_class and config, "
                "or pass strict=False."
            )
    quantized = any(hasattr(m, "_packed_params") for m in model.modules()) if isinstance(model, nn.Module) else False
    target = "cpu" if quantized else pick_device(device)  # dynamic quantization runs on CPU only
    if hasattr(model, "to"):
        model = model.to(target)
    if hasattr(model, "eval"):
        model.eval()
    return model, as_tokenizer(tokenizer).raw if tokenizer is not None else None
