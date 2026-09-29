"""The ``analyze()`` entry point: load the model, run each check, build the report."""

from __future__ import annotations

import datetime as _dt
import difflib
import platform
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from . import __version__
from ._utils import set_seed
from .analyzers import DEFAULT_CHECKS, REGISTRY
from .analyzers.base import ERROR, AnalyzerResult, RunConfig
from .report import Report


def _pick_device(device: Optional[str]) -> str:
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
    model.to(_pick_device(device))
    return model, tokenizer


def _model_name(model) -> str:
    cfg = getattr(model, "config", None)
    return getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None) or type(model).__name__


def _as_list(value) -> List[str]:
    """Accept "perplexity", "a,b", ["a", "b"] or ("a",) and return a list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def _resolve_inputs(model, tokenizer, device, verbose):
    """Turn whatever the user passed into a (model, tokenizer) pair."""
    # llmreport.analyze(llmreport.load("gpt2")) or analyze((model, tokenizer))
    if isinstance(model, tuple) and len(model) == 2:
        model, tokenizer = model[0], tokenizer or model[1]

    # A transformers pipeline: pipeline("text-generation", model="gpt2")
    if not isinstance(model, (str, torch.nn.Module)) and hasattr(model, "model"):
        tokenizer = tokenizer or getattr(model, "tokenizer", None)
        model = model.model

    # A name on the Hugging Face Hub or a local folder
    if isinstance(model, str):
        if verbose:
            print(f"Loading {model} ...")
        model, loaded_tok = load(model, device or "auto")
        tokenizer = tokenizer or loaded_tok
    elif device is not None:
        model.to(_pick_device(device))

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Expected a Hugging Face model, a model name like 'gpt2', or a pipeline; got {type(model).__name__}."
        )
    if not hasattr(model, "generate"):
        raise TypeError(
            f"{type(model).__name__} cannot generate text. llmreport needs a causal language model, "
            "for example one loaded with AutoModelForCausalLM."
        )

    # Try to find the tokenizer from the model's own name or folder.
    if tokenizer is None:
        source = getattr(getattr(model, "config", None), "_name_or_path", "")
        if source:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(source)
                if verbose:
                    print(f"Loaded tokenizer from {source}")
            except Exception:
                tokenizer = None
    if tokenizer is None:
        raise ValueError(
            "Could not find a tokenizer for this model. Pass it explicitly: "
            "llmreport.analyze(model, tokenizer)"
        )
    return model, tokenizer


def analyze(
    model: Union[str, Any],
    tokenizer: Any = None,
    checks: Union[str, Sequence[str], None] = None,
    *,
    mode: str = "quick",
    skip: Union[str, Sequence[str]] = (),
    device: Optional[str] = None,
    seed: int = 0,
    max_new_tokens: Optional[int] = None,
    use_chat_template: Any = "auto",
    prompts: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Dict[str, Any]]] = None,
    name: Optional[str] = None,
    verbose: bool = True,
    raise_errors: bool = False,
) -> Report:
    """Run a set of checks on a language model and return a :class:`Report`.

    Args:
        model: Any of: a Hugging Face causal LM object, a model name or local folder
            (for example ``"gpt2"``), a ``(model, tokenizer)`` tuple, or a
            ``transformers`` text-generation pipeline.
        tokenizer: The matching tokenizer. Optional: if left out, llmreport loads it
            from the model's name or folder.
        checks: Which checks to run: ``"perplexity"``, ``"architecture,repetition"``
            or ``["architecture", "repetition"]``. Defaults to all built-in checks.
            See :func:`llmreport.list_checks`.
        mode: ``"quick"`` (small prompt sets, minutes on CPU) or ``"full"``.
        skip: Checks to leave out, for example ``skip="performance"``.
        device: Where to run. ``None`` leaves an already-loaded model where it is;
            ``"auto"`` picks CUDA, then Apple MPS, then CPU.
        seed: Random seed for reproducible results.
        max_new_tokens: Override how many tokens behavior checks generate.
        use_chat_template: ``"auto"``, ``True`` or ``False``.
        prompts: Replace built-in prompt sets, keyed by check name.
            ``{"repetition": [...], "consistency": [[q1, q2], ...], "perplexity": [...],
            "robustness": [...], "refusal": {"should_refuse": [...], "should_answer": [...]}}``
        options: Per-check settings, for example
            ``{"performance": {"prompt_lengths": [64, 256], "repeats": 2}}``.
        name: Display name for the model in the report.
        verbose: Print progress while running.
        raise_errors: Re-raise exceptions from checks instead of recording them in the report.

    Returns:
        A :class:`Report`. Display it in a notebook, ``print`` it, or export it with
        ``to_html``, ``to_markdown`` or ``to_json``.

    Example:
        >>> import llmreport
        >>> report = llmreport.analyze("gpt2")
        >>> report.to_html("gpt2_report.html")
    """
    model, tokenizer = _resolve_inputs(model, tokenizer, device, verbose)

    selected = _as_list(checks) if checks is not None else list(DEFAULT_CHECKS)
    skip = _as_list(skip)
    unknown = [c for c in selected + skip if c not in REGISTRY]
    if unknown:
        hint = ""
        close = [name for u in unknown for name in REGISTRY if u.lower() in name or name in u.lower()]
        close += [m for u in unknown for m in difflib.get_close_matches(u.lower(), list(REGISTRY), n=2, cutoff=0.6)]
        if close:
            hint = f" Did you mean {sorted(set(close))}?"
        raise ValueError(f"Unknown check(s) {unknown}. Available: {sorted(REGISTRY)}.{hint}")
    selected = [c for c in selected if c not in set(skip)]
    if mode not in ("quick", "full"):
        raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")

    config = RunConfig(
        mode=mode,
        seed=seed,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        prompts=prompts or {},
        options=options or {},
    )

    was_training = model.training
    model.eval()
    results: List[AnalyzerResult] = []
    started = time.perf_counter()
    try:
        for i, check in enumerate(selected, 1):
            analyzer = REGISTRY[check]()
            if verbose:
                print(f"[{i}/{len(selected)}] {analyzer.title} ...", end=" ", flush=True)
            set_seed(seed)
            t0 = time.perf_counter()
            try:
                result = analyzer.run(model, tokenizer, config)
            except Exception as exc:  # keep going; one broken check should not sink the report
                if raise_errors:
                    raise
                result = AnalyzerResult(
                    name=analyzer.name,
                    title=analyzer.title,
                    summary=f"This check failed: {type(exc).__name__}: {exc}",
                    status=ERROR,
                    details={"traceback": traceback.format_exc()},
                )
            result.duration_s = round(time.perf_counter() - t0, 3)
            results.append(result)
            if verbose:
                print(f"{result.status} ({result.duration_s:.1f}s)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model.train(was_training)

    import transformers

    metadata = {
        "model": name or _model_name(model),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "mode": mode,
        "seed": seed,
        "checks": selected,
        "duration_s": round(time.perf_counter() - started, 2),
        "device": str(next(model.parameters()).device),
        "llmreport_version": __version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
    }
    return Report(results=results, metadata=metadata)
