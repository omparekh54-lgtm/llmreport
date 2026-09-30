"""The ``analyze()`` entry point: load the model, run each check, build the report."""

from __future__ import annotations

import datetime as _dt
import difflib
import os
import platform
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from . import __version__
from ._utils import set_seed
from .adapters import LanguageModel, NotSupportedByModel, TokenizerAdapter
from .analyzers import DEFAULT_CHECKS, REGISTRY
from .analyzers.base import ERROR, SKIPPED, AnalyzerResult, RunConfig
from .loading import CHECKPOINT_SUFFIXES, load, load_checkpoint, pick_device
from .report import Report


def _pick_device(device: Optional[str]) -> str:
    return pick_device(device)


def _skip_reason(analyzer, lm) -> Optional[AnalyzerResult]:
    """A 'skipped' result when the check can't apply to this kind of model, otherwise None."""
    if getattr(analyzer, "handles_any_model", False):
        return None
    if analyzer.needs_generation and not lm.can_generate:
        why = f"{lm.name} is a {lm.kind_label}; it doesn't generate text, so this check doesn't apply."
    elif analyzer.name == "perplexity" and not lm.can_score:
        why = f"{lm.name} is a {lm.kind_label}, so it can't score text."
    else:
        return None
    return AnalyzerResult(name=analyzer.name, title=analyzer.title, summary=why, status=SKIPPED)


def _as_list(value) -> List[str]:
    """Accept "perplexity", "a,b", ["a", "b"] or ("a",) and return a list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def _resolve_inputs(model, tokenizer, device, verbose, model_class=None):
    """Turn whatever the user passed into a (model, tokenizer) pair."""
    # llmreport.analyze(llmreport.load("gpt2")) or analyze((model, tokenizer))
    if isinstance(model, tuple) and len(model) == 2:
        model, tokenizer = model[0], tokenizer if tokenizer is not None else model[1]

    # A transformers pipeline: pipeline("text-generation", model="gpt2")
    if not isinstance(model, (str, os.PathLike, torch.nn.Module)) and hasattr(model, "model") \
            and hasattr(model, "tokenizer"):
        tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        model = model.model

    # A checkpoint file, a local folder or a name on the Hugging Face Hub
    if isinstance(model, (str, os.PathLike)):
        path = os.fspath(model)
        if verbose:
            print(f"Loading {path} ...")
        is_checkpoint = os.path.isfile(path) and (model_class is not None or path.endswith(CHECKPOINT_SUFFIXES))
        if is_checkpoint:
            model, tokenizer = load_checkpoint(path, model_class, tokenizer=tokenizer, device=device or "auto")
        else:
            model, loaded_tok = load(path, device or "auto")
            tokenizer = tokenizer if tokenizer is not None else loaded_tok
    elif device is not None and hasattr(model, "to"):
        model.to(_pick_device(device))

    if not isinstance(model, torch.nn.Module) and not callable(model):
        raise TypeError(
            f"Expected a PyTorch language model, a model name like 'gpt2', a checkpoint path or a pipeline; "
            f"got {type(model).__name__}."
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
            "Could not find a tokenizer for this model. Pass it explicitly, for example "
            "llmreport.analyze(model, tokenizer) or llmreport.analyze(model, \"tokenizer.json\")."
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
    prompt_template: Optional[str] = None,
    model_class: Any = None,
    context_length: Optional[int] = None,
    forward: Optional[Callable] = None,
    generation: str = "auto",
    stop_tokens: Union[str, Sequence[str]] = (),
    prompts: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Dict[str, Any]]] = None,
    name: Optional[str] = None,
    verbose: bool = True,
    raise_errors: bool = False,
) -> Report:
    """Run a set of checks on a language model and return a :class:`Report`.

    Works with any PyTorch language model: Hugging Face models, models written from
    scratch (nanoGPT-style GPTs, Llama-style models with RoPE, recurrent models ...),
    TorchScript and dynamically quantized models. The only requirement is that calling
    the model on a batch of token ids returns next-token scores (logits).

    Args:
        model: Any of: a PyTorch model object (Hugging Face or your own ``nn.Module``), a
            model name or local folder (for example ``"gpt2"``), a checkpoint file
            (``.pt``/``.pth``/``.bin``/``.ckpt``/``.safetensors``, see ``model_class``), a
            ``(model, tokenizer)`` tuple, or a ``transformers`` text-generation pipeline.
        tokenizer: The matching tokenizer: a Hugging Face tokenizer, a ``tokenizers.Tokenizer``,
            a SentencePiece or tiktoken tokenizer, any object with ``encode``/``decode``, or a
            path to ``tokenizer.json``, a SentencePiece ``.model`` file or a tokenizer folder.
            Optional for Hugging Face models (loaded from the model's name or folder).
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
        prompt_template: How to wrap a user message for an instruction-tuned model without a
            chat template, for example ``"<|instruction|>{prompt}<|response|>"``.
        model_class: Your model's class, needed when ``model`` is a checkpoint that only holds
            weights (for example ``GPT`` from ``gpt_model.py``).
        context_length: The model's context window. Detected automatically; set it if the
            report says it couldn't be found.
        forward: Function ``ids -> logits`` for models that need a special call, for example
            ``lambda ids: model(ids, start_pos=0)``.
        generation: ``"auto"`` (the model's own Hugging Face-style ``generate`` if it has one,
            otherwise llmreport's greedy loop), ``"builtin"`` or ``"model"``.
        stop_tokens: Extra tokens that end a generated answer, for example ``"<|end|>"``.
            End-of-text tokens are found automatically.
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
    model, tokenizer = _resolve_inputs(model, tokenizer, device, verbose, model_class)
    tok = TokenizerAdapter(tokenizer, [stop_tokens] if isinstance(stop_tokens, str) else stop_tokens)
    lm = LanguageModel(model, tok, context_length=context_length, forward=forward, generation=generation)
    lm.probe()  # fail early, with a clear message, if the model can't be called on token ids

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
        prompt_template=prompt_template,
        prompts=prompts or {},
        options=options or {},
    )

    was_training = lm.training
    lm.eval()
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
                result = _skip_reason(analyzer, lm) or analyzer.run(lm, tok, config)
            except NotSupportedByModel as exc:
                result = AnalyzerResult(name=analyzer.name, title=analyzer.title, summary=str(exc), status=SKIPPED)
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
        lm.train(was_training)

    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:  # pragma: no cover
        transformers_version = None
    metadata = {
        "model": name or lm.name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "mode": mode,
        "seed": seed,
        "checks": selected,
        "duration_s": round(time.perf_counter() - started, 2),
        "device": str(lm.device),
        "model_class": type(lm.module).__name__,
        "transformers_model": lm.is_hf,
        "model_kind": lm.kind,
        "tokenizer_class": tok.name,
        "generation": "model.generate" if lm.uses_model_generate else "llmreport greedy loop",
        "llmreport_version": __version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
    }
    if prompt_template:
        metadata["prompt_template"] = prompt_template
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return Report(results=results, metadata=metadata)
