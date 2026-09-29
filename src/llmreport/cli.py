"""Command-line interface.

    llmreport gpt2 --html report.html
    llmreport checkpoints/final.pt --model-class gpt_model:GPT --tokenizer tokenizer.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys

from . import __version__
from .analyzers import DEFAULT_CHECKS, list_checks


def load_class(spec: str):
    """'package.module:Class' or 'path/to/file.py:Class' -> the class."""
    if ":" not in spec:
        raise SystemExit(f"--model-class must look like module:Class or file.py:Class, got {spec!r}")
    target, _, name = spec.rpartition(":")
    if target.endswith(".py") or os.sep in target or "/" in target:
        path = os.path.abspath(target)
        sys.path.insert(0, os.path.dirname(path))  # so the file can import its neighbours
        module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module  # pickled configs refer to the module by name
        module_spec.loader.exec_module(module)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        module = importlib.import_module(target)
    try:
        return getattr(module, name)
    except AttributeError:
        raise SystemExit(f"{target} has no class named {name!r}") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmreport",
        description="Run behavior and performance checks on any PyTorch language model.",
    )
    parser.add_argument("model", nargs="?",
                        help="Hugging Face name, local folder, or checkpoint file (.pt/.pth/.bin/.ckpt/.safetensors)")
    parser.add_argument("--tokenizer", help="Tokenizer: tokenizer.json, SentencePiece .model, folder or Hub name")
    parser.add_argument("--model-class", help="Class to rebuild a weights-only checkpoint, e.g. gpt_model:GPT")
    parser.add_argument("--template", help="Prompt template for instruction-tuned models, e.g. "
                                           "'<|instruction|>{prompt}<|response|>'")
    parser.add_argument("--context-length", type=int, help="Context window, if it isn't detected")
    parser.add_argument("--stop-tokens", default="", help="Comma-separated extra tokens that end an answer")
    parser.add_argument("--checks", help=f"Comma-separated checks (default: {','.join(DEFAULT_CHECKS)})")
    parser.add_argument("--skip", default="", help="Comma-separated checks to leave out")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--html", help="Write an HTML report to this path")
    parser.add_argument("--markdown", help="Write a Markdown model card to this path")
    parser.add_argument("--json", help="Write raw results as JSON to this path")
    parser.add_argument("--list-checks", action="store_true", help="List available checks and exit")
    parser.add_argument("--version", action="version", version=f"llmreport {__version__}")
    args = parser.parse_args(argv)

    if args.list_checks:
        for name, desc in list_checks().items():
            print(f"{name:<14} {desc}")
        return 0
    if not args.model:
        parser.error("the model argument is required (or use --list-checks)")

    from .core import analyze

    split = lambda s: [c.strip() for c in s.split(",") if c.strip()]  # noqa: E731
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())  # checkpoints often pickle classes from the current folder
    report = analyze(
        args.model,
        args.tokenizer,
        model_class=load_class(args.model_class) if args.model_class else None,
        prompt_template=args.template,
        context_length=args.context_length,
        stop_tokens=split(args.stop_tokens),
        checks=split(args.checks) if args.checks else None,
        skip=split(args.skip),
        mode=args.mode,
        device=args.device,
        seed=args.seed,
    )
    print()
    report.show()
    for path, fn in ((args.html, report.to_html), (args.markdown, report.to_markdown), (args.json, report.to_json)):
        if path:
            fn(path)
            print(f"Wrote {path}")
    return 1 if any(r.status == "error" for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
