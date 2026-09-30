"""Command-line interface.

    llmreport gpt2 --html report.html
    llmreport checkpoints/final.pt --model-class gpt_model:GPT --tokenizer tokenizer.json
    llmreport info checkpoints/final.pt --model-class gpt_model:GPT --tokenizer tokenizer.json
    llmreport health checkpoints/final.pt --model-class gpt_model:GPT
    llmreport watch runs/pretrain                 # follow a training run live
    llmreport dashboard runs/pretrain
    llmreport progress checkpoints/pretrain --model-class gpt_model:GPT --tokenizer tokenizer.json
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


_SUBCOMMANDS = ("watch", "dashboard", "info", "health", "progress")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Hugging Face name, local folder, or checkpoint file")
    p.add_argument("--tokenizer", help="Tokenizer: tokenizer.json, SentencePiece .model, folder or Hub name")
    p.add_argument("--model-class", help="Class to rebuild a weights-only checkpoint, e.g. gpt_model:GPT")
    p.add_argument("--device", default="auto")


def _load_model(args):
    from .loading import CHECKPOINT_SUFFIXES, load, load_checkpoint

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    cls = load_class(args.model_class) if args.model_class else None
    if os.path.isfile(args.model) and (cls is not None or args.model.endswith(CHECKPOINT_SUFFIXES)):
        return load_checkpoint(args.model, cls, tokenizer=args.tokenizer, device=args.device)
    model, tok = load(args.model, args.device)
    return model, args.tokenizer or tok


def _watch(log_dir: str, interval: float, once: bool) -> int:
    """Follow a run from another terminal: progress lines, evaluations and alerts as they happen."""
    import time

    from .tracking import Run, _fmt_duration

    seen_rows, seen_events, last_print = 0, 0, 0.0
    print(f"Watching {log_dir} (Ctrl+C to stop)")
    try:
        while True:
            try:
                run = Run(log_dir)
            except FileNotFoundError:
                if once:
                    print("No run found yet.")
                    return 1
                time.sleep(interval)
                continue
            for e in run.events[seen_events:]:
                kind = e.get("kind")
                if kind == "alert":
                    print(f"[{e.get('severity', 'info')}] step {e.get('step', 0):,}: {e.get('message', '')}")
                elif kind == "eval":
                    vals = ", ".join(f"{k} {v:.4g}" for k, v in e.items()
                                     if isinstance(v, (int, float)) and k not in ("step", "time", "eval_seconds"))
                    print(f"[eval] step {e.get('step', 0):,}: {vals}")
                elif kind in ("start", "resume", "end"):
                    print(f"[{kind}] {e.get('message', '')}")
            seen_events = len(run.events)
            if len(run.rows) > seen_rows and (time.time() - last_print >= interval or once):
                r = run.rows[-1]
                s = run.summary()
                parts = [f"step {r['step']:,}" + (f"/{s['total_steps']:,}" if s.get("total_steps") else "")]
                if r.get("loss") is not None:
                    parts.append(f"loss {r['loss']:.4f}" + (f" (avg {r['loss_avg']:.4f})" if r.get("loss_avg") else ""))
                if r.get("lr") is not None:
                    parts.append(f"lr {r['lr']:.2e}")
                if r.get("grad_norm") is not None:
                    parts.append(f"grad {r['grad_norm']:.3g}")
                if s.get("tokens_per_s"):
                    parts.append(f"{s['tokens_per_s']:,.0f} tok/s")
                if s.get("elapsed_s"):
                    parts.append(f"elapsed {_fmt_duration(s['elapsed_s'])}")
                print(" | ".join(parts), flush=True)
                seen_rows, last_print = len(run.rows), time.time()
            if once or any(e.get("kind") == "end" for e in run.events[-2:]):
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def _subcommand(argv) -> int:
    parser = argparse.ArgumentParser(prog="llmreport")
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("watch", help="Follow a training run live from another terminal")
    w.add_argument("log_dir")
    w.add_argument("--interval", type=float, default=5.0)
    w.add_argument("--once", action="store_true", help="Print the current state and exit")
    d = sub.add_parser("dashboard", help="Write the HTML dashboard for a run")
    d.add_argument("log_dir")
    d.add_argument("--out", help="Where to write it (default: <log_dir>/dashboard.html)")
    i = sub.add_parser("info", help="Everything about a model: structure, size, memory, health")
    _model_args(i)
    h = sub.add_parser("health", help="Check a model's weights for NaNs, dead layers and other problems")
    _model_args(h)
    pr = sub.add_parser("progress", help="Evaluate a folder of training checkpoints and chart the progress")
    pr.add_argument("folder")
    pr.add_argument("--tokenizer", required=True)
    pr.add_argument("--model-class")
    pr.add_argument("--texts", help="Text file with validation text (blank lines separate samples)")
    pr.add_argument("--prompts", default="", help="Comma-separated prompts to generate from at each checkpoint")
    pr.add_argument("--template")
    pr.add_argument("--code", action="store_true", help="Also run the Python code check")
    pr.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.command == "watch":
        return _watch(args.log_dir, args.interval, args.once)
    if args.command == "dashboard":
        from .tracking import load_run

        path = load_run(args.log_dir).write_dashboard(args.out)
        print(f"Wrote {path}")
        return 0
    if args.command in ("info", "health"):
        from .functions import info
        from .healthcheck import health

        model, tok = _load_model(args)
        result = info(model, tok) if args.command == "info" else health(model)
        print(result)
        return 0 if args.command == "info" or result.status != "critical" else 1
    if args.command == "progress":
        from .tracking import track_checkpoints

        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        texts = None
        if args.texts:
            with open(args.texts, encoding="utf-8") as f:
                texts = [t.strip() for t in f.read().split("\n\n") if t.strip()]
        run = track_checkpoints(args.folder, load_class(args.model_class) if args.model_class else None,
                                args.tokenizer, texts=texts, code=args.code, prompt_template=args.template,
                                sample_prompts=[p for p in args.prompts.split(",") if p.strip()] or None,
                                device=args.device)
        print(f"Wrote {os.path.join(run.log_dir, 'dashboard.html')}")
        return 0
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _SUBCOMMANDS:
        return _subcommand(argv)
    parser = argparse.ArgumentParser(
        prog="llmreport",
        description="Run behavior and performance checks on any PyTorch language model. "
                    "Other commands: watch, dashboard, info, health, progress (see llmreport <command> -h).",
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
