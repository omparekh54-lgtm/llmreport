"""Command-line interface: ``llmreport gpt2 --html report.html``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .analyzers import DEFAULT_CHECKS, list_checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmreport",
        description="Run behavior and performance checks on a Hugging Face language model.",
    )
    parser.add_argument("model", nargs="?", help="Model name on the Hugging Face Hub or a local path")
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
    report = analyze(
        args.model,
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
