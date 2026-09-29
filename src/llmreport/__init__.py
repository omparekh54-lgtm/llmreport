"""llmreport: a one-line behavior and performance report for your language model.

    >>> import llmreport
    >>> report = llmreport.analyze(model, tokenizer)   # or llmreport.analyze("gpt2")
    >>> report                                          # renders in a notebook
    >>> report.to_html("report.html")
"""

__version__ = "0.1.0"

from .analyzers import (  # noqa: E402
    DEFAULT_CHECKS,
    Analyzer,
    AnalyzerResult,
    RunConfig,
    list_checks,
    register_analyzer,
)
from .core import analyze, load  # noqa: E402
from .report import Comparison, Report  # noqa: E402

__all__ = [
    "__version__",
    "analyze",
    "load",
    "Report",
    "Comparison",
    "Analyzer",
    "AnalyzerResult",
    "RunConfig",
    "register_analyzer",
    "list_checks",
    "DEFAULT_CHECKS",
]
