"""llmreport: reports, quick functions and live training tracking for any language model.

    >>> import llmreport
    >>> report = llmreport.analyze(model, tokenizer)   # any PyTorch LM, or llmreport.analyze("gpt2")
    >>> report                                          # renders in a notebook
    >>> report.to_html("report.html")
"""

__version__ = "0.3.1"

from .adapters import LanguageModel, TokenizerAdapter  # noqa: E402
from .analyzers import (  # noqa: E402
    DEFAULT_CHECKS,
    Analyzer,
    AnalyzerResult,
    RunConfig,
    list_checks,
    register_analyzer,
)
from .core import analyze  # noqa: E402
from .functions import (  # noqa: E402
    ModelInfo,
    architecture,
    check,
    code_score,
    compare,
    generate,
    grad_norm,
    info,
    memory,
    param_breakdown,
    params,
    perplexity,
    predict_next,
    speed,
    text_stats,
    weight_norm,
)
from .healthcheck import HealthReport, health  # noqa: E402
from .loading import load, load_checkpoint  # noqa: E402
from .report import Comparison, Report  # noqa: E402
from .tracking import Run, Tracker, load_run, track_checkpoints  # noqa: E402

__all__ = [
    "__version__",
    "analyze",
    "load",
    "load_checkpoint",
    "LanguageModel",
    "params",
    "param_breakdown",
    "memory",
    "architecture",
    "info",
    "ModelInfo",
    "health",
    "HealthReport",
    "grad_norm",
    "weight_norm",
    "perplexity",
    "generate",
    "predict_next",
    "speed",
    "check",
    "code_score",
    "text_stats",
    "compare",
    "Tracker",
    "Run",
    "load_run",
    "track_checkpoints",
    "TokenizerAdapter",
    "Report",
    "Comparison",
    "Analyzer",
    "AnalyzerResult",
    "RunConfig",
    "register_analyzer",
    "list_checks",
    "DEFAULT_CHECKS",
]
