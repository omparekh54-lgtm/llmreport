"""Base classes shared by every analyzer.

An analyzer is a small class with a ``name`` and a ``run`` method. It receives
the model, the tokenizer and a :class:`RunConfig`, and returns an
:class:`AnalyzerResult`. Adding a new check to llmreport means writing one of
these and registering it with :func:`llmreport.register_analyzer`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Allowed values for AnalyzerResult.status
OK = "ok"
WARNING = "warning"
INFO = "info"
SKIPPED = "skipped"
ERROR = "error"
STATUSES = (OK, WARNING, INFO, SKIPPED, ERROR)


@dataclass
class AnalyzerResult:
    """The output of one analyzer.

    Attributes:
        name: Machine name of the analyzer, for example ``"perplexity"``.
        title: Human-readable section title, for example ``"Perplexity"``.
        summary: One plain-English sentence describing the finding.
        metrics: Flat mapping of metric name to a number or short string.
            Numeric metrics are what :meth:`Report.compare` diffs.
        details: Anything extra (tables, sample generations) for the full report.
        status: One of ``"ok"``, ``"warning"``, ``"info"``, ``"skipped"``, ``"error"``.
        notes: Caveats about how the metric was computed.
        duration_s: Wall-clock time the analyzer took, filled in by the runner.
    """

    name: str
    title: str
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = OK
    notes: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Settings passed to every analyzer.

    Attributes:
        mode: ``"quick"`` (a few minutes on CPU for small models) or ``"full"``.
        seed: Random seed so results are reproducible.
        max_new_tokens: Upper bound on tokens generated per prompt.
        use_chat_template: ``"auto"`` uses the tokenizer's chat template when it
            has one, ``True`` forces it, ``False`` never uses it.
        prompts: Optional overrides for built-in prompt sets, keyed by analyzer name.
        options: Optional per-analyzer keyword options, keyed by analyzer name.
    """

    mode: str = "quick"
    seed: int = 0
    max_new_tokens: Optional[int] = None
    use_chat_template: Any = "auto"
    prompts: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("quick", "full"):
            raise ValueError("mode must be 'quick' or 'full'")

    @property
    def quick(self) -> bool:
        return self.mode == "quick"

    def gen_tokens(self, default_quick: int, default_full: int) -> int:
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return default_quick if self.quick else default_full

    def sample(self, items: list, n_quick: int) -> list:
        """Use a subset of a prompt set in quick mode, all of it in full mode."""
        return list(items[:n_quick]) if self.quick else list(items)

    def option(self, analyzer: str, key: str, default: Any = None) -> Any:
        return self.options.get(analyzer, {}).get(key, default)


class Analyzer:
    """Base class for all checks.

    Subclasses set ``name`` and ``title`` and implement :meth:`run`.
    Set ``needs_generation = True`` if the analyzer calls ``model.generate``,
    so users can tell which checks are slower.
    """

    name: str = "base"
    title: str = "Base analyzer"
    description: str = ""
    needs_generation: bool = False

    def run(self, model, tokenizer, config: RunConfig) -> AnalyzerResult:  # pragma: no cover
        raise NotImplementedError

    def result(self, summary: str, **kwargs: Any) -> AnalyzerResult:
        """Shortcut that fills in ``name`` and ``title``."""
        return AnalyzerResult(name=self.name, title=self.title, summary=summary, **kwargs)
