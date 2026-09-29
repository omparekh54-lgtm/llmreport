"""Built-in analyzers and the registry that holds them."""

from __future__ import annotations

from typing import Dict, List, Type

from .architecture import ArchitectureAnalyzer
from .base import Analyzer, AnalyzerResult, RunConfig
from .consistency import ConsistencyAnalyzer
from .performance import PerformanceAnalyzer
from .perplexity import PerplexityAnalyzer
from .refusal import RefusalAnalyzer
from .repetition import RepetitionAnalyzer
from .robustness import RobustnessAnalyzer

REGISTRY: Dict[str, Type[Analyzer]] = {}

# Order here is the order sections appear in the report.
DEFAULT_CHECKS: List[str] = [
    "architecture",
    "performance",
    "perplexity",
    "repetition",
    "consistency",
    "robustness",
    "refusal",
]


def register_analyzer(cls: Type[Analyzer]) -> Type[Analyzer]:
    """Register a custom analyzer so ``analyze(checks=[...])`` can use it.

    Can be used as a decorator::

        @llmreport.register_analyzer
        class MyCheck(llmreport.Analyzer):
            name = "my_check"
            title = "My check"
            def run(self, model, tokenizer, config):
                return self.result("Everything looks fine.", metrics={"score": 1.0})
    """
    if not (isinstance(cls, type) and issubclass(cls, Analyzer)):
        raise TypeError("register_analyzer expects a subclass of llmreport.Analyzer")
    if not cls.name or cls.name == "base":
        raise ValueError("Analyzer subclasses must set a unique `name`")
    REGISTRY[cls.name] = cls
    return cls


for _cls in (
    ArchitectureAnalyzer,
    PerformanceAnalyzer,
    PerplexityAnalyzer,
    RepetitionAnalyzer,
    ConsistencyAnalyzer,
    RobustnessAnalyzer,
    RefusalAnalyzer,
):
    register_analyzer(_cls)


def list_checks() -> Dict[str, str]:
    """Return ``{name: description}`` for every registered check."""
    return {name: cls.description for name, cls in REGISTRY.items()}


__all__ = [
    "Analyzer",
    "AnalyzerResult",
    "RunConfig",
    "REGISTRY",
    "DEFAULT_CHECKS",
    "register_analyzer",
    "list_checks",
]
