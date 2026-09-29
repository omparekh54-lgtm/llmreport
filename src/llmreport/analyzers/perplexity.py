"""How well the model predicts ordinary text."""

from __future__ import annotations

import math

from .._utils import load_data, perplexity
from .base import OK, WARNING, Analyzer, RunConfig


def _texts(items) -> list:
    return [t["text"] if isinstance(t, dict) else str(t) for t in items]


def perplexity_texts(config: RunConfig) -> list:
    if "perplexity" in config.prompts:
        return _texts(config.prompts["perplexity"])
    return _texts(load_data("perplexity")["texts"])


class PerplexityAnalyzer(Analyzer):
    name = "perplexity"
    title = "Perplexity"
    description = "How surprised the model is by ordinary English text (lower is better)."

    def run(self, model, tokenizer, config: RunConfig):
        texts = perplexity_texts(config)
        res = perplexity(model, tokenizer, texts)
        ppl = res["perplexity"]
        domains = []
        if "perplexity" not in config.prompts:
            domains = [t["domain"] for t in load_data("perplexity")["texts"]]

        if math.isnan(ppl):
            return self.result("Could not compute perplexity (texts were too short).", status=WARNING)

        what = "your texts" if "perplexity" in config.prompts else "everyday English"
        if ppl < 20:
            verdict, status = f"fluent on {what}", OK
        elif ppl < 60:
            verdict, status = f"reasonably fluent on {what}", OK
        elif ppl < 200:
            verdict, status = f"struggles with {what}", WARNING
        else:
            verdict, status = (f"barely predicts {what} (undertrained, damaged, or trained on a different "
                               "language or domain)"), WARNING

        per_text = [
            {"domain": domains[i] if i < len(domains) else f"text {i + 1}", "perplexity": round(p, 2)}
            for i, p in enumerate(res["per_text"])
        ]
        hardest = max(per_text, key=lambda r: r["perplexity"]) if per_text else None

        summary = f"Perplexity {ppl:,.2f} over {res['tokens']:,} tokens: {verdict}."
        notes = [
            "Perplexity depends on the tokenizer, so only compare it between models that share one.",
            "Bits per character does not depend on the tokenizer and is fairer across models.",
        ]
        if hardest and len(per_text) > 1:
            notes.append(f"Hardest text: {hardest['domain']} (perplexity {hardest['perplexity']:,.2f}).")

        return self.result(
            summary,
            metrics={
                "perplexity": round(ppl, 3),
                "bits_per_char": round(res["bits_per_char"], 4),
                "tokens_scored": res["tokens"],
            },
            details={"per_text": per_text},
            status=status,
            notes=notes,
        )
