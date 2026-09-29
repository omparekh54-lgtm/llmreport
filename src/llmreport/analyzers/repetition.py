"""Does the model get stuck in loops or keep repeating phrases?"""

from __future__ import annotations

from .._utils import distinct_n, generate, load_data, mean, repeated_ngram_rate, words
from ..adapters import as_model, as_tokenizer
from .base import OK, WARNING, Analyzer, RunConfig


class RepetitionAnalyzer(Analyzer):
    name = "repetition"
    title = "Repetition"
    description = "Measures how often generated text repeats words and phrases."
    needs_generation = True

    def run(self, model, tokenizer, config: RunConfig):
        tokenizer = as_tokenizer(tokenizer)
        model = as_model(model, tokenizer)
        prompts = config.prompts.get(self.name) or load_data("repetition")["prompts"]
        prompts = config.sample(prompts, 5)
        max_new = config.gen_tokens(60, 120)

        rows = []
        for prompt in prompts:
            text = generate(model, tokenizer, prompt, max_new, config)
            toks = words(text)
            rows.append(
                {
                    "prompt": prompt,
                    "output": text,
                    "distinct_1": round(distinct_n(toks, 1), 3),
                    "distinct_2": round(distinct_n(toks, 2), 3),
                    "repeated_3gram_rate": round(repeated_ngram_rate(toks, 3), 3),
                }
            )

        rep = mean(r["repeated_3gram_rate"] for r in rows)
        d1 = mean(r["distinct_1"] for r in rows)
        d2 = mean(r["distinct_2"] for r in rows)
        looping = sum(1 for r in rows if r["repeated_3gram_rate"] > 0.3)
        empty = sum(1 for r in rows if not r["output"])

        if rep > 0.3:
            verdict, status = "often gets stuck repeating itself", WARNING
        elif rep > 0.1:
            verdict, status = "repeats itself sometimes", WARNING
        else:
            verdict, status = "rarely repeats itself", OK

        summary = (
            f"The model {verdict}: {rep:.0%} of 3-word phrases are repeats "
            f"({looping} of {len(rows)} outputs look like loops)."
        )
        notes = [
            "Uses greedy decoding, which repeats more than sampling. Real-world settings "
            "(temperature, repetition penalty) usually reduce this.",
            "distinct-n = unique n-grams / total n-grams; closer to 1 means more varied text.",
        ]
        if empty:
            notes.append(f"{empty} prompt(s) produced empty output.")

        return self.result(
            summary,
            metrics={
                "repeated_3gram_rate": round(rep, 4),
                "distinct_1": round(d1, 4),
                "distinct_2": round(d2, 4),
                "looping_outputs": looping,
                "samples": len(rows),
            },
            details={"samples": rows},
            status=status,
            notes=notes,
        )
