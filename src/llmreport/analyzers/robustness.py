"""Does quality drop when the input contains typos?"""

from __future__ import annotations

import math
import random

import torch

from .._utils import (
    add_typos,
    generate,
    is_degenerate,
    load_data,
    max_context,
    mean,
    mentions,
    model_device,
    similarity,
)
from .base import INFO, OK, WARNING, Analyzer, RunConfig
from .perplexity import perplexity_texts


@torch.no_grad()
def _target_nll(model, tokenizer, context: str, target: str) -> float:
    """Mean negative log-likelihood of ``target`` given ``context``."""
    device = model_device(model)
    ctx = tokenizer(context, return_tensors="pt")["input_ids"]
    tgt = tokenizer(" " + target, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([ctx, tgt], dim=1)[:, : max_context(model, tokenizer)].to(device)
    labels = ids.clone()
    labels[:, : ctx.shape[1]] = -100  # only score the (clean) target
    if (labels != -100).sum() == 0:
        return float("nan")
    return float(model(input_ids=ids, labels=labels).loss)


def _split(text: str):
    words = text.split(" ")
    half = len(words) // 2
    return " ".join(words[:half]), " ".join(words[half:])


class RobustnessAnalyzer(Analyzer):
    name = "robustness"
    title = "Typo robustness"
    description = "Adds typos to the input and measures how much predictions and answers change."
    needs_generation = True

    def run(self, model, tokenizer, config: RunConfig):
        rate = config.option(self.name, "typo_rate", 0.15)
        rng = random.Random(config.seed)

        # Part 1: how much harder does the clean second half become after a noisy first half?
        texts = config.sample(perplexity_texts(config), 6)
        empty_context = tokenizer.bos_token or tokenizer.eos_token or ""
        nll_rows = []
        for text in texts:
            context, target = _split(text)
            noisy = add_typos(context, rate, rng)
            clean_nll = _target_nll(model, tokenizer, context, target)
            noisy_nll = _target_nll(model, tokenizer, noisy, target)
            bare_nll = _target_nll(model, tokenizer, empty_context, target)
            if any(math.isnan(v) for v in (clean_nll, noisy_nll, bare_nll)):
                continue
            nll_rows.append({"noisy_context": noisy, "no_context_nll": round(bare_nll, 4),
                             "clean_nll": round(clean_nll, 4), "noisy_nll": round(noisy_nll, 4)})
        clean = mean(r["clean_nll"] for r in nll_rows)
        noisy = mean(r["noisy_nll"] for r in nll_rows)
        bare = mean(r["no_context_nll"] for r in nll_rows)
        ppl_increase = math.exp(noisy - clean) - 1 if nll_rows else float("nan")
        # How much the context helps at all. If it barely helps, typos in it can't hurt much either.
        context_benefit = 1 - math.exp(clean - bare) if nll_rows else float("nan")
        # Share of the context's benefit that typos destroy.
        benefit_lost = (noisy - clean) / (bare - clean) if nll_rows and bare - clean > 1e-6 else float("nan")

        # Part 2: are misspelled questions still answered correctly?
        custom = config.prompts.get(self.name)
        if custom is None:
            items = [(a, answers) for a, _, answers in load_data("consistency")["pairs"]]
        else:
            items = [(q, None) if isinstance(q, str) else (q[0], list(q[1])) for q in custom]
        items = config.sample(items, 6)
        max_new = config.gen_tokens(32, 64)
        gen_rows = []
        for q, answers in items:
            noisy_q = add_typos(q, max(rate, 0.3), rng)
            a = generate(model, tokenizer, q, max_new, config.use_chat_template)
            b = generate(model, tokenizer, noisy_q, max_new, config.use_chat_template)
            row = {"question": q, "noisy_question": noisy_q, "answer": a, "noisy_answer": b,
                   "similarity": round(similarity(a, b), 3)}
            if answers:
                row["correct"] = mentions(a, answers)
                row["noisy_correct"] = mentions(b, answers)
            gen_rows.append(row)
        n = len(gen_rows)
        stability = mean(r["similarity"] for r in gen_rows)
        baseline = (
            mean(similarity(gen_rows[i]["answer"], gen_rows[(i + 1) % n]["noisy_answer"]) for i in range(n))
            if n > 1 else 0.0
        )
        degenerate = sum(is_degenerate(r["answer"]) for r in gen_rows)
        labeled = [r for r in gen_rows if "correct" in r]
        clean_right = [r for r in labeled if r["correct"]]
        retained = mean(r["noisy_correct"] for r in clean_right) if clean_right else float("nan")

        # ---- verdict
        notes = [
            f"About {rate:.0%} of words in context passages (and 30% in questions) get one typo: "
            "a swapped, dropped or doubled letter.",
        ]
        answers_usable = degenerate < max(1, n / 2) and (bool(clean_right) if labeled else baseline <= 0.6)
        context_usable = not math.isnan(context_benefit) and context_benefit >= 0.05

        pred_part = (
            f"typos in the context cancel {benefit_lost:.0%} of the benefit the model gets from it"
            if context_usable else "the model barely uses earlier context, so typos in it have little effect"
        )
        if labeled and answers_usable:
            ans_part = (f"it still answers {retained:.0%} of the {len(clean_right)} questions it gets right "
                        "when they are misspelled")
            ans_score = retained
        elif answers_usable:
            ans_part = f"answers to misspelled questions overlap {stability:.0%} with the originals (unrelated {baseline:.0%})"
            ans_score = min(1.0, max(0.0, (stability - baseline) / max(1e-9, 1 - baseline)))
        else:
            ans_part = None
            ans_score = float("nan")
            reason = ("its answers are mostly empty or repetitive" if degenerate >= max(1, n / 2)
                      else "it answered none of the clean questions correctly")
            notes.append(f"The answer part was not judged because {reason}.")

        pred_score = 1 - benefit_lost if context_usable else float("nan")
        scores = [x for x in (pred_score, ans_score) if not math.isnan(x)]
        if not scores:
            verdict, status = None, INFO
        else:
            worst = min(scores)
            if worst < 0.5:
                verdict, status = "is sensitive to typos", WARNING
            elif worst < 0.8:
                verdict, status = "is somewhat sensitive to typos", OK
            else:
                verdict, status = "handles typos well", OK
            if len(scores) == 1:
                status = INFO if status == OK else status
        if verdict is None:
            summary = f"Typo robustness can't be judged: {pred_part}, and " + (
                "its answers are mostly empty or repetitive." if degenerate >= max(1, n / 2)
                else "it answered none of the clean questions correctly.")
        else:
            summary = f"The model {verdict}: {pred_part}" + (f", and {ans_part}." if ans_part else ".")
        if context_usable:
            notes.append(f"Perplexity of the clean second half rises {ppl_increase:+.1%} when the first half has typos "
                         f"(context lowers it by {context_benefit:.0%} without typos).")

        metrics = {
            "typo_perplexity_increase": None if math.isnan(ppl_increase) else round(ppl_increase, 4),
            "context_benefit": None if math.isnan(context_benefit) else round(context_benefit, 4),
            "typo_context_benefit_lost": None if math.isnan(benefit_lost) else round(benefit_lost, 4),
        }
        if labeled:
            metrics["clean_accuracy"] = round(mean(r["correct"] for r in labeled), 4)
            metrics["typo_accuracy"] = round(mean(r["noisy_correct"] for r in labeled), 4)
            metrics["typo_correct_retained"] = round(retained, 4) if clean_right else None
        metrics.update({
            "typo_answer_overlap": round(stability, 4),
            "unrelated_baseline": round(baseline, 4),
            "typo_rate": rate,
        })
        return self.result(
            summary,
            metrics=metrics,
            details={"prediction": nll_rows, "answers": gen_rows},
            status=status,
            notes=notes,
        )
