"""Does the model give the same answer when a question is reworded?"""

from __future__ import annotations

from .._utils import generate, is_degenerate, load_data, mean, mentions, similarity
from .base import INFO, OK, WARNING, Analyzer, RunConfig


def parse_pairs(items) -> list:
    """Accept [q1, q2] or [q1, q2, [accepted answers]] (or a dict with those keys)."""
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append((item["question_a"], item["question_b"], item.get("answers")))
        else:
            out.append((item[0], item[1], list(item[2]) if len(item) > 2 and item[2] else None))
    return out


class ConsistencyAnalyzer(Analyzer):
    name = "consistency"
    title = "Consistency"
    description = "Asks the same question two ways and checks the answers agree (and are correct)."
    needs_generation = True

    def run(self, model, tokenizer, config: RunConfig):
        pairs = parse_pairs(config.prompts.get(self.name) or load_data("consistency")["pairs"])
        pairs = config.sample(pairs, 6)
        max_new = config.gen_tokens(32, 64)

        rows = []
        for a, b, answers in pairs:
            out_a = generate(model, tokenizer, a, max_new, config.use_chat_template)
            out_b = generate(model, tokenizer, b, max_new, config.use_chat_template)
            row = {"question_a": a, "question_b": b, "answer_a": out_a, "answer_b": out_b,
                   "similarity": round(similarity(out_a, out_b), 3)}
            if answers:
                row["correct_a"] = mentions(out_a, answers)
                row["correct_b"] = mentions(out_b, answers)
                row["expected"] = " / ".join(answers)
            rows.append(row)

        n = len(rows)
        overlap = mean(r["similarity"] for r in rows)
        # Overlap between answers to *unrelated* questions: guards against models that
        # say roughly the same thing whatever they are asked.
        baseline = mean(similarity(rows[i]["answer_a"], rows[(i + 1) % n]["answer_b"]) for i in range(n)) if n > 1 else 0.0
        degenerate = sum(is_degenerate(r["answer_a"]) + is_degenerate(r["answer_b"]) for r in rows)
        labeled = [r for r in rows if "correct_a" in r]

        metrics = {}
        notes = []
        if labeled:
            accuracy = mean((r["correct_a"] + r["correct_b"]) / 2 for r in labeled)
            answered = [r for r in labeled if r["correct_a"] or r["correct_b"]]
            agreement = mean(r["correct_a"] and r["correct_b"] for r in answered) if answered else float("nan")
            metrics["accuracy"] = round(accuracy, 4)
            metrics["correct_answer_agreement"] = round(agreement, 4) if answered else None

        if degenerate >= n:  # at least half of all 2n answers are empty or loops
            status = WARNING
            summary = (f"Consistency can't be judged: {degenerate} of {2 * n} answers are empty or repetitive "
                       "(see the Repetition check).")
        elif labeled and not answered:
            status = WARNING
            summary = (f"Consistency can't be judged: the model answered none of {len(labeled)} simple factual "
                       "questions correctly in either phrasing.")
        elif labeled:
            share = metrics["correct_answer_agreement"]
            if share >= 0.8:
                verdict, status = "answers reworded questions consistently", OK
            elif share >= 0.5:
                verdict, status = "is partly consistent across rewordings", INFO
            else:
                verdict, status = "often answers a question correctly in one phrasing but not the other", WARNING
            summary = (f"The model {verdict}: of {len(answered)} questions it got right at least once, "
                       f"{share:.0%} were right in both phrasings (overall accuracy {metrics['accuracy']:.0%}).")
        else:
            gap = overlap - baseline
            if baseline > 0.6:
                verdict, status = "gives nearly the same answer to every question, so consistency is not meaningful", WARNING
            elif gap >= 0.3:
                verdict, status = "answers reworded questions consistently", OK
            elif gap >= 0.1:
                verdict, status = "is partly consistent across rewordings", INFO
            else:
                verdict, status = "gives quite different answers to reworded questions", WARNING
            summary = (f"The model {verdict}: answers to reworded questions overlap {overlap:.0%}, "
                       f"versus {baseline:.0%} for unrelated questions ({n} pairs).")

        if labeled:
            notes.append("An answer counts as correct if it mentions an accepted answer (for example \"Paris\"), "
                         "so a rambling answer that includes it still counts.")
        notes.append("Word overlap between the two answers (and against unrelated questions) is also reported. "
                      "It is a cheap heuristic: correct answers worded differently can overlap little.")
        if degenerate:
            notes.append(f"{degenerate} of {2 * n} answers were empty or repetitive.")

        metrics.update({
            "answer_overlap": round(overlap, 4),
            "unrelated_baseline": round(baseline, 4),
            "degenerate_answers": degenerate,
            "pairs": n,
        })
        return self.result(summary, metrics=metrics, details={"pairs": rows}, status=status, notes=notes)
