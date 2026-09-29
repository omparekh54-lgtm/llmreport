"""How fast and how heavy the model is: prefill latency, decode speed and memory."""

from __future__ import annotations

import statistics

import psutil
import torch

from .._utils import human_bytes, timer
from ..adapters import as_model, as_tokenizer
from .base import OK, WARNING, Analyzer, RunConfig

_FILLER = (
    "Language models read text as tokens and predict what comes next. "
    "This sentence is only here to make a prompt of a known length. "
)


def _prompt_ids(tok, length: int, vocab_limit=None):
    ids = tok.encode(_FILLER)
    if vocab_limit:
        ids = [i for i in ids if i < vocab_limit]
    if not ids:
        ids = [tok.eos_id or 0]
    reps = length // len(ids) + 1
    return (ids * reps)[:length]


class PerformanceAnalyzer(Analyzer):
    name = "performance"
    title = "Performance"
    description = "Time to first token, generation speed and memory at several prompt lengths."
    needs_generation = True

    @torch.no_grad()
    def run(self, model, tokenizer, config: RunConfig):
        tok = as_tokenizer(tokenizer)
        lm = as_model(model, tok)
        device = lm.device
        new_tokens = config.option(self.name, "new_tokens", 32 if config.quick else 64)
        repeats = config.option(self.name, "repeats", 3 if config.quick else 5)
        lengths = config.option(
            self.name, "prompt_lengths", [32, 128, 512] if config.quick else [32, 128, 512, 1024, 2048]
        )
        limit = lm.usable_context()
        usable = [n for n in lengths if n + new_tokens <= limit]
        skipped = [n for n in lengths if n not in usable]
        if not usable:
            usable = [max(1, limit - new_tokens)]

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        process = psutil.Process()
        rss_before = process.memory_info().rss

        def run_generate(ids):
            return lm.generate_ids(ids, new_tokens, min_new_tokens=new_tokens)  # fixed length: comparable speeds

        # Warm-up: the first call pays one-off costs (kernel compilation, allocation).
        warm = _prompt_ids(tok, usable[0])
        lm.logits(torch.tensor([warm], device=device))
        run_generate(warm)

        rows = []
        for length in usable:
            ids = _prompt_ids(tok, length)
            tensor = torch.tensor([ids], device=device)
            prefill, total, produced = [], [], new_tokens
            for _ in range(repeats):
                with timer(device) as t:
                    lm.logits(tensor)
                prefill.append(t["seconds"])
                with timer(device) as t:
                    out = run_generate(ids)
                total.append(t["seconds"])
                produced = max(1, len(out))
            ttft = statistics.median(prefill)
            gen_time = statistics.median(total)
            decode_time = max(gen_time - ttft, 1e-9)
            rows.append(
                {
                    "prompt_tokens": length,
                    "new_tokens": produced,
                    "ttft_ms": round(ttft * 1000, 2),
                    "total_ms": round(gen_time * 1000, 2),
                    "decode_tokens_per_s": round(max(1, produced - 1) / decode_time, 2),
                    "prefill_tokens_per_s": round(length / max(ttft, 1e-9), 1),
                }
            )

        metrics = {
            "device": str(device),
            "ttft_ms": rows[0]["ttft_ms"],
            "decode_tokens_per_s": rows[0]["decode_tokens_per_s"],
            "ttft_ms_longest_prompt": rows[-1]["ttft_ms"],
            "decode_tokens_per_s_longest_prompt": rows[-1]["decode_tokens_per_s"],
        }
        if device.type == "cuda":
            metrics["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["process_rss_bytes"] = process.memory_info().rss
        metrics["rss_growth_bytes"] = max(0, metrics["process_rss_bytes"] - rss_before)

        speed = rows[0]["decode_tokens_per_s"]
        summary = (
            f"Generates about {speed:,.1f} tokens/s on {device}, "
            f"with {rows[0]['ttft_ms']:,.0f} ms to first token for a {rows[0]['prompt_tokens']}-token prompt"
        )
        if len(rows) > 1:
            summary += f" ({rows[-1]['ttft_ms']:,.0f} ms at {rows[-1]['prompt_tokens']} tokens)."
        else:
            summary += "."

        notes = [
            f"Median of {repeats} runs after one warm-up, greedy decoding, batch size 1.",
            "Speeds depend heavily on hardware, dtype and other programs running.",
        ]
        if not lm.uses_model_generate:
            notes.append("Generated with llmreport's built-in loop, which re-runs the whole sequence for every "
                         "new token (no KV cache), exactly like a model without a cache. If your own generate() "
                         "has a KV cache, it will be faster, especially for long prompts.")
        if "peak_gpu_memory_bytes" in metrics:
            notes.append(f"Peak GPU memory during the test: {human_bytes(metrics['peak_gpu_memory_bytes'])}.")
        if skipped:
            notes.append(f"Skipped prompt lengths longer than the context window: {skipped}.")
        status = OK
        if speed < 1:
            status = WARNING
            notes.append("Under 1 token/s: consider a GPU, half precision or quantization.")

        return self.result(summary, metrics=metrics, details={"by_prompt_length": rows}, status=status, notes=notes)
