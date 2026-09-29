"""How fast and how heavy the model is: prefill latency, decode speed and memory."""

from __future__ import annotations

import statistics

import psutil
import torch

from .._utils import human_bytes, max_context, model_device, pad_id, timer
from .base import OK, WARNING, Analyzer, RunConfig

_FILLER = (
    "Language models read text as tokens and predict what comes next. "
    "This sentence is only here to make a prompt of a known length. "
)


def _prompt_ids(tokenizer, length: int, device) -> torch.Tensor:
    ids = tokenizer(_FILLER, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    reps = length // len(ids) + 1
    return torch.tensor([(ids * reps)[:length]], device=device)


class PerformanceAnalyzer(Analyzer):
    name = "performance"
    title = "Performance"
    description = "Time to first token, generation speed and memory at several prompt lengths."
    needs_generation = True

    @torch.no_grad()
    def run(self, model, tokenizer, config: RunConfig):
        device = model_device(model)
        new_tokens = config.option(self.name, "new_tokens", 32 if config.quick else 64)
        repeats = config.option(self.name, "repeats", 3 if config.quick else 5)
        lengths = config.option(
            self.name, "prompt_lengths", [32, 128, 512] if config.quick else [32, 128, 512, 1024, 2048]
        )
        limit = max_context(model, tokenizer)
        usable = [n for n in lengths if n + new_tokens <= limit]
        skipped = [n for n in lengths if n not in usable]
        if not usable:
            usable = [max(1, limit - new_tokens)]

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        process = psutil.Process()
        rss_before = process.memory_info().rss

        gen_kwargs = dict(
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,  # force a fixed length so speeds are comparable
            do_sample=False,
            pad_token_id=pad_id(tokenizer),
        )

        # Warm-up: the first call pays one-off costs (kernel compilation, allocation).
        warm = _prompt_ids(tokenizer, usable[0], device)
        model(input_ids=warm)
        model.generate(input_ids=warm, attention_mask=torch.ones_like(warm), **gen_kwargs)

        rows = []
        for length in usable:
            ids = _prompt_ids(tokenizer, length, device)
            mask = torch.ones_like(ids)
            prefill, total = [], []
            for _ in range(repeats):
                with timer(device) as t:
                    model(input_ids=ids, attention_mask=mask)
                prefill.append(t["seconds"])
                with timer(device) as t:
                    model.generate(input_ids=ids, attention_mask=mask, **gen_kwargs)
                total.append(t["seconds"])
            ttft = statistics.median(prefill)
            gen_time = statistics.median(total)
            decode_time = max(gen_time - ttft, 1e-9)
            rows.append(
                {
                    "prompt_tokens": length,
                    "new_tokens": new_tokens,
                    "ttft_ms": round(ttft * 1000, 2),
                    "total_ms": round(gen_time * 1000, 2),
                    "decode_tokens_per_s": round((new_tokens - 1) / decode_time, 2),
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
        if "peak_gpu_memory_bytes" in metrics:
            notes.append(f"Peak GPU memory during the test: {human_bytes(metrics['peak_gpu_memory_bytes'])}.")
        if skipped:
            notes.append(f"Skipped prompt lengths longer than the context window: {skipped}.")
        status = OK
        if speed < 1:
            status = WARNING
            notes.append("Under 1 token/s: consider a GPU, half precision or quantization.")

        return self.result(summary, metrics=metrics, details={"by_prompt_length": rows}, status=status, notes=notes)
