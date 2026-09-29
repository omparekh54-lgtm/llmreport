"""Shared helpers: device handling, generation, perplexity, text metrics, data loading."""

from __future__ import annotations

import json
import math
import random
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from typing import Dict, Iterable, List, Optional, Sequence

import torch

# ---------------------------------------------------------------- data files


@lru_cache(maxsize=None)
def load_data(name: str):
    """Load a bundled JSON prompt set from ``llmreport/data``."""
    text = resources.files("llmreport").joinpath("data", f"{name}.json").read_text("utf-8")
    return json.loads(text)


def prompts_for(config, analyzer: str, default_file: str, key: Optional[str] = None):
    """Return user-supplied prompts if given, otherwise the bundled set."""
    if analyzer in config.prompts:
        return config.prompts[analyzer]
    data = load_data(default_file)
    return data[key] if key else data


# ---------------------------------------------------------------- torch helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@contextmanager
def timer(device: torch.device):
    """Context manager that measures wall-clock seconds, synchronising the GPU."""
    box = {}
    sync(device)
    start = time.perf_counter()
    try:
        yield box
    finally:
        sync(device)
        box["seconds"] = time.perf_counter() - start


def max_context(model, tokenizer=None, fallback: int = 2048) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10**7:
        return tok_max
    return fallback


def pad_id(tokenizer) -> int:
    """Padding id for generate(), without modifying the user's tokenizer."""
    for value in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if value is not None:
            return value if isinstance(value, int) else value[0]
    return 0


def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_prompt(tokenizer, prompt: str, use_chat_template="auto") -> str:
    """Wrap a user message in the tokenizer's chat template when appropriate."""
    use = has_chat_template(tokenizer) if use_chat_template == "auto" else bool(use_chat_template)
    if not use:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    use_chat_template="auto",
) -> str:
    """Greedy-decode a completion and return only the newly generated text."""
    device = model_device(model)
    text = build_prompt(tokenizer, prompt, use_chat_template)
    # Chat templates already contain special tokens such as <bos>.
    add_special = text == prompt
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special)
    enc = {k: v.to(device) for k, v in enc.items()}
    limit = max_context(model, tokenizer)
    if enc["input_ids"].shape[1] + max_new_tokens > limit:
        keep = max(1, limit - max_new_tokens)
        enc = {k: v[:, -keep:] for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id(tokenizer),
    )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def perplexity(model, tokenizer, texts: Sequence[str], max_length: Optional[int] = None) -> Dict:
    """Token-weighted perplexity over a list of texts.

    Returns a dict with ``perplexity``, ``mean_nll`` (natural log) and ``tokens``.
    """
    device = model_device(model)
    limit = min(max_length or 10**9, max_context(model, tokenizer))
    total_nll, total_tokens, total_chars = 0.0, 0, 0
    per_text = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :limit].to(device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1  # number of predicted tokens
        total_nll += float(out.loss) * n
        total_tokens += n
        total_chars += len(tokenizer.decode(ids[0, 1:], skip_special_tokens=True))
        per_text.append(math.exp(float(out.loss)))
    if total_tokens == 0:
        nan = float("nan")
        return {"perplexity": nan, "mean_nll": nan, "bits_per_char": nan, "tokens": 0, "per_text": []}
    mean_nll = total_nll / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        # Bits per character does not depend on the tokenizer, so it is fairer
        # than perplexity when comparing models with different vocabularies.
        "bits_per_char": total_nll / max(1, total_chars) / math.log(2),
        "tokens": total_tokens,
        "per_text": per_text,
    }


# ---------------------------------------------------------------- text metrics

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: Sequence[str], n: int) -> float:
    grams = ngrams(tokens, n)
    return len(set(grams)) / len(grams) if grams else 1.0


def repeated_ngram_rate(tokens: Sequence[str], n: int = 3) -> float:
    """Fraction of n-grams that already appeared earlier in the same text."""
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    seen, repeats = set(), 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1] (Jaccard over unigrams and bigrams).

    A cheap, dependency-free stand-in for semantic similarity. It rewards the
    same key words and phrases, so it underrates answers that say the same thing
    in different words.
    """
    wa, wb = words(a), words(b)
    if not wa and not wb:
        return 1.0
    sa = set(wa) | set(ngrams(wa, 2))
    sb = set(wb) | set(ngrams(wb, 2))
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def mentions(text: str, answers: Sequence[str]) -> bool:
    """True if ``text`` contains any accepted answer as a whole word or phrase."""
    lowered = text.lower()
    return any(re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", lowered) for a in answers)


def is_degenerate(text: str) -> bool:
    """Empty output, or output that is mostly repeated phrases or the same few words."""
    toks = words(text)
    if len(toks) < 2:
        return not text.strip()
    if len(toks) >= 8 and (repeated_ngram_rate(toks, 3) > 0.5 or distinct_n(toks, 1) < 0.3):
        return True
    return False


def add_typos(text: str, rate: float = 0.1, rng: Optional[random.Random] = None) -> str:
    """Inject realistic typos into roughly ``rate`` of the words in ``text``.

    Each chosen word gets one of: swap two adjacent letters, drop a letter,
    or double a letter. Words shorter than 3 letters are left alone.
    """
    rng = rng or random.Random(0)
    out = []
    for word in text.split(" "):
        letters = [i for i, c in enumerate(word) if c.isalpha()]
        if len(letters) >= 3 and rng.random() < rate:
            i = rng.choice(letters[:-1])
            kind = rng.choice(("swap", "drop", "double"))
            if kind == "swap":
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            elif kind == "drop":
                word = word[:i] + word[i + 1:]
            else:
                word = word[:i] + word[i] + word[i:]
        out.append(word)
    return " ".join(out)


def human_number(n: float) -> str:
    """1234567 -> '1.23M'."""
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return str(int(n)) if float(n).is_integer() else f"{n:.2f}"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"
