"""Shared test fixtures.

Tests use a tiny, randomly initialised GPT-2 and a small BPE tokenizer built
on the fly, so they run offline in seconds and never download anything.
The model's outputs are gibberish, which is fine: we test that the machinery
works and returns well-formed results, not that the model is smart.
"""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llmreport._utils import load_data

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _corpus():
    texts = [t["text"] for t in load_data("perplexity")["texts"]]
    texts += load_data("repetition")["prompts"]
    texts += [q for pair in load_data("consistency")["pairs"] for q in pair]
    return texts * 3


def build_raw_tokenizer(specials=("<|endoftext|>",), vocab_size: int = 400) -> Tokenizer:
    """A byte-level BPE `tokenizers.Tokenizer` (the kind nanoGPT-style projects save as tokenizer.json)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(specials),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_corpus(), trainer)
    return tok


def build_tokenizer(chat: bool = False):
    tok = build_raw_tokenizer()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    if chat:
        fast.chat_template = CHAT_TEMPLATE
    return fast


def build_model(vocab_size: int):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.config._name_or_path = "tiny-random-gpt2"
    return model.eval()


@pytest.fixture(scope="session")
def tokenizer():
    return build_tokenizer()


@pytest.fixture(scope="session")
def chat_tokenizer():
    return build_tokenizer(chat=True)


@pytest.fixture(scope="session")
def model(tokenizer):
    return build_model(len(tokenizer))


@pytest.fixture(scope="session")
def full_report(model, tokenizer):
    """One full default run, shared by the report tests."""
    import llmreport

    return llmreport.analyze(
        model,
        tokenizer,
        verbose=False,
        max_new_tokens=12,
        options={"performance": {"prompt_lengths": [16, 64], "repeats": 1, "new_tokens": 8}},
    )
