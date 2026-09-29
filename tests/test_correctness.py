"""Known-answer tests: does llmreport report the *right* numbers?

Each test sets up a situation where the correct result is known in advance
(published parameter counts, a model whose perplexity is exactly its vocabulary
size, scripted models whose behavior we control, a model with a known speed)
and checks that llmreport reports it.
"""

import difflib
import math
import time

import pytest
import torch
from conftest import build_model
from torch import nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutput

import llmreport
from llmreport._utils import load_data
from llmreport.analyzers.architecture import ArchitectureAnalyzer
from llmreport.analyzers.base import RunConfig

FAST = dict(verbose=False, max_new_tokens=40)


# ======================================================== architecture
# Models are built on the "meta" device: correct shapes, no memory used.


def _arch(model):
    return ArchitectureAnalyzer().run(model, None, RunConfig())


def test_gpt2_small_parameter_count_matches_published():
    with torch.device("meta"):
        model = GPT2LMHeadModel(GPT2Config())
    r = _arch(model)
    assert r.metrics["total_params"] == 124_439_808  # GPT-2 "124M"
    assert r.metrics["layers"] == 12 and r.metrics["attention_heads"] == 12
    assert r.metrics["max_context"] == 1024


def _llama2_7b():
    cfg = LlamaConfig(
        vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=4096,
    )
    with torch.device("meta"):
        return LlamaForCausalLM(cfg).to(torch.float16)


def test_llama2_7b_parameter_count_and_breakdown():
    r = _arch(_llama2_7b())
    assert r.metrics["total_params"] == 6_738_415_616  # Llama-2-7B
    b = r.details["param_breakdown"]
    assert b["embeddings"] == 32000 * 4096
    assert b["output_head"] == 32000 * 4096
    assert b["attention"] == 4 * 4096 * 4096 * 32
    assert b["mlp"] == 3 * 4096 * 11008 * 32
    assert b["norm"] == 4096 * (2 * 32 + 1)  # two norms per layer + final norm
    assert "other" not in b


def test_llama2_7b_memory_and_kv_cache():
    r = _arch(_llama2_7b())
    # 2 (K and V) x 32 layers x 32 heads x 128 head_dim x 2 bytes = 512 KiB per token
    assert r.metrics["kv_cache_bytes_per_token"] == 524_288
    assert r.metrics["kv_cache_bytes_full_context"] == 524_288 * 4096  # 2 GiB at 4k tokens
    assert r.metrics["weights_bytes"] == 6_738_415_616 * 2
    assert r.metrics["dtype"] == "float16"


def test_mistral_7b_grouped_query_attention():
    cfg = MistralConfig(
        vocab_size=32000, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8,
    )
    with torch.device("meta"):
        model = MistralForCausalLM(cfg).to(torch.bfloat16)
    r = _arch(model)
    assert r.metrics["total_params"] == 7_241_732_096  # Mistral-7B
    # Only 8 KV heads: 2 x 32 x 8 x 128 x 2 bytes = 128 KiB per token
    assert r.metrics["kv_cache_bytes_per_token"] == 131_072


def test_frozen_layers_are_reported():
    model = build_model(400)
    for p in model.transformer.h[0].parameters():
        p.requires_grad = False
    r = _arch(model)
    frozen = sum(p.numel() for p in model.transformer.h[0].parameters())
    assert r.metrics["trainable_params"] == r.metrics["total_params"] - frozen
    for p in model.parameters():
        p.requires_grad = True


# ======================================================== perplexity


def test_uniform_model_has_perplexity_equal_to_vocab_size(tokenizer):
    """A model that gives every token equal probability has perplexity = vocab size."""
    model = build_model(len(tokenizer))
    with torch.no_grad():
        model.transformer.wte.weight.zero_()  # lm_head is tied, so every logit is 0
    r = llmreport.analyze(model, tokenizer, checks="perplexity", verbose=False)["perplexity"]
    vocab = model.config.vocab_size
    assert r.metrics["perplexity"] == pytest.approx(vocab, rel=1e-4)


def test_perplexity_matches_manual_computation(model, tokenizer):
    texts = ["The river flows north to the sea.", "Plants turn sunlight into sugar and oxygen."]
    r = llmreport.analyze(model, tokenizer, checks="perplexity", prompts={"perplexity": texts},
                          verbose=False)["perplexity"]
    nll, n, chars = 0.0, 0, 0
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt")["input_ids"]
            logp = torch.log_softmax(model(ids).logits[0, :-1], dim=-1)
            nll -= logp.gather(1, ids[0, 1:, None]).sum().item()
            n += ids.shape[1] - 1
            chars += len(tokenizer.decode(ids[0, 1:]))
    assert r.metrics["perplexity"] == pytest.approx(math.exp(nll / n), rel=1e-3)
    assert r.metrics["bits_per_char"] == pytest.approx(nll / chars / math.log(2), rel=1e-3)
    assert r.metrics["tokens_scored"] == n


# ======================================================== scripted models
# A fake language model whose answers we write ourselves, so we know what
# the behavior checks *should* conclude.


class ScriptedLM(nn.Module):
    def __init__(self, tokenizer, respond):
        super().__init__()
        self.tok = tokenizer
        self.respond = respond
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = type("Cfg", (), {"max_position_embeddings": 2048, "_name_or_path": "scripted"})()

    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        vocab = len(self.tok)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
        loss = torch.tensor(math.log(vocab)) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits)

    def generate(self, input_ids, max_new_tokens=20, attention_mask=None, **kw):
        prompt = self.tok.decode(input_ids[0])
        reply = self.tok(self.respond(prompt), add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat([input_ids, reply[:, :max_new_tokens]], dim=1)


def run(tok, respond, check, **kw):
    return llmreport.analyze(ScriptedLM(tok, respond), tok, checks=check, **FAST, **kw)[check]


# ---- repetition


def test_repetition_flags_a_looping_model(tokenizer):
    r = run(tokenizer, lambda p: "the cat sat on the mat " * 10, "repetition")
    assert r.status == "warning"
    assert r.metrics["repeated_3gram_rate"] > 0.7
    assert r.metrics["looping_outputs"] == r.metrics["samples"]


def test_repetition_passes_varied_text(tokenizer):
    r = run(tokenizer, lambda p: " ".join(f"w{i}" for i in range(12)), "repetition")  # fits in max_new_tokens
    assert r.status == "ok"
    assert r.metrics["repeated_3gram_rate"] == 0
    assert r.metrics["distinct_1"] == 1.0


# ---- consistency

PAIRS = load_data("consistency")["pairs"]


def _pair_index(prompt):
    for i, (a, b, _) in enumerate(PAIRS):
        if a in prompt or b in prompt:
            return i
    return None


def _correct(prompt):
    i = _pair_index(prompt)
    return f"The answer is {PAIRS[i][2][0]}." if i is not None else "Not sure."


def test_consistency_correct_in_both_phrasings(tokenizer):
    r = run(tokenizer, _correct, "consistency")
    assert r.status == "ok"
    assert r.metrics["accuracy"] == 1.0
    assert r.metrics["correct_answer_agreement"] == 1.0


def test_consistency_flags_answers_that_depend_on_wording(tokenizer):
    first_phrasings = [a for a, _, _ in PAIRS]
    r = run(tokenizer, lambda p: _correct(p) if any(q in p for q in first_phrasings) else "Maybe blue.",
            "consistency")
    assert r.metrics["accuracy"] == 0.5
    assert r.metrics["correct_answer_agreement"] == 0.0
    assert r.status == "warning" and "one phrasing but not the other" in r.summary


def test_consistency_is_not_fooled_by_one_reply_for_everything(tokenizer):
    r = run(tokenizer, lambda p: "I like turtles very much.", "consistency")
    assert r.status == "warning"
    assert "none of" in r.summary and r.metrics["accuracy"] == 0


def test_consistency_is_not_fooled_by_loops(tokenizer):
    r = run(tokenizer, lambda p: "the the the the the the the the the the the the", "consistency")
    assert r.status == "warning" and "empty or repetitive" in r.summary


def test_consistency_custom_pairs_without_answers(tokenizer):
    pairs = [["Favourite colour?", "Which colour do you like best?"], ["Your name?", "What are you called?"],
             ["Where do you live?", "What is your home town?"]]
    same = {0: "I like green leaves", 1: "My name is Robo Tron", 2: "I live near the sea"}
    idx = lambda p: next(i for i, pr in enumerate(pairs) if pr[0] in p or pr[1] in p)  # noqa: E731
    good = run(tokenizer, lambda p: same[idx(p)], "consistency", prompts={"consistency": pairs})
    assert good.status == "ok" and "accuracy" not in good.metrics
    lazy = run(tokenizer, lambda p: "I like green leaves", "consistency", prompts={"consistency": pairs})
    assert lazy.status == "warning" and "same answer to every question" in lazy.summary


# ---- typo robustness

QUESTIONS = [a for a, _, _ in PAIRS]


def _tolerant(prompt):
    # Recovers the intended question even with typos, like a robust model.
    best = difflib.get_close_matches(prompt.strip(), QUESTIONS, n=1, cutoff=0.0)[0]
    return f"The answer is {PAIRS[QUESTIONS.index(best)][2][0]}."


def _brittle(prompt):
    for i, q in enumerate(QUESTIONS):
        if q in prompt:
            return f"The answer is {PAIRS[i][2][0]}."
    return "I do not understand the question."


def test_robustness_passes_typo_tolerant_model(tokenizer):
    r = run(tokenizer, _tolerant, "robustness")
    assert r.metrics["clean_accuracy"] == 1.0 and r.metrics["typo_correct_retained"] == 1.0
    assert "handles typos well" in r.summary
    assert r.status == "info"  # scripted model ignores context, so only half of the check applies
    assert "barely uses earlier context" in r.summary


def test_robustness_flags_brittle_model(tokenizer):
    r = run(tokenizer, _brittle, "robustness")
    assert r.metrics["clean_accuracy"] == 1.0
    assert r.metrics["typo_correct_retained"] < 0.5
    assert r.status == "warning" and "sensitive" in r.summary


class ContextLM(ScriptedLM):
    """Loss is low (1.0) when the model 'understands' the context, high (3.0) otherwise."""

    def __init__(self, tokenizer, respond, understands):
        super().__init__(tokenizer, respond)
        self.understands = understands

    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        out = super().forward(input_ids, labels, attention_mask)
        if labels is not None:
            text = self.tok.decode(input_ids[0])
            out.loss = torch.tensor(1.0 if self.understands(text) else 3.0)
        return out


CLEAN_CONTEXTS = [" ".join(t["text"].split(" ")[: len(t["text"].split(" ")) // 2])
                  for t in load_data("perplexity")["texts"]]


def test_robustness_prediction_part_known_answers(tokenizer):
    # Brittle: only understands the exact clean context -> typos destroy 100% of the context benefit.
    brittle = ContextLM(tokenizer, _tolerant, lambda text: any(c in text for c in CLEAN_CONTEXTS))
    every_word = {"robustness": {"typo_rate": 1.0}}  # so every passage surely gets typos
    r = llmreport.analyze(brittle, tokenizer, checks="robustness", options=every_word, **FAST)["robustness"]
    assert r.metrics["context_benefit"] == pytest.approx(1 - math.exp(-2), rel=1e-3)
    assert r.metrics["typo_context_benefit_lost"] == pytest.approx(1.0)
    assert r.status == "warning" and "sensitive" in r.summary

    # Robust: understands any real context, typos or not -> nothing lost.
    robust = ContextLM(tokenizer, _tolerant, lambda text: not text.startswith("<|endoftext|>"))
    r = llmreport.analyze(robust, tokenizer, checks="robustness", **FAST)["robustness"]
    assert r.metrics["typo_context_benefit_lost"] == pytest.approx(0.0)
    assert r.status == "ok" and "handles typos well" in r.summary


def test_robustness_does_not_judge_a_model_that_knows_nothing(tokenizer):
    r = run(tokenizer, lambda p: "Once upon a time there was a small green frog.", "robustness")
    assert r.status == "info" and "can't be judged" in r.summary


# ---- refusals

REF = load_data("refusal")
REFUSAL = "I'm sorry, but I can't help with that."
HELPFUL = "Sure! Here is a clear answer with the steps you need."


def _safe(prompt):
    return REFUSAL if any(h in prompt for h in REF["should_refuse"]) else HELPFUL


def test_refusal_well_behaved_chat_model(chat_tokenizer):
    r = run(chat_tokenizer, _safe, "refusal")
    assert r.metrics["harmful_refusal_rate"] == 1.0
    assert r.metrics["over_refusal_rate"] == 0.0
    assert r.status == "ok"


def test_refusal_flags_over_refusing_model(chat_tokenizer):
    r = run(chat_tokenizer, lambda p: REFUSAL, "refusal")
    assert r.metrics["over_refusal_rate"] == 1.0
    assert r.status == "warning"


def test_refusal_flags_model_that_never_refuses(chat_tokenizer):
    r = run(chat_tokenizer, lambda p: HELPFUL, "refusal")
    assert r.metrics["harmful_refusal_rate"] == 0.0
    assert r.status == "warning"


def test_refusal_ignores_ordinary_negative_sentences(chat_tokenizer):
    r = run(chat_tokenizer, lambda p: "I will not stay here tonight, my lord, I cannot wait.", "refusal")
    assert r.metrics["harmful_refusal_rate"] == 0.0 and r.metrics["over_refusal_rate"] == 0.0


def test_refusal_chat_template_is_applied(chat_tokenizer):
    seen = []
    run(chat_tokenizer, lambda p: seen.append(p) or HELPFUL, "refusal")
    assert all("<|user|>" in p and "<|assistant|>" in p for p in seen)


# ======================================================== performance


class SlowGPT2(GPT2LMHeadModel):
    """Every forward pass takes at least 20 ms, so decoding runs at about 50 tokens/s."""

    def forward(self, *args, **kwargs):
        time.sleep(0.02)
        return super().forward(*args, **kwargs)


def test_performance_measures_known_speed(tokenizer):
    torch.manual_seed(0)
    model = SlowGPT2(GPT2Config(vocab_size=len(tokenizer), n_positions=256, n_embd=32, n_layer=2, n_head=2))
    r = llmreport.analyze(
        model.eval(), tokenizer, checks="performance", verbose=False,
        options={"performance": {"prompt_lengths": [16], "repeats": 1, "new_tokens": 10}},
    )["performance"]
    assert 20 <= r.metrics["ttft_ms"] < 150  # at least the 20 ms sleep; shared CI machines add overhead
    assert 20 < r.metrics["decode_tokens_per_s"] <= 51  # can never beat 1 step per 20 ms


# ======================================================== easy calling


def test_many_ways_to_call(model, tokenizer, tmp_path):
    from transformers import pipeline

    ways = [
        llmreport.analyze(model, tokenizer, checks="architecture", verbose=False),
        llmreport.analyze((model, tokenizer), checks=["architecture"], verbose=False),
        llmreport.analyze(pipeline("text-generation", model=model, tokenizer=tokenizer),
                          checks="architecture", verbose=False),
    ]
    # A saved folder: the tokenizer is found automatically.
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    ways.append(llmreport.analyze(str(tmp_path), checks="architecture", device="cpu", verbose=False))
    loaded = GPT2LMHeadModel.from_pretrained(tmp_path)
    ways.append(llmreport.analyze(loaded, checks="architecture", verbose=False))  # no tokenizer passed
    counts = {w["architecture"].metrics["total_params"] for w in ways}
    assert counts == {sum(p.numel() for p in model.parameters())}


def test_comma_separated_checks_and_skip(model, tokenizer):
    r = llmreport.analyze(model, tokenizer, checks="architecture, perplexity", skip="perplexity", verbose=False)
    assert [x.name for x in r] == ["architecture"]


def test_helpful_errors(model, tokenizer):
    with pytest.raises(ValueError, match="Did you mean"):
        llmreport.analyze(model, tokenizer, checks="perplexity_check", verbose=False)
    with pytest.raises(ValueError, match=r"Did you mean \['perplexity'\]"):
        llmreport.analyze(model, tokenizer, checks="perplexty", verbose=False)
    with pytest.raises(TypeError, match="cannot generate text"):
        llmreport.analyze(nn.Linear(2, 2), tokenizer, verbose=False)
    with pytest.raises(ValueError, match="mode"):
        llmreport.analyze(model, tokenizer, mode="fast", verbose=False)


def test_does_not_modify_users_tokenizer(model, tokenizer):
    before = tokenizer.pad_token
    llmreport.analyze(model, tokenizer, checks="repetition", **FAST)
    assert tokenizer.pad_token == before
