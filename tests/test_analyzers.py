import math

import pytest

import llmreport
from llmreport._utils import add_typos, distinct_n, repeated_ngram_rate, similarity, words
from llmreport.analyzers.refusal import is_refusal

FAST = dict(verbose=False, max_new_tokens=8)


def test_version():
    assert llmreport.__version__.count(".") == 2


def test_list_checks_has_defaults():
    checks = llmreport.list_checks()
    for name in llmreport.DEFAULT_CHECKS:
        assert name in checks and checks[name]


# ------------------------------------------------------------- text helpers


def test_distinct_and_repetition():
    toks = words("the cat sat on the mat the cat sat on the mat")
    assert 0 < distinct_n(toks, 1) < 1
    assert repeated_ngram_rate(toks, 3) > 0.3
    assert repeated_ngram_rate(words("one two three four five"), 3) == 0


def test_similarity_bounds():
    assert similarity("Paris is the capital", "Paris is the capital") == 1.0
    assert similarity("apples", "rockets") == 0.0
    assert 0 < similarity("The capital is Paris", "Paris is the capital") < 1


def test_add_typos_is_deterministic_and_changes_text():
    import random

    text = "The quick brown fox jumps over the lazy sleeping dog again and again"
    a = add_typos(text, 0.5, random.Random(1))
    b = add_typos(text, 0.5, random.Random(1))
    assert a == b and a != text
    assert add_typos(text, 0.0) == text


def test_refusal_detection():
    patterns = ["i can't", "i'm sorry, but"]
    assert is_refusal("I can’t help with that.", patterns)  # curly apostrophe
    assert is_refusal("I'm sorry, but no.", patterns)
    assert not is_refusal("Sure! Here is how to kill a Python process.", patterns)


# ------------------------------------------------------------- analyzers


def test_architecture_matches_model(model, tokenizer):
    r = llmreport.analyze(model, tokenizer, checks=["architecture"], **FAST)["architecture"]
    expected = sum(p.numel() for p in model.parameters())
    assert r.status == "ok"
    assert r.metrics["total_params"] == expected
    assert r.metrics["layers"] == 2
    assert r.metrics["hidden_size"] == 32
    assert r.metrics["max_context"] == 256
    assert r.metrics["est_fp16_bytes"] == expected * 2
    assert r.metrics["kv_cache_bytes_per_token"] == 2 * 2 * 2 * 16 * 4  # k+v * layers * heads * head_dim * fp32
    assert sum(r.details["param_breakdown"].values()) == expected
    assert "attention" in r.details["param_breakdown"]


def test_performance_respects_context(model, tokenizer):
    r = llmreport.analyze(
        model, tokenizer, checks=["performance"],
        options={"performance": {"prompt_lengths": [16, 10_000], "repeats": 1, "new_tokens": 4}}, **FAST,
    )["performance"]
    assert r.status in ("ok", "warning")
    assert [row["prompt_tokens"] for row in r.details["by_prompt_length"]] == [16]
    assert r.metrics["decode_tokens_per_s"] > 0
    assert any("Skipped prompt lengths" in n for n in r.notes)


def test_perplexity_random_model_is_high(model, tokenizer):
    r = llmreport.analyze(model, tokenizer, checks=["perplexity"], **FAST)["perplexity"]
    ppl = r.metrics["perplexity"]
    # A random model is close to uniform over the vocabulary.
    assert math.isfinite(ppl) and ppl > 50
    assert r.status == "warning"
    assert r.metrics["bits_per_char"] > 0


def test_custom_prompts_are_used(model, tokenizer):
    report = llmreport.analyze(
        model, tokenizer, checks=["repetition", "consistency"],
        prompts={"repetition": ["Hello there"], "consistency": [["Hi?", "Hello?"]]}, **FAST,
    )
    assert report["repetition"].metrics["samples"] == 1
    assert report["repetition"].details["samples"][0]["prompt"] == "Hello there"
    assert report["consistency"].metrics["pairs"] == 1


def test_robustness_and_refusal_shapes(model, tokenizer):
    report = llmreport.analyze(model, tokenizer, checks=["robustness", "refusal"], **FAST)
    rob = report["robustness"].metrics
    assert math.isfinite(rob["typo_perplexity_increase"])
    assert 0 <= rob["typo_answer_overlap"] <= 1
    assert rob["context_benefit"] is None or math.isfinite(rob["context_benefit"])
    ref = report["refusal"]
    assert ref.status == "info"  # no chat template -> base model
    assert 0 <= ref.metrics["harmful_refusal_rate"] <= 1


def test_chat_template_path(model, chat_tokenizer):
    report = llmreport.analyze(model, chat_tokenizer, checks=["refusal"], **FAST)
    assert report["refusal"].status in ("ok", "warning")  # treated as a chat model


def test_full_mode_uses_more_prompts(model, tokenizer):
    quick = llmreport.analyze(model, tokenizer, checks=["consistency"], **FAST)
    full = llmreport.analyze(model, tokenizer, checks=["consistency"], mode="full", **FAST)
    assert full["consistency"].metrics["pairs"] > quick["consistency"].metrics["pairs"]


def test_reproducible(model, tokenizer):
    a = llmreport.analyze(model, tokenizer, checks=["robustness"], seed=3, **FAST)
    b = llmreport.analyze(model, tokenizer, checks=["robustness"], seed=3, **FAST)
    assert a["robustness"].metrics == b["robustness"].metrics


# ------------------------------------------------------------- runner behaviour


def test_unknown_check_raises(model, tokenizer):
    with pytest.raises(ValueError, match="Unknown check"):
        llmreport.analyze(model, tokenizer, checks=["nope"], **FAST)


def test_skip(model, tokenizer):
    report = llmreport.analyze(model, tokenizer, checks=["architecture", "perplexity"], skip=["perplexity"], **FAST)
    assert [r.name for r in report] == ["architecture"]


def test_training_mode_restored(model, tokenizer):
    model.train()
    llmreport.analyze(model, tokenizer, checks=["architecture"], **FAST)
    assert model.training
    model.eval()


def test_custom_analyzer_and_error_isolation(model, tokenizer):
    @llmreport.register_analyzer
    class Broken(llmreport.Analyzer):
        name = "broken_for_test"
        title = "Broken"

        def run(self, model, tokenizer, config):
            raise RuntimeError("boom")

    @llmreport.register_analyzer
    class Fine(llmreport.Analyzer):
        name = "fine_for_test"
        title = "Fine"

        def run(self, model, tokenizer, config):
            return self.result("All good.", metrics={"score": 1.0})

    report = llmreport.analyze(model, tokenizer, checks=["broken_for_test", "fine_for_test"], **FAST)
    assert report["broken_for_test"].status == "error"
    assert "boom" in report["broken_for_test"].summary
    assert report["fine_for_test"].metrics == {"score": 1.0}

    with pytest.raises(RuntimeError):
        llmreport.analyze(model, tokenizer, checks=["broken_for_test"], raise_errors=True, **FAST)


def test_register_rejects_bad_classes():
    with pytest.raises(TypeError):
        llmreport.register_analyzer(object)
