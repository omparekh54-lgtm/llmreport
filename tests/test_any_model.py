"""llmreport on models it has never seen: written from scratch, loaded from checkpoints,
recurrent, quantized, TorchScript, or just a function.

Every test has a known right answer: a published parameter count, a manual perplexity
computed straight from the logits, the same weights run through Hugging Face, or a
scripted model whose output we control.
"""

import math
import os
import sys
import time

import pytest
import torch
from conftest import build_raw_tokenizer
from torch import nn

import llmreport
from llmreport._utils import generate
from llmreport.adapters import LanguageModel, TokenizerAdapter
from llmreport.analyzers.architecture import ArchitectureAnalyzer
from llmreport.analyzers.base import RunConfig
from llmreport.analyzers.code import extract_function, run_tests, truncate_body

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scratch"))
from gpt_model import GPT, GPTConfig  # noqa: E402  (the author's own from-scratch model)
from models import (  # noqa: E402
    LSTMLM,
    LastOnlyGPT,
    LlamaArgs,
    LlamaStyle,
    NeedsStartPos,
    ScriptedGPT,
    SeqFirstTransformer,
    uniform_logits_model,
)

SPECIALS = ["<|endoftext|>", "<|endoffile|>", "<|pad|>", "<|instruction|>", "<|response|>"]
FAST = dict(verbose=False)
TEXTS = ["The river flows north to the sea.", "Plants turn sunlight into sugar and oxygen.",
         "def add(a, b):\n    return a + b\n"]


@pytest.fixture(scope="module")
def tok():
    """The same kind of tokenizer the PyCoder project uses: byte-level BPE with its special tokens."""
    return build_raw_tokenizer(SPECIALS)


def tiny_gpt(vocab, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, block_size=128, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    return GPT(cfg).eval()


def manual_perplexity(logits_fn, tok, texts):
    nll, n = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([tok.encode(t).ids])
            logp = torch.log_softmax(logits_fn(ids)[0, :-1].float(), dim=-1)
            nll -= logp.gather(1, ids[0, 1:, None]).sum().item()
            n += ids.shape[1] - 1
    return math.exp(nll / n)


def ppl(model, tok, **kw):
    return llmreport.analyze(model, tok, checks="perplexity", prompts={"perplexity": TEXTS}, **FAST, **kw)[
        "perplexity"].metrics["perplexity"]


# ============================================================== structure


def test_scratch_gpt_structure(tok):
    model = tiny_gpt(tok.get_vocab_size())
    a = llmreport.analyze(model, tok, checks="architecture", **FAST)["architecture"]
    s, m = a.details["structure"], a.metrics
    assert (m["layers"], m["hidden_size"], m["attention_heads"], m["head_dim"]) == (2, 64, 4, 16)
    assert m["kv_heads"] == 4 and s["attention_type"] == "multi-head (MHA)"
    assert m["intermediate_size"] == 256 and m["vocab_size"] == tok.get_vocab_size()
    assert m["max_context"] == 128
    assert s["position_encoding"] == "learned absolute (128 positions)"
    assert s["norm_type"] == "LayerNorm (no bias)" and s["norm_placement"] == "pre-norm"
    assert s["activation"] == "GELU" and s["tied_embeddings"] is True
    assert s["family"] == "decoder-only transformer"
    assert m["total_params"] == sum(p.numel() for p in model.parameters())
    assert m["params_without_position_embeddings"] == model.num_params()
    assert sum(a.details["param_breakdown"].values()) == m["total_params"]
    assert a.status == "ok"


def test_scratch_gpt_full_size_matches_published_count():
    """PyCoder's spec sheet: 142,627,840 parameters (model.num_params(), without position embeddings)."""
    with torch.device("meta"):
        model = GPT(GPTConfig())
    a = ArchitectureAnalyzer().run(model, None, RunConfig())
    assert a.metrics["params_without_position_embeddings"] == 142_627_840
    assert a.metrics["total_params"] == 142_627_840 + 512 * 1024
    assert a.metrics["layers"] == 10 and a.metrics["attention_heads"] == 16 and a.metrics["head_dim"] == 64
    assert a.metrics["intermediate_size"] == 4096 and a.metrics["max_context"] == 512
    # K and V, 10 layers, 16 heads, 64 dims, 4 bytes (float32)
    assert a.metrics["kv_cache_bytes_per_token"] == 2 * 10 * 16 * 64 * 4
    b = a.details["param_breakdown"]
    assert b["embeddings"] == 16384 * 1024 + 512 * 1024
    assert b["attention"] == 10 * 4 * 1024 * 1024 and b["mlp"] == 10 * 2 * 1024 * 4096
    assert b["norm"] == (2 * 10 + 1) * 1024 and "output_head" not in b  # the head is tied to the embedding


def test_llama_style_structure(tok):
    torch.manual_seed(0)
    model = LlamaStyle(LlamaArgs(vocab_size=tok.get_vocab_size())).eval()
    a = llmreport.analyze(model, tok, checks="architecture", **FAST)["architecture"]
    s, m = a.details["structure"], a.metrics
    assert (m["layers"], m["hidden_size"], m["attention_heads"], m["kv_heads"], m["head_dim"]) == (3, 64, 8, 2, 8)
    assert s["attention_type"].startswith("grouped-query") and "4 query heads per KV head" in s["attention_type"]
    assert s["position_encoding"] == "rotary (RoPE)"
    assert s["norm_type"] == "RMSNorm" and s["norm_placement"] == "pre-norm"
    assert s["mlp_type"] == "gated (GLU-style)" and s["activation"] == "SwiGLU (gated SiLU)"
    assert m["intermediate_size"] == 160 and m["max_context"] == 128
    assert s["tied_embeddings"] is False
    b = a.details["param_breakdown"]
    v = tok.get_vocab_size()
    assert b["embeddings"] == v * 64 and b["output_head"] == v * 64
    assert b["attention"] == 3 * (64 * 64 * 2 + 64 * 16 * 2)  # wq, wo full size; wk, wv 2 heads x 8 dims
    assert b["mlp"] == 3 * 3 * 64 * 160
    assert b["norm"] == 64 * (2 * 3 + 1)
    assert "other" not in b
    assert m["kv_cache_bytes_per_token"] == 2 * 3 * 2 * 8 * 4


def test_recurrent_model(tok):
    torch.manual_seed(0)
    model = LSTMLM(tok.get_vocab_size()).eval()
    r = llmreport.analyze(model, tok, checks="architecture,perplexity,repetition", max_new_tokens=8, **FAST)
    a = r["architecture"]
    assert a.details["structure"]["family"] == "recurrent (LSTM)"
    assert a.metrics["layers"] == 2 and a.metrics["hidden_size"] == 64
    assert "max_context" not in a.metrics and "no fixed context window" in a.summary
    assert a.details["param_breakdown"]["recurrent"] == sum(p.numel() for p in model.rnn.parameters())
    assert r["perplexity"].metrics["perplexity"] == pytest.approx(
        manual_perplexity(lambda ids: model(ids)[0], tok, TEXTS[:0] + load_texts()), rel=1e-4)
    assert r["repetition"].status in ("ok", "warning")


def load_texts():
    from llmreport._utils import load_data

    return [t["text"] for t in load_data("perplexity")["texts"]]


def test_seq_first_post_norm_transformer(tok):
    torch.manual_seed(0)
    model = SeqFirstTransformer(tok.get_vocab_size()).eval()
    a = llmreport.analyze(model, tok, checks="architecture", **FAST)["architecture"]
    s = a.details["structure"]
    assert s["norm_placement"] == "post-norm"  # nn.TransformerEncoderLayer defaults to norm_first=False
    assert s["activation"] == "GELU" and a.metrics["attention_heads"] == 4 and a.metrics["max_context"] == 96
    assert s["attention_type"] == "multi-head (MHA)"


# ============================================================== accuracy of the numbers


@pytest.mark.parametrize("kind", ["gpt", "llama", "lstm", "seq_first"])
def test_perplexity_matches_manual_computation(tok, kind):
    v = tok.get_vocab_size()
    torch.manual_seed(1)
    model = {"gpt": lambda: tiny_gpt(v, 1), "llama": lambda: LlamaStyle(LlamaArgs(vocab_size=v)),
             "lstm": lambda: LSTMLM(v), "seq_first": lambda: SeqFirstTransformer(v)}[kind]().eval()
    extract = {"gpt": lambda ids: model(ids)[0], "llama": model, "lstm": lambda ids: model(ids)[0],
               "seq_first": lambda ids: model(ids)["logits"].transpose(0, 1)}[kind]
    assert ppl(model, tok) == pytest.approx(manual_perplexity(extract, tok, TEXTS), rel=1e-4)


def test_last_position_only_models_are_scored_exactly(tok):
    """nanoGPT returns only the last position without targets; the score must not change."""
    inner = tiny_gpt(tok.get_vocab_size(), 2)
    full = ppl(inner, tok)
    assert ppl(LastOnlyGPT(inner), tok) == pytest.approx(full, rel=1e-5)

    class NoTargets(nn.Module):  # only ever returns the last position: scored prefix by prefix
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, idx):
            return self.inner(idx)[0][:, -1, :]

    lm = LanguageModel(NoTargets(), tok)
    lm.probe()
    assert lm.full_logits is False
    assert ppl(NoTargets(), tok) == pytest.approx(full, rel=1e-5)


def test_uniform_model_has_perplexity_equal_to_vocab_size(tok):
    v = tok.get_vocab_size()
    assert ppl(uniform_logits_model(v), tok, context_length=256) == pytest.approx(v, rel=1e-6)


def _to_hf_gpt2(gpt):
    from transformers import GPT2Config, GPT2LMHeadModel

    c = gpt.config
    hf = GPT2LMHeadModel(GPT2Config(
        vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd, n_layer=c.n_layer, n_head=c.n_head,
        n_inner=c.n_embd * c.ffn_mult, activation_function="gelu", resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        layer_norm_epsilon=1e-5, tie_word_embeddings=True)).eval()
    sd, dst = gpt.state_dict(), hf.state_dict()
    with torch.no_grad():
        dst["transformer.wte.weight"].copy_(sd["tok_emb.weight"])
        dst["transformer.wpe.weight"].copy_(sd["pos_emb.weight"])
        for i in range(c.n_layer):
            s, h = f"blocks.{i}.", f"transformer.h.{i}."
            for a, b in (("ln1", "ln_1"), ("ln2", "ln_2")):
                dst[h + b + ".weight"].copy_(sd[s + a + ".weight"])
                dst[h + b + ".bias"].zero_()
            for a, b in (("attn.qkv_proj", "attn.c_attn"), ("attn.out_proj", "attn.c_proj"),
                         ("mlp.fc_in", "mlp.c_fc"), ("mlp.fc_out", "mlp.c_proj")):
                dst[h + b + ".weight"].copy_(sd[s + a + ".weight"].t())
                dst[h + b + ".bias"].zero_()
        dst["transformer.ln_f.weight"].copy_(sd["ln_f.weight"])
        dst["transformer.ln_f.bias"].zero_()
    return hf


def test_scratch_gpt_gives_same_results_as_its_huggingface_twin(tok):
    """Same weights, two implementations: every number and every generated token must agree."""
    gpt = tiny_gpt(tok.get_vocab_size(), 3)
    hf = _to_hf_gpt2(gpt)
    ids = torch.tensor([tok.encode(TEXTS[0]).ids])
    assert torch.allclose(gpt(ids)[0], hf(ids).logits, atol=1e-4)

    assert ppl(gpt, tok) == pytest.approx(ppl(hf, tok), rel=1e-4)
    for prompt in ("The capital of France is", "def add(a, b):", "Once upon a time"):
        # built-in greedy loop (scratch model) vs Hugging Face generate()
        assert generate(gpt, tok, prompt, 20) == generate(hf, tok, prompt, 20)
    a_gpt = llmreport.analyze(gpt, tok, checks="architecture", **FAST)["architecture"].metrics
    a_hf = llmreport.analyze(hf, tok, checks="architecture", **FAST)["architecture"].metrics
    for key in ("layers", "hidden_size", "attention_heads", "kv_heads", "head_dim", "intermediate_size",
                "max_context", "kv_cache_bytes_per_token"):
        assert a_gpt[key] == a_hf[key], key
    # HF adds zero biases, so it has exactly that many more parameters
    extra = sum(p.numel() for n, p in hf.named_parameters() if n.endswith(".bias"))
    assert a_hf["total_params"] == a_gpt["total_params"] + extra


class SlowScratch(nn.Module):
    """From-scratch model where every forward pass takes at least 20 ms."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, idx):
        time.sleep(0.02)
        return self.inner(idx)[0]


def test_performance_measures_known_speed_without_generate(tok):
    model = SlowScratch(tiny_gpt(tok.get_vocab_size()))
    r = llmreport.analyze(model, tok, checks="performance", verbose=False,
                          options={"performance": {"prompt_lengths": [16], "repeats": 1, "new_tokens": 10}})[
        "performance"]
    assert 20 <= r.metrics["ttft_ms"] < 150
    assert 20 < r.metrics["decode_tokens_per_s"] <= 51
    assert any("no KV cache" in n for n in r.notes)


# ============================================================== generation and prompts


def _scripted(tok, respond):
    return ScriptedGPT(tok, respond, tok.token_to_id("<|endoftext|>"), tok.get_vocab_size())


def test_generation_stops_at_end_of_text(tok):
    model = _scripted(tok, lambda p: "hello world")
    assert generate(model, tok, "Say hi", 40) == "hello world"


def test_prompt_template_wraps_prompts_with_special_tokens(tok):
    model = _scripted(tok, lambda p: "Sure, here it is.")
    template = "<|instruction|>{prompt}<|response|>"
    r = llmreport.analyze(model, tok, checks="repetition", prompt_template=template, max_new_tokens=30, **FAST)
    prompts = model.seen[1:]  # the first call is llmreport checking how to call the model
    assert len(prompts) == 5
    assert all(t.startswith("<|instruction|>") and t.endswith("<|response|>") for t in prompts)
    ids = TokenizerAdapter(tok).encode(template.replace("{prompt}", "hi"))
    assert ids[0] == tok.token_to_id("<|instruction|>") and ids[-1] == tok.token_to_id("<|response|>")
    assert r["repetition"].details["samples"][0]["output"] == "Sure, here it is."


def test_forward_override_for_models_with_extra_arguments(tok):
    torch.manual_seed(0)
    model = NeedsStartPos(LlamaArgs(vocab_size=tok.get_vocab_size())).eval()
    with pytest.raises(TypeError, match="forward=lambda ids"):
        llmreport.analyze(model, tok, checks="perplexity", **FAST)
    expected = manual_perplexity(lambda ids: model(ids, 0), tok, TEXTS)
    assert ppl(model, tok, forward=lambda ids: model(ids, 0)) == pytest.approx(expected, rel=1e-4)


def test_wrong_tokenizer_is_flagged(tok):
    torch.manual_seed(0)
    small = GPT(GPTConfig(vocab_size=100, block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()
    a = llmreport.analyze(small, tok, checks="architecture", **FAST)["architecture"]
    assert a.status == "warning" and any("wrong tokenizer" in n for n in a.notes)


# ============================================================== code check

SOLUTIONS = {
    "add": "def add(a, b):\n    return a + b\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
    "factorial": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
    "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
    "find_max": "def find_max(numbers):\n    return max(numbers)\n",
}


def _task(prompt):
    return next((name for name in SOLUTIONS if name + "(" in prompt), "add")


def _body(code):
    return code.split("\n", 1)[1]


def test_code_check_instruction_style(tok):
    template = "<|instruction|>{prompt}<|response|>"
    good = _scripted(tok, lambda p: "Here you go:\n```python\n" + SOLUTIONS[_task(p)] + "```\nThis works.")
    r = llmreport.analyze(good, tok, checks="code", prompt_template=template, max_new_tokens=300, **FAST)["code"]
    assert r.details["style"] == "instruction"
    assert r.metrics["pass_rate"] == 1.0 and r.metrics["syntax_valid_rate"] == 1.0 and r.status == "ok"

    wrong = _scripted(tok, lambda p: SOLUTIONS[_task(p)].replace("return", "return None and"))
    r = llmreport.analyze(wrong, tok, checks="code", prompt_template=template, max_new_tokens=300, **FAST)["code"]
    assert r.metrics["pass_rate"] == 0.0 and r.metrics["syntax_valid_rate"] == 1.0 and r.status == "warning"

    broken = _scripted(tok, lambda p: "def oops(:\n    return")
    r = llmreport.analyze(broken, tok, checks="code", prompt_template=template, max_new_tokens=300, **FAST)["code"]
    assert r.metrics["syntax_valid_rate"] == 0.0 and r.metrics["pass_rate"] == 0.0


def test_code_check_completion_style(tok):
    # A base model continues the signature + docstring; anything after the function is cut off.
    model = _scripted(tok, lambda p: _body(SOLUTIONS[_task(p)]) + "\n\nprint('extra stuff')\nclass X: pass\n")
    r = llmreport.analyze(model, tok, checks="code", max_new_tokens=300, **FAST)["code"]
    assert r.details["style"] == "completion"
    assert r.metrics["pass_rate"] == 1.0

    silent = _scripted(tok, lambda p: "\nprint('no body')")  # signature + docstring alone must not count
    r = llmreport.analyze(silent, tok, checks="code", max_new_tokens=100, **FAST)["code"]
    assert r.metrics["syntax_valid_rate"] == 0.0 and r.metrics["pass_rate"] == 0.0


def test_code_helpers():
    text = "Sure!\n```python\nimport math\n\ndef area(r):\n    return math.pi * r ** 2\n```\nDone."
    code = extract_function(text, "area")
    assert code.startswith("import math") and "def area(r):" in code and "Done" not in code
    assert truncate_body("    return 1\n\ndef other():\n    pass") == "    return 1\n"
    assert run_tests("def f():\n    return 1\n", ["assert f() == 1"]) == (True, "")
    ok, err = run_tests("def f():\n    while True:\n        pass\n", ["f()"], timeout=2)
    assert not ok and err == "timed out"


# ============================================================== tokenizers


class CharTokenizer:
    """The simplest possible tokenizer: one id per character, with an end-of-text token."""

    def __init__(self, alphabet):
        self.chars = ["<|endoftext|>"] + sorted(set(alphabet))
        self.stoi = {c: i for i, c in enumerate(self.chars)}  # the usual name in char-level projects

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids if i > 0)

    @property
    def vocab_size(self):
        return len(self.chars)


def _sentencepiece(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(load_texts() * 5), encoding="utf-8")
    prefix = str(tmp_path / "sp")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=120, model_type="bpe",
                                   minloglevel=2)
    return prefix + ".model"


def _tiktoken():
    tiktoken = pytest.importorskip("tiktoken")
    ranks = {bytes([i]): i for i in range(256)}
    return tiktoken.Encoding("bytes", pat_str=r"\S+|\s+", mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": 256})


@pytest.mark.parametrize("kind", ["tokenizers", "tokenizer_json", "huggingface", "sentencepiece", "tiktoken", "chars"])
def test_every_kind_of_tokenizer(tok, tmp_path, kind):
    from transformers import PreTrainedTokenizerFast

    if kind == "tokenizers":
        t = tok
    elif kind == "tokenizer_json":
        tok.save(str(tmp_path / "tokenizer.json"))
        t = str(tmp_path / "tokenizer.json")
    elif kind == "huggingface":
        t = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>")
    elif kind == "sentencepiece":
        t = _sentencepiece(tmp_path)
    elif kind == "tiktoken":
        t = _tiktoken()
    else:
        t = CharTokenizer("".join(load_texts()))
    adapter = TokenizerAdapter(t)
    vocab = adapter.vocab_size
    assert vocab and vocab > 20
    text = "The river flows north."
    assert adapter.decode(adapter.encode(text)) .strip() == text
    assert adapter.eos_id is not None and adapter.eos_id in adapter.stop_ids
    # A uniform model's perplexity is exactly its vocabulary size, whatever the tokenizer.
    r = llmreport.analyze(uniform_logits_model(vocab), t, checks="perplexity", context_length=512,
                          prompts={"perplexity": TEXTS[:2]}, **FAST)["perplexity"]
    assert r.metrics["perplexity"] == pytest.approx(vocab, rel=1e-6)


# ============================================================== loading checkpoints


def _save(obj, tmp_path, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(obj, path)
    return str(path)


def test_load_checkpoint_like_pycoder_saves_it(tok, tmp_path):
    gpt = tiny_gpt(tok.get_vocab_size(), 4)
    path = _save({"model": gpt.state_dict(), "config": gpt.config, "step": 60000}, tmp_path)
    tok.save(str(tmp_path / "tokenizer.json"))
    model, t = llmreport.load_checkpoint(path, GPT, tokenizer=str(tmp_path / "tokenizer.json"), device="cpu")
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(model(ids)[0], gpt(ids)[0])
    # analyze() straight from the file
    r = llmreport.analyze(path, str(tmp_path / "tokenizer.json"), model_class=GPT, device="cpu",
                          checks="architecture,perplexity", prompts={"perplexity": TEXTS}, **FAST)
    assert r["architecture"].metrics["total_params"] == sum(p.numel() for p in gpt.parameters())
    assert r["perplexity"].metrics["perplexity"] == pytest.approx(ppl(gpt, tok), rel=1e-6)


def test_load_checkpoint_variants(tok, tmp_path):
    v = tok.get_vocab_size()
    torch.manual_seed(5)
    llama = LlamaStyle(LlamaArgs(vocab_size=v)).eval()
    ids = torch.tensor([[5, 6, 7, 8]])
    want = llama(ids)

    # weights only + config as a plain dict: the dataclass is found from the type annotation
    cfg = {"dim": 64, "n_layers": 3, "n_heads": 8, "n_kv_heads": 2, "vocab_size": v, "hidden_dim": 160,
           "max_seq_len": 128}
    m, _ = llmreport.load_checkpoint(_save(llama.state_dict(), tmp_path), LlamaStyle, config=cfg, device="cpu")
    assert torch.allclose(m(ids), want)
    # torch.compile prefix and a "state_dict" key (Lightning style)
    compiled = {"state_dict": {"_orig_mod." + k: t for k, t in llama.state_dict().items()}, "hparams": cfg}
    m, _ = llmreport.load_checkpoint(_save(compiled, tmp_path, "c.pt"), LlamaStyle, device="cpu")
    assert torch.allclose(m(ids), want)
    # the whole model object: no class needed
    m, _ = llmreport.load_checkpoint(_save(llama, tmp_path, "whole.pt"), device="cpu")
    assert torch.allclose(m(ids), want)
    # wrong class -> a clear error
    with pytest.raises((ValueError, TypeError)):
        llmreport.load_checkpoint(_save(llama.state_dict(), tmp_path, "w.pt"), GPT, device="cpu")
    with pytest.raises(ValueError, match="model_class"):
        llmreport.load_checkpoint(_save(llama.state_dict(), tmp_path, "x.pt"))


def test_torchscript_model(tok, tmp_path):
    gpt = tiny_gpt(tok.get_vocab_size(), 6)

    class LogitsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt = gpt

        def forward(self, idx):
            return self.gpt(idx)[0]

    traced = torch.jit.trace(LogitsOnly().eval(), torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), check_trace=False)
    path = str(tmp_path / "traced.pt")
    traced.save(path)
    model, _ = llmreport.load_checkpoint(path, device="cpu")
    assert ppl(model, tok, context_length=128) == pytest.approx(ppl(gpt, tok), rel=1e-4)


def test_int8_dynamic_quantized_checkpoint(tok, tmp_path):
    """Like PyCoder's export/model_int8.pt: quantize_dynamic, then save {"model": ..., "config": ...}."""
    gpt = tiny_gpt(tok.get_vocab_size(), 7)
    q = torch.ao.quantization.quantize_dynamic(gpt, {nn.Linear}, dtype=torch.qint8)
    path = _save({"model": q.state_dict(), "config": gpt.config}, tmp_path, "model_int8.pt")
    model, _ = llmreport.load_checkpoint(path, GPT, device="cpu")
    ids = torch.tensor([tok.encode(TEXTS[0]).ids])
    assert torch.allclose(model(ids)[0], q(ids)[0])
    r = llmreport.analyze(model, tok, checks="architecture,perplexity", prompts={"perplexity": TEXTS}, **FAST)
    a = r["architecture"]
    assert a.metrics["layers"] == 2 and a.metrics["attention_heads"] == 4 and "qint8" in a.details["dtypes"]
    # every weight is counted: float ones plus the int8 weights hidden inside the quantized layers
    fp = sum(p.numel() for p in model.parameters())
    packed = sum(m.weight().numel() + (m.bias().numel() if m.bias() is not None else 0)
                 for m in model.modules() if hasattr(m, "_packed_params") and callable(getattr(m, "weight", None)))
    assert a.metrics["total_params"] == fp + packed
    assert r["perplexity"].metrics["perplexity"] == pytest.approx(
        manual_perplexity(lambda i: q(i)[0], tok, TEXTS), rel=1e-4)


def test_cli_with_checkpoint(tok, tmp_path, monkeypatch):
    from llmreport.cli import main

    gpt = tiny_gpt(tok.get_vocab_size(), 8)
    path = _save({"model": gpt.state_dict(), "config": gpt.config}, tmp_path)
    tok.save(str(tmp_path / "tokenizer.json"))
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), "scratch"))
    out = tmp_path / "r.json"
    code = main([path, "--model-class", "gpt_model:GPT", "--tokenizer", str(tmp_path / "tokenizer.json"),
                 "--template", "<|instruction|>{prompt}<|response|>", "--checks", "architecture,repetition",
                 "--device", "cpu", "--json", str(out)])
    assert code == 0
    data = Report_from(out)
    assert data["architecture"].metrics["layers"] == 2
    assert data.metadata["prompt_template"] == "<|instruction|>{prompt}<|response|>"


def Report_from(path):
    return llmreport.Report.from_json(str(path)) if hasattr(llmreport.Report, "from_json") else \
        llmreport.Report.load(str(path))


# ============================================================== Hugging Face architecture zoo


def _hf(name, vocab):
    import transformers as T

    common = dict(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, max_position_embeddings=256)
    return {
        "llama": lambda: T.LlamaConfig(intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, **common),
        "qwen2": lambda: T.Qwen2Config(intermediate_size=128, num_attention_heads=4, num_key_value_heads=1, **common),
        "gpt_neox": lambda: T.GPTNeoXConfig(intermediate_size=128, num_attention_heads=4, **common),
        "gemma": lambda: T.GemmaConfig(intermediate_size=128, num_attention_heads=4, num_key_value_heads=1, head_dim=16,
                                       **common),
        "mixtral": lambda: T.MixtralConfig(intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
                                           num_local_experts=4, num_experts_per_tok=2, **common),
        "mamba": lambda: T.MambaConfig(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, state_size=8),
    }[name]()


@pytest.mark.parametrize("name, attention, mlp, activation", [
    ("llama", "grouped-query (GQA, 2 query heads per KV head)", "gated (GLU-style)", "SwiGLU (gated SiLU)"),
    ("qwen2", "multi-query (MQA)", "gated (GLU-style)", "SwiGLU (gated SiLU)"),
    ("gpt_neox", "multi-head (MHA)", "standard (up, activation, down)", "GELU"),
    ("gemma", "multi-query (MQA)", "gated (GLU-style)", "GeGLU (gated GELU (tanh approximation))"),
    ("mixtral", "grouped-query (GQA, 2 query heads per KV head)", "mixture of experts (4 experts)", "SiLU"),
    ("mamba", None, None, "SiLU"),
])
def test_huggingface_architectures(tokenizer, name, attention, mlp, activation):
    import transformers as T

    torch.manual_seed(0)
    model = T.AutoModelForCausalLM.from_config(_hf(name, len(tokenizer))).eval()
    r = llmreport.analyze(model, tokenizer, checks="architecture,perplexity,repetition", max_new_tokens=6, **FAST)
    a = r["architecture"]
    s = a.details["structure"]
    assert s.get("attention_type") == attention and s.get("mlp_type") == mlp and s.get("activation") == activation
    assert a.metrics["total_params"] == sum(p.numel() for p in model.parameters())
    assert sum(a.details["param_breakdown"].values()) == a.metrics["total_params"]
    assert "other" not in a.details["param_breakdown"]
    assert s["norm_placement"] == "pre-norm" and a.metrics["layers"] == 2
    if name == "mamba":
        assert s["family"] == "state-space / linear-recurrent" and "attention" not in a.details["param_breakdown"]
    assert all(x.status != "error" for x in r)


def test_empty_answers_are_not_valid_code(tok):
    silent = _scripted(tok, lambda p: "")
    r = llmreport.analyze(silent, tok, checks="code", prompt_template="<|instruction|>{prompt}<|response|>",
                          max_new_tokens=20, **FAST)["code"]
    assert r.metrics["syntax_valid_rate"] == 0.0 and r.metrics["pass_rate"] == 0.0


def test_report_formats_new_metrics():
    from llmreport.report import format_value

    assert format_value("params_without_position_embeddings", 142_627_840) == "142.63M"
    assert format_value("kv_cache_bytes_per_token", 81920) == "80.0 KB"
    assert format_value("pass_rate", 0.5) == "50.0%"
