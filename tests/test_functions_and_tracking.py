"""The numpy-style functions, the live training Tracker, and more kinds of models.

Everything is checked against a known answer: hand-computed numbers, injected faults
(NaN, gradient explosions, dead layers) that must be caught, and healthy runs that must
raise no false alarms.
"""

import json
import math
import os
import sys

import pytest
import torch
from conftest import _corpus, build_raw_tokenizer
from torch import nn

import llmreport as lr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scratch"))
from gpt_model import GPT, GPTConfig  # noqa: E402
from models import LlamaArgs, LlamaStyle, uniform_logits_model  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|endoffile|>", "<|pad|>", "<|instruction|>", "<|response|>"]
TEXTS = ["The river flows north to the sea.", "Plants turn sunlight into sugar and oxygen."]


@pytest.fixture(scope="module")
def tok():
    return build_raw_tokenizer(SPECIALS)


def tiny_gpt(vocab, seed=0, block=64):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=4, n_embd=64, dropout=0.0))


# ================================================================== functions


def test_params_and_breakdown(tok):
    m = tiny_gpt(tok.get_vocab_size())
    total = sum(p.numel() for p in m.parameters())
    assert lr.params(m) == total
    assert sum(lr.param_breakdown(m).values()) == total
    for p in m.blocks[0].parameters():
        p.requires_grad = False
    frozen = sum(p.numel() for p in m.blocks[0].parameters())
    assert lr.params(m, trainable_only=True) == total - frozen
    with torch.device("meta"):
        full = GPT(GPTConfig())
    assert lr.params(full) == 142_627_840 + 512 * 1024


def test_memory(tok):
    m = tiny_gpt(tok.get_vocab_size())
    n = lr.params(m)
    mem = lr.memory(m, tok, batch_size=4)
    assert mem["fp32_bytes"] == 4 * n and mem["fp16_bytes"] == 2 * n and mem["int4_bytes"] == n // 2
    assert mem["weights_bytes"] == 4 * n and mem["training_adamw_bytes"] == 16 * n
    per_token = 2 * 2 * 4 * 16 * 4  # K and V, 2 layers, 4 heads, 16 dims, float32
    assert mem["kv_cache_bytes_per_token"] == per_token
    assert mem["kv_cache_bytes"] == per_token * 64 * 4
    assert mem["inference_bytes"] == mem["weights_bytes"] + mem["kv_cache_bytes"]


def test_info_on_a_model_that_is_training(tok):
    m = tiny_gpt(tok.get_vocab_size())
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, 100, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    opt.step()
    info = lr.info(m, tok, optimizer=opt)
    assert m.training  # info() must not switch the model to eval mode
    assert info.mode == "training" and info.layers == 2 and info.attention_heads == 4
    assert info.optimizer == "AdamW" and info.learning_rate == pytest.approx(3e-4)
    assert info.optimizer_state_bytes >= 8 * lr.params(m)  # two float32 moments per weight
    assert info.grad_norm == pytest.approx(lr.grad_norm(m)) and info.grad_norm > 0
    assert info.health_status == "ok" and info.tokenizer_vocab == tok.get_vocab_size()
    assert "Training state" in repr(info) and "<table" in info._repr_html_()


def test_perplexity_generate_and_predict_next(tok):
    v = tok.get_vocab_size()
    uniform = uniform_logits_model(v)
    assert lr.perplexity(uniform, tok, TEXTS) == pytest.approx(v, rel=1e-6)
    top = lr.predict_next(uniform, tok, "Hello", k=3)
    assert len(top) == 3 and all(p == pytest.approx(1 / v, rel=1e-4) for _, p in top)

    m = tiny_gpt(v, 3)
    m.eval()
    full = lr.analyze(m, tok, checks="perplexity", prompts={"perplexity": TEXTS}, verbose=False)["perplexity"]
    assert lr.perplexity(m, tok, TEXTS) == pytest.approx(full.metrics["perplexity"], rel=1e-5)
    d = lr.perplexity(m, tok, TEXTS, details=True)
    assert d["tokens"] == full.metrics["tokens_scored"]
    ids = torch.tensor([tok.encode("The river flows").ids])
    best = int(torch.argmax(m(ids)[0][0, -1]))
    assert lr.predict_next(m, tok, "The river flows", 1)[0][0] == tok.decode([best], skip_special_tokens=False)
    assert isinstance(lr.generate(m, tok, "The river", 8), str)


def test_text_stats():
    s = lr.text_stats("one two three one two three")
    assert s["words"] == 6 and s["distinct_1"] == 0.5 and s["repeated_3gram_rate"] == pytest.approx(0.25)  # 1 of 4 3-grams repeats


def test_check_and_speed(tok):
    m = tiny_gpt(tok.get_vocab_size())
    r = lr.check("repetition", m, tok, max_new_tokens=8)
    assert r.name == "repetition"
    sp = lr.speed(m, tok, prompt_tokens=16, new_tokens=4, repeats=1)
    assert sp["decode_tokens_per_s"] > 0 and sp["ttft_ms"] > 0


# ================================================================== health


def test_health_finds_nothing_wrong_in_healthy_models(tok):
    import transformers as T

    models = [tiny_gpt(tok.get_vocab_size()), LlamaStyle(LlamaArgs()),
              T.GPT2LMHeadModel(T.GPT2Config(vocab_size=400, n_embd=64, n_layer=4, n_head=4)),
              T.OPTForCausalLM(T.OPTConfig(vocab_size=400, hidden_size=64, ffn_dim=128, num_hidden_layers=4,
                                           num_attention_heads=4, word_embed_proj_dim=64)),
              T.BertForMaskedLM(T.BertConfig(vocab_size=400, hidden_size=64, num_hidden_layers=4,
                                             num_attention_heads=4, intermediate_size=128))]
    for m in models:
        h = lr.health(m)
        assert h.ok and not h.issues, (type(m).__name__, h.issues)


def test_health_catches_broken_weights(tok):
    m = tiny_gpt(tok.get_vocab_size())
    with torch.no_grad():
        m.blocks[0].mlp.fc_in.weight[0, 0] = float("nan")
        m.blocks[1].ln1.weight.zero_()
        m.blocks[1].mlp.fc_out.weight.zero_()
        m.blocks[0].attn.qkv_proj.weight[:30] = 0  # 30 of 192 rows dead
    h = lr.health(m)
    assert h.status == "critical" and not h.ok
    problems = " ".join(str(i) for i in h.issues)
    assert "NaN" in problems and "blocks.0.mlp.fc_in.weight" in problems
    assert "blocks.1.ln1.weight" in problems and "blocks.1.mlp.fc_out.weight" in problems
    assert "blocks.0.attn.qkv_proj.weight (30/192 rows)" in problems


def test_health_and_info_on_int8_quantized_model(tok):
    """Like PyCoder's export/model_int8.pt: quantized weights must be checked, not crash."""
    m = tiny_gpt(tok.get_vocab_size()).eval()
    q = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    h = lr.health(q)
    assert h.ok and h.totals["tensors"] > 0
    assert "qint8" in lr.info(q, tok).dtype or lr.info(q, tok).health_status == "ok"
    with torch.no_grad():
        q.blocks[0].mlp.fc_out.set_weight_bias(torch.quantize_per_tensor(
            torch.zeros(64, 256), 0.1, 0, torch.qint8), None)
    assert any("entirely zero" in i.problem for i in lr.health(q).issues)


def test_health_checks_gradients(tok):
    m = tiny_gpt(tok.get_vocab_size())
    m.extra = nn.Linear(4, 4)  # never used in forward: gets no gradient
    x = torch.randint(0, 100, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    h = lr.health(m)
    assert h.status == "ok" and any("received no gradient" in i.problem for i in h.issues)
    assert h.totals["grad_norm"] == pytest.approx(lr.grad_norm(m))
    m.blocks[0].mlp.fc_in.weight.grad[0, 0] = float("inf")
    assert lr.health(m).status == "critical"
    assert math.isnan(lr.grad_norm(m))


# ================================================================== tracker


def _quiet(tmp_path, name="run", **kw):
    return lr.Tracker(log_dir=str(tmp_path / name), print_every=0, dashboard_every=0, **kw)


def _alerts(t, key=None):
    return [a for a in t.alerts if key is None or a["key"] == key]


def test_real_training_loop_is_tracked_without_false_alarms(tok, tmp_path, capsys):
    v = tok.get_vocab_size()
    m = tiny_gpt(v, 1)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    ids = torch.tensor(tok.encode(" ".join(t for t in _corpus()[:60] if isinstance(t, str))).ids)
    t = lr.Tracker(m, tok, log_dir=str(tmp_path / "run"), eval_texts=TEXTS, eval_every=50,
                   sample_prompts=["The river"], total_steps=200, print_every=0, dashboard_every=0)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        i = torch.randint(0, len(ids) - 33, (8,), generator=g)
        x = torch.stack([ids[j:j + 32] for j in i])
        y = torch.stack([ids[j + 1:j + 33] for j in i])
        _, loss = m(x, y)
        loss.backward()
        t.step(loss, optimizer=opt, tokens=x.numel())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    t.close()
    assert m.training  # evaluation restored training mode
    serious = [a for a in t.alerts if a["severity"] in ("critical", "warning")]
    assert serious == []
    rows = [json.loads(line) for line in open(tmp_path / "run" / "metrics.jsonl")]
    assert len(rows) == 200 and rows[-1]["step"] == 200
    assert all(r["lr"] == pytest.approx(2e-3) for r in rows) and all(r["grad_norm"] > 0 for r in rows)
    assert rows[-1]["loss_avg"] < rows[0]["loss"]  # it learned
    evals = [e for e in t.events if e["kind"] == "eval"]
    assert [e["step"] for e in evals] == [50, 100, 150, 200]
    assert evals[-1]["eval_perplexity"] < evals[0]["eval_perplexity"]
    s = t.summary()
    assert s["steps"] == 200 and s["tokens_seen"] == 200 * 8 * 32
    page = open(tmp_path / "run" / "dashboard.html").read()
    assert "Training loss" in page and "Evaluation perplexity" in page and "http-equiv" not in page  # finished
    run = lr.load_run(str(tmp_path / "run"))
    assert run.summary()["best_loss"] == s["best_loss"] and len(run.series("eval_perplexity")) == 4


def test_nan_loss_and_gradient_problems_are_reported(tmp_path):
    t = _quiet(tmp_path)
    for i in range(60):
        t.step(2.0 + 0.01 * (i % 3), grad_norm=1.0 + 0.01 * (i % 5))
    t.step(2.0, grad_norm=50.0)  # explosion
    t.step(float("nan"), grad_norm=float("nan"))
    for _ in range(3):
        t.step(2.0, grad_norm=0.0)  # vanishing
    t.close()
    assert _alerts(t, "grad_spike") and _alerts(t, "loss_nan")[0]["severity"] == "critical"
    assert _alerts(t, "grad_nan") and _alerts(t, "grad_vanish")
    assert "Critical at step 62" in open(tmp_path / "run" / "dashboard.html").read()


def test_loss_spike_detection(tmp_path):
    t = _quiet(tmp_path)
    for i in range(60):
        t.step(2.0 + 0.02 * ((i * 7) % 5))
    t.step(2.2)  # within normal variation: no alert
    assert not _alerts(t, "loss_spike")
    t.step(4.0)
    assert _alerts(t, "loss_spike")


def test_plateau_and_overfitting(tmp_path):
    t = _quiet(tmp_path, "flat")
    for i in range(450):
        t.step(3.0 + 0.001 * (i % 2))
    assert _alerts(t, "plateau")

    t = _quiet(tmp_path, "overfit")
    for i in range(300):
        t.step(3.0 - i * 0.005)
        if i % 100 == 99:
            t.log(val_loss={99: 2.0, 199: 2.1, 299: 2.3}[i])
    assert _alerts(t, "overfit")


def test_resume_continues_the_same_run(tmp_path):
    t = _quiet(tmp_path)
    for _ in range(10):
        t.step(3.0)
    t.close()
    t2 = _quiet(tmp_path)
    assert t2.resumed and t2.step(2.5)["step"] == 11
    t2.close()
    assert len(lr.load_run(str(tmp_path / "run")).rows) == 11


def test_speed_excludes_the_trackers_own_work(tok, tmp_path):
    import time

    m = tiny_gpt(tok.get_vocab_size())
    t = lr.Tracker(m, tok, log_dir=str(tmp_path / "run"), eval_texts=TEXTS * 20, eval_every=5, print_every=0,
                   sample_prompts=["The river"], sample_tokens=48)
    for _ in range(20):
        time.sleep(0.01)  # "training"
        t.step(2.0, tokens=100)
    t.close()
    times = [r["step_time"] for r in t.rows if r.get("step_time")]
    eval_time = min(e["eval_seconds"] for e in t.events if e["kind"] == "eval")
    assert max(times) < 0.01 + eval_time / 2  # evaluations are not counted as training time
    assert not _alerts(t, "slow")


def test_huggingface_trainer_callback(tokenizer, tmp_path):
    pytest.importorskip("accelerate")
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=64, n_embd=32, n_layer=2, n_head=2))
    data = [{"input_ids": torch.randint(0, len(tokenizer), (32,))} for _ in range(16)]
    for d in data:
        d["labels"] = d["input_ids"].clone()
    t = lr.Tracker(model, tokenizer, log_dir=str(tmp_path / "hf"), print_every=0)
    args = TrainingArguments(output_dir=str(tmp_path / "out"), max_steps=6, logging_steps=1,
                             per_device_train_batch_size=4, report_to=[], use_cpu=True, save_strategy="no")
    Trainer(model=model, args=args, train_dataset=data, callbacks=[t.hf_callback()]).train()
    assert t.closed and [r["step"] for r in t.rows] == [1, 2, 3, 4, 5, 6]
    assert all("lr" in r and "grad_norm" in r and math.isfinite(r["loss"]) for r in t.rows)


def test_watch_and_dashboard_commands(tmp_path, capsys):
    from llmreport.cli import main

    t = _quiet(tmp_path, total_steps=10)
    for i in range(10):
        t.step(3.0 - 0.1 * i, lr=1e-3)
    t.log(val_loss=2.5)
    t.close()
    assert main(["watch", str(tmp_path / "run"), "--once"]) == 0
    out = capsys.readouterr().out
    assert "step 10/10" in out and "[eval] step 10: val_loss 2.5" in out
    assert main(["dashboard", str(tmp_path / "run"), "--out", str(tmp_path / "d.html")]) == 0
    assert "Val loss" in open(tmp_path / "d.html").read()


def test_track_checkpoints_charts_progress(tok, tmp_path):
    v = tok.get_vocab_size()
    m = tiny_gpt(v, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.tensor(tok.encode(" ".join(TEXTS * 10)).ids)
    folder = tmp_path / "ckpt"
    folder.mkdir()
    for step in (100, 200, 300):
        for _ in range(40):
            i = torch.randint(0, len(ids) - 17, (8,))
            x = torch.stack([ids[j:j + 16] for j in i])
            y = torch.stack([ids[j + 1:j + 17] for j in i])
            _, loss = m(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.save({"model": m.state_dict(), "config": m.config, "step": step}, folder / f"step_{step:07d}.pt")
    run = lr.track_checkpoints(str(folder), GPT, tok, texts=TEXTS, sample_prompts=["The river"], verbose=False)
    ppl = run.series("eval_perplexity")
    assert [s for s, _ in ppl] == [100, 200, 300]
    assert ppl[-1][1] < ppl[0][1]
    assert os.path.exists(folder / "llmreport_progress" / "dashboard.html")


# ================================================================== more kinds of models


@pytest.fixture(scope="module")
def hf_tok():
    from transformers import PreTrainedTokenizerFast

    raw = build_raw_tokenizer(["<|endoftext|>", "<pad>", "<mask>", "<s>", "</s>"])
    return PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="</s>", pad_token="<pad>", mask_token="<mask>",
                                   bos_token="<s>")


def test_masked_language_model(hf_tok):
    import transformers as T

    torch.manual_seed(0)
    bert = T.BertForMaskedLM(T.BertConfig(vocab_size=len(hf_tok), hidden_size=64, num_hidden_layers=2,
                                          num_attention_heads=4, intermediate_size=128)).eval()
    r = lr.analyze(bert, hf_tok, checks="architecture,perplexity,repetition", verbose=False)
    assert r.metadata["model_kind"] == "masked"
    assert r["architecture"].details["structure"]["family"] == "encoder-only (bidirectional) transformer"
    assert r["repetition"].status == "skipped" and "doesn't generate text" in r["repetition"].summary
    assert r["perplexity"].summary.startswith("Pseudo-perplexity")
    # Pseudo-perplexity by hand: mask each non-special token and score it.
    text = TEXTS[0]
    ids = hf_tok(text, return_tensors="pt")["input_ids"]
    special = set(hf_tok.all_special_ids)
    nll, n = 0.0, 0
    with torch.no_grad():
        for t in range(1, ids.shape[1]):
            if int(ids[0, t]) in special:
                continue
            masked = ids.clone()
            masked[0, t] = hf_tok.mask_token_id
            nll -= float(torch.log_softmax(bert(masked).logits[0, t], -1)[ids[0, t]])
            n += 1
    assert lr.perplexity(bert, hf_tok, [text]) == pytest.approx(math.exp(nll / n), rel=1e-4)
    with pytest.raises(lr.adapters.NotSupportedByModel):
        lr.generate(bert, hf_tok, "hi")


def test_encoder_decoder_model(hf_tok):
    import transformers as T

    torch.manual_seed(0)
    t5 = T.T5ForConditionalGeneration(T.T5Config(
        vocab_size=len(hf_tok), d_model=64, d_ff=128, num_layers=2, num_heads=4, d_kv=16,
        decoder_start_token_id=hf_tok.pad_token_id, pad_token_id=hf_tok.pad_token_id,
        eos_token_id=hf_tok.eos_token_id)).eval()
    r = lr.analyze(t5, hf_tok, checks="architecture,perplexity,repetition", max_new_tokens=6, verbose=False)
    a = r["architecture"]
    assert r.metadata["model_kind"] == "seq2seq"
    assert a.metrics["encoder_layers"] == 2 and a.metrics["decoder_layers"] == 2
    assert "encoder-decoder" in a.details["structure"]["family"]
    assert r["repetition"].status in ("ok", "warning")
    # second half of the text scored given the first half, by hand
    ids = hf_tok(TEXTS[0], return_tensors="pt")["input_ids"]
    half = ids.shape[1] // 2
    with torch.no_grad():
        loss = float(t5(input_ids=ids[:, :half], labels=ids[:, half:]).loss)
    assert lr.perplexity(t5, hf_tok, [TEXTS[0]]) == pytest.approx(math.exp(loss), rel=1e-4)
    out = t5.generate(input_ids=hf_tok("The river", return_tensors="pt")["input_ids"], max_new_tokens=6,
                      do_sample=False)
    assert lr.generate(t5, hf_tok, "The river", 6) == hf_tok.decode(out[0], skip_special_tokens=True).strip()


def test_models_without_a_language_model_head(hf_tok):
    import transformers as T

    torch.manual_seed(0)
    base = T.GPT2Model(T.GPT2Config(vocab_size=len(hf_tok), n_embd=64, n_layer=2, n_head=4, n_positions=64)).eval()
    r = lr.analyze(base, hf_tok, checks="architecture,performance,perplexity,code", verbose=False)
    assert r.metadata["model_kind"] == "encoder"
    assert r["perplexity"].status == "skipped" and r["code"].status == "skipped"
    assert r["performance"].status == "ok" and "forward_ms" in r["performance"].metrics
    assert lr.info(base, hf_tok).kind == "base model without a language-model head"


def test_causal_vs_bidirectional_is_measured(tok, hf_tok):
    import transformers as T

    from llmreport.adapters import LanguageModel

    assert LanguageModel(tiny_gpt(tok.get_vocab_size()).eval(), tok).is_causal() is True
    bert = T.BertModel(T.BertConfig(vocab_size=len(hf_tok), hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=128)).eval()
    assert LanguageModel(bert, hf_tok).is_causal() is False


def test_roberta_context_accounts_for_position_offset(hf_tok):
    import transformers as T

    m = T.RobertaForMaskedLM(T.RobertaConfig(vocab_size=len(hf_tok), hidden_size=64, num_hidden_layers=1,
                                             num_attention_heads=4, intermediate_size=128,
                                             max_position_embeddings=130, pad_token_id=1)).eval()
    assert lr.architecture(m, hf_tok)["max_context"] == 128
    long_text = " ".join(TEXTS * 40)
    assert math.isfinite(lr.perplexity(m, hf_tok, [long_text]))  # cut to fit, no index error
