"""Language models written from scratch in plain PyTorch, one per architecture family.

None of them use Hugging Face. They exist to check that llmreport handles models it
has never seen: different call signatures, output formats, position encodings, norms,
attention layouts and recurrent layers.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ------------------------------------------------------------------ Llama-style


@dataclass
class LlamaArgs:
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 8
    n_kv_heads: int = 2
    vocab_size: int = 400
    hidden_dim: int = 160
    max_seq_len: int = 128


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    return torch.polar(torch.ones(end, dim // 2), torch.outer(t, freqs))


def apply_rotary(x, freqs_cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis[None, :, None, :]).flatten(3)
    return out.type_as(x)


class LlamaAttention(nn.Module):
    def __init__(self, a: LlamaArgs):
        super().__init__()
        self.n_heads, self.n_kv_heads = a.n_heads, a.n_kv_heads
        self.head_dim = a.dim // a.n_heads
        self.wq = nn.Linear(a.dim, a.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(a.dim, a.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(a.n_heads * self.head_dim, a.dim, bias=False)

    def forward(self, x, freqs_cis):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary(q, freqs_cis), apply_rotary(k, freqs_cis)
        rep = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, a: LlamaArgs):
        super().__init__()
        self.w1 = nn.Linear(a.dim, a.hidden_dim, bias=False)
        self.w2 = nn.Linear(a.hidden_dim, a.dim, bias=False)
        self.w3 = nn.Linear(a.dim, a.hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self, a: LlamaArgs):
        super().__init__()
        self.attention_norm = RMSNorm(a.dim)
        self.attention = LlamaAttention(a)
        self.ffn_norm = RMSNorm(a.dim)
        self.feed_forward = SwiGLU(a)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        return x + self.feed_forward(self.ffn_norm(x))


class LlamaStyle(nn.Module):
    """RoPE, RMSNorm, grouped-query attention, SwiGLU, untied output. Returns a bare tensor."""

    def __init__(self, params: LlamaArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(LlamaBlock(params) for _ in range(params.n_layers))
        self.norm = RMSNorm(params.dim)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len),
                             persistent=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[: tokens.shape[1]]
        for layer in self.layers:
            h = layer(h, freqs)
        return self.output(self.norm(h))


class NeedsStartPos(LlamaStyle):
    """Meta's reference Llama signature: forward(tokens, start_pos)."""

    def forward(self, tokens, start_pos):
        assert start_pos == 0
        return super().forward(tokens)


# ------------------------------------------------------------------ nanoGPT-style (last position only)


class LastOnlyGPT(nn.Module):
    """Like nanoGPT: without targets, only the last position's logits are returned."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, idx, targets=None):
        logits, _ = self.inner(idx)
        if targets is None:
            return logits[:, [-1], :], None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ------------------------------------------------------------------ recurrent


class LSTMLM(nn.Module):
    """Embedding -> 2-layer LSTM -> Linear. Returns (logits, hidden state)."""

    def __init__(self, vocab=400, dim=48, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True)
        self.decoder = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        out, state = self.rnn(self.embed(x), state)
        return self.decoder(out), state


# ------------------------------------------------------------------ seq-first nn.Transformer


class SeqFirstTransformer(nn.Module):
    """Uses nn.TransformerEncoder with the default (seq, batch, dim) layout and a dict output."""

    def __init__(self, vocab=400, d_model=64, nhead=4, layers=2, max_len=96):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.0, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(d_model, vocab)

    def forward(self, src):
        x = src.transpose(0, 1)  # (T, B)
        t = x.shape[0]
        h = self.embed(x) + self.pos(torch.arange(t, device=src.device))[:, None, :]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=src.device), diagonal=1)
        h = self.encoder(h, mask=mask, is_causal=True)
        return {"logits": self.lm_head(h), "hidden": h}  # logits are (T, B, V)


# ------------------------------------------------------------------ scripted


class ScriptedGPT(nn.Module):
    """A from-scratch model (no generate(), no Hugging Face) whose greedy output we script.

    ``respond(prompt_text) -> reply_text``. The first call on a new sequence treats it as
    the prompt; later calls extend it one reply token at a time, then emit ``end_id``.
    """

    def __init__(self, tok, respond, end_id, vocab):
        super().__init__()
        self.tok, self.respond, self.end_id, self.vocab = tok, respond, end_id, vocab
        self.dummy = nn.Parameter(torch.zeros(1))
        self.prompt, self.reply, self.seen = None, [], []
        self.block_size = 4096

    def forward(self, idx):
        seq = idx[0].tolist()
        n = len(self.prompt) if self.prompt is not None else 0
        if self.prompt is None or seq[:n] != self.prompt or len(seq) - n > len(self.reply):
            self.prompt = seq
            text = self.tok.decode(seq, skip_special_tokens=False)
            self.seen.append(text)
            self.reply = self.tok.encode(self.respond(text)).ids
            n = len(seq)
        pos = len(seq) - n
        nxt = self.reply[pos] if pos < len(self.reply) else self.end_id
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab)
        logits[:, -1, nxt] = 10.0
        return logits


def uniform_logits_model(vocab):
    """A plain function (not an nn.Module) that gives every token the same score."""

    def forward(ids):
        return torch.zeros(ids.shape[0], ids.shape[1], vocab)

    return forward
