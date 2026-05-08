"""A small decoder-only Transformer with KV-cache support.

The KV-cache exposes truncate(n) so we can keep the cache for an unchanged
prefix and discard everything beyond the first edited position. This is the
core enabler of edit-aware incremental decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    max_seq_len: int = 2048
    dropout: float = 0.1


class KVCache:
    """One per layer. Holds keys and values for tokens already processed."""

    def __init__(self, max_len: int, n_heads: int, head_dim: int, device, dtype):
        self.max_len = max_len
        self.k = torch.zeros(1, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(1, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.length = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = k_new.shape[2]
        self.k[:, :, self.length:self.length + n] = k_new
        self.v[:, :, self.length:self.length + n] = v_new
        self.length += n
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def truncate(self, new_length: int) -> None:
        """Drop everything after new_length. Used after an edit at position p."""
        assert 0 <= new_length <= self.length
        self.length = new_length


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cache: Optional[KVCache]) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.append(k, v)

        # Causal mask only when we're processing >1 query token at once.
        is_causal = T > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, cache: Optional[KVCache]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache)
        x = x + self.ff(self.ln2(x))
        return x


class ScoreLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def make_caches(self, device, dtype) -> List[KVCache]:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        return [
            KVCache(self.cfg.max_seq_len, self.cfg.n_heads, head_dim, device, dtype)
            for _ in range(self.cfg.n_layers)
        ]

    def forward(
        self,
        ids: torch.Tensor,                  # (B, T)
        caches: Optional[List[KVCache]] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        T = ids.shape[1]
        positions = torch.arange(offset, offset + T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(positions)[None, :, :]
        for blk, cache in zip(self.blocks, caches or [None] * self.cfg.n_layers):
            x = blk(x, cache)
        x = self.ln_f(x)
        return self.head(x)
