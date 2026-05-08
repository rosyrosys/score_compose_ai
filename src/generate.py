"""Sampling utilities. Used both for cold generation and for the suffix
re-decoding step in edit_engine.continue_from."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .model import KVCache, ScoreLM
from .tokenizer import EOS_ID, PAD_ID


@torch.no_grad()
def warmup_caches(model: ScoreLM, prefix_ids: Sequence[int], device) -> List[KVCache]:
    """Run the prefix through the model once so caches are filled."""
    caches = model.make_caches(device=device, dtype=torch.float32)
    if len(prefix_ids) > 0:
        x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        model(x, caches=caches, offset=0)
    return caches


def _filter_logits(logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def sample_next(
    model: ScoreLM,
    last_token: int,
    caches: List[KVCache],
    temperature: float = 1.0,
    top_p: float = 0.92,
) -> int:
    device = next(model.parameters()).device
    x = torch.tensor([[last_token]], dtype=torch.long, device=device)
    offset = caches[0].length
    logits = model(x, caches=caches, offset=offset)[0, -1]
    logits[PAD_ID] = float("-inf")
    logits = _filter_logits(logits, top_p=top_p, temperature=temperature)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(
    model: ScoreLM,
    prefix_ids: Sequence[int],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.92,
    stop_at_eos: bool = True,
) -> List[int]:
    """Generate a continuation. Returns prefix + new tokens."""
    device = next(model.parameters()).device
    caches = warmup_caches(model, prefix_ids[:-1] if len(prefix_ids) > 1 else [], device)
    out = list(prefix_ids)
    last = out[-1] if out else 0
    for _ in range(max_new_tokens):
        nxt = sample_next(model, last, caches, temperature, top_p)
        out.append(nxt)
        last = nxt
        if stop_at_eos and nxt == EOS_ID:
            break
    return out
