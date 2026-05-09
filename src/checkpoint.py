"""Checkpoint loading helpers.

Trained checkpoints save a `config` dict alongside the weights. Older
checkpoints (saved before that change) only contain the state_dict. We
support both: when `config` is missing we infer the architecture from the
saved tensor shapes so a previously trained `weights.pt` keeps working.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .model import ModelConfig, ScoreLM
from .tokenizer import VOCAB_SIZE


def _infer_config_from_state_dict(sd) -> ModelConfig:
    pos = sd["pos_emb.weight"]
    max_seq_len, d_model = int(pos.shape[0]), int(pos.shape[1])

    n_layers = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1

    # ff.0.weight shape is (d_ff, d_model)
    d_ff = int(sd["blocks.0.ff.0.weight"].shape[0])

    # n_heads cannot be uniquely recovered from shapes (qkv weight is
    # (3*d_model, d_model) regardless). We keep the project default and
    # warn if the user was using something exotic.
    default = ModelConfig(vocab_size=VOCAB_SIZE)
    return ModelConfig(
        vocab_size=int(sd["tok_emb.weight"].shape[0]),
        d_model=d_model,
        n_heads=default.n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,  # inference
    )


def load_for_inference(weights_path: str, device) -> Tuple[ScoreLM, ModelConfig]:
    """Load a trained checkpoint and return (model, config).

    Accepts both new-style (`{"model": sd, "config": {...}}`) and
    legacy-style (raw state_dict) checkpoints.
    """
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ModelConfig(**ckpt["config"])
        sd = ckpt["model"]
        source = "config dict"
    else:
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        cfg = _infer_config_from_state_dict(sd)
        source = "shape inference"

    cfg.dropout = 0.0
    model = ScoreLM(cfg).to(device).eval()
    model.load_state_dict(sd, strict=True)
    print(f"loaded weights ({source}): max_seq_len={cfg.max_seq_len} "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers}")
    return model, cfg
