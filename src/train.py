"""Training loop. Designed to be runnable on a single Colab T4 / A100.

Features kept in mind for Colab specifically:

* Mixed-precision (fp16 autocast + GradScaler) — roughly halves T4 step time
  and keeps the 6-layer / d=384 model well inside 16GB.
* Periodic checkpoints (every `ckpt_every` steps) saved to `out_path`,
  plus an `out_path + ".last"` so you can resume after a runtime restart.
* Validation split by file count (`val_frac`).
* Graceful resume from `--resume`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .dataset import MidiTokenDataset, collate_lm
from .model import ModelConfig, ScoreLM
from .tokenizer import PAD_ID, VOCAB_SIZE


def _split(ds: MidiTokenDataset, val_frac: float, seed: int = 0):
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_frac))
    return Subset(ds, idx[n_val:]), Subset(ds, idx[:n_val])


def _save(model, opt, sched, scaler, step, path, cfg: ModelConfig):
    # Persist the ModelConfig alongside the weights so the loader can
    # rebuild the architecture without guessing.
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size":   cfg.vocab_size,
            "d_model":      cfg.d_model,
            "n_heads":      cfg.n_heads,
            "n_layers":     cfg.n_layers,
            "d_ff":         cfg.d_ff,
            "max_seq_len":  cfg.max_seq_len,
            "dropout":      cfg.dropout,
        },
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
    }, path)


def _load(path, model, opt=None, sched=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    if sched is not None and "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)


@torch.no_grad()
def _evaluate(model, dl, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD_ID,
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(
    midi_dir: str,
    out_path: str,
    epochs: int = 8,
    batch_size: int = 16,
    seq_len: int = 1024,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    val_frac: float = 0.05,
    ckpt_every: int = 500,
    resume: str = "",
    log_every: int = 50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    ds = MidiTokenDataset(midi_dir, seq_len=seq_len, stride=seq_len // 2)
    print(f"Loaded {len(ds)} windows from {midi_dir}")
    if len(ds) < 4:
        raise RuntimeError("Not enough data; check that midi_dir contains .mid files.")
    train_set, val_set = _split(ds, val_frac=val_frac)

    pin = use_amp
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_lm, num_workers=2,
                          pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_lm, num_workers=2,
                        pin_memory=pin, drop_last=False)

    cfg = ModelConfig(vocab_size=VOCAB_SIZE, max_seq_len=seq_len)
    model = ScoreLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    steps_total = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    if resume and Path(resume).exists():
        start_step = _load(resume, model, opt, sched, scaler)
        print(f"resumed from {resume} at step {start_step}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path + ".last"

    step = start_step
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1),
                    ignore_index=PAD_ID,
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    ppl=f"{math.exp(loss.item()):.2f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                )
            if step % ckpt_every == 0:
                _save(model, opt, sched, scaler, step, last_path, cfg)

        val_loss = _evaluate(model, val_dl, device)
        print(f"epoch {epoch} done. val_loss={val_loss:.3f} val_ppl={math.exp(val_loss):.2f}"
              f"  elapsed={(time.time()-t_start)/60:.1f}min")

        # Per-epoch checkpoint (overwrites last main file).
        _save(model, opt, sched, scaler, step, out_path, cfg)
        _save(model, opt, sched, scaler, step, last_path, cfg)
        print(f"saved -> {out_path}")

    print(f"training done. total elapsed = {(time.time()-t_start)/60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dir", required=True)
    p.add_argument("--out", default="weights.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--resume", default="")
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()
    train(
        midi_dir=args.midi_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        ckpt_every=args.ckpt_every,
        resume=args.resume,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
