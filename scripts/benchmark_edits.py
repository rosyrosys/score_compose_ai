"""Latency benchmark for edit-aware decoding.

Measures three quantities for various sequence lengths:

  - cold_decode    : full forward over the prefix (baseline)
  - replay_only    : edit-aware reconcile (truncate + replay)
  - continuation   : sample 32 new tokens after the edit

Run:
    python scripts/benchmark_edits.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.edit_engine import EditEngine
from src.model import ModelConfig, ScoreLM
from src.tokenizer import POSITIONS_PER_BAR, VOCAB_SIZE, Note


def random_notes(n: int, rng) -> List[Note]:
    return [
        Note(
            bar=i // 4,
            position=(i * 4) % POSITIONS_PER_BAR,
            pitch=int(rng.integers(48, 84)),
            duration=int(rng.choice([1, 2, 4, 8])),
            velocity_bin=int(rng.integers(0, 8)),
        )
        for i in range(n)
    ]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, max_seq_len=4096)
    model = ScoreLM(cfg).to(device).eval()

    rng = __import__("numpy").random.default_rng(0)

    print(f"{'N_notes':>8} {'cold_ms':>10} {'replay_ms':>10} {'cont_ms':>10}")
    for n_notes in (32, 128, 512, 1024):
        notes = random_notes(n_notes, rng)
        eng = EditEngine(model)
        eng.reset(notes)

        # cold decode
        t0 = time.perf_counter()
        x = torch.tensor([eng.state.token_ids], device=device)
        with torch.no_grad():
            model(x)
        cold = (time.perf_counter() - t0) * 1000

        # replay-only after editing the LAST note
        t = []
        for _ in range(5):
            last = eng.state.notes[-1]
            new_note = Note(last.bar, last.position,
                            (last.pitch + 1) if last.pitch < 108 else last.pitch - 1,
                            last.duration, last.velocity_bin)
            t0 = time.perf_counter()
            eng.replace_note(len(eng.state.notes) - 1, new_note)
            t.append((time.perf_counter() - t0) * 1000)
        replay = mean(t)

        # continuation (32 tokens)
        t0 = time.perf_counter()
        eng.continue_from(max_new_tokens=32, temperature=1.0, top_p=0.92)
        cont = (time.perf_counter() - t0) * 1000

        print(f"{n_notes:>8d} {cold:>10.1f} {replay:>10.1f} {cont:>10.1f}")


if __name__ == "__main__":
    main()
