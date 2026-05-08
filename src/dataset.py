"""MIDI -> tensor pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import (BOS_ID, EOS_ID, PAD_ID, encode_notes, midi_to_notes)


class MidiTokenDataset(Dataset):
    """Sliding-window dataset over a directory of MIDI files."""

    def __init__(self, midi_dir: str, seq_len: int = 1024, stride: int = 512):
        self.seq_len = seq_len
        self.stride = stride
        self.windows: List[torch.Tensor] = []
        self._load(midi_dir)

    def _load(self, midi_dir: str) -> None:
        paths = sorted(Path(midi_dir).rglob("*.mid")) + sorted(Path(midi_dir).rglob("*.midi"))
        for p in paths:
            try:
                notes = midi_to_notes(str(p))
            except Exception:
                continue
            if len(notes) < 8:
                continue
            ids = encode_notes(notes)
            t = torch.tensor(ids, dtype=torch.long)
            for start in range(0, max(1, len(t) - self.seq_len + 1), self.stride):
                window = t[start : start + self.seq_len]
                if len(window) < self.seq_len:
                    pad = torch.full((self.seq_len - len(window),), PAD_ID, dtype=torch.long)
                    window = torch.cat([window, pad])
                self.windows.append(window)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def collate_lm(batch: Sequence[torch.Tensor]):
    """Returns (inputs, targets) for next-token prediction."""
    x = torch.stack(batch, dim=0)
    return x[:, :-1], x[:, 1:]
