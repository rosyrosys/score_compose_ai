"""Edit-aware incremental decoding.

The user-facing object is a list of Notes. Each Note maps to a contiguous
4- or 5-token span (the leading <bar> only when the bar changes). When the
user edits note i, we:

  1. Mutate the Note in place.
  2. Re-tokenize the whole sequence (cheap: O(N) over notes).
  3. Find the first token index that differs from the previous tokenization.
  4. Truncate every layer's KV-cache to that index.
  5. Replay only the changed prefix (the *changed* tokens, not the entire
     prefix) through the model so caches are valid again.

Step 5 is the key optimization: in the common case where only one note's
pitch or duration changes, the cache truncation point is right before that
note, and we replay 4-5 tokens instead of thousands.

A `continue_from()` method then samples additional tokens after the edit.
This is what users hit when they want the model to extend the score *given*
their edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .generate import sample_next
from .model import KVCache, ScoreLM
from .tokenizer import (BAR_ID, BOS_ID, EOS_ID, ID2TOKEN, Note,
                        TOKEN2ID, decode_tokens, encode_notes)


@dataclass
class EditState:
    """Mutable state for a single editing session."""
    notes: List[Note] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    caches: Optional[List[KVCache]] = None  # filled lazily
    cache_valid_to: int = 0                  # token index up to which caches are valid


def _first_diff(a: Sequence[int], b: Sequence[int]) -> int:
    """Index of first differing element, or len(min) if one is a prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class EditEngine:
    """Wraps a model and exposes note-level edit ops with incremental decoding."""

    def __init__(self, model: ScoreLM, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.state = EditState()

    # ---------- session lifecycle ----------

    def reset(self, notes: Optional[Sequence[Note]] = None) -> None:
        self.state = EditState()
        if notes:
            self.state.notes = list(notes)
        self.state.token_ids = encode_notes(self.state.notes) if self.state.notes else [BOS_ID]
        self.state.caches = self.model.make_caches(self.device, torch.float32)
        self.state.cache_valid_to = 0
        self._sync_cache_to_tokens()

    # ---------- edit operations ----------

    def insert_note(self, idx: int, note: Note) -> int:
        idx = max(0, min(idx, len(self.state.notes)))
        self.state.notes.insert(idx, note)
        return self._reconcile()

    def delete_note(self, idx: int) -> int:
        if not (0 <= idx < len(self.state.notes)):
            raise IndexError(idx)
        del self.state.notes[idx]
        return self._reconcile()

    def replace_note(self, idx: int, note: Note) -> int:
        if not (0 <= idx < len(self.state.notes)):
            raise IndexError(idx)
        self.state.notes[idx] = note
        return self._reconcile()

    def transpose(self, semitones: int, idx_range: Optional[Tuple[int, int]] = None) -> int:
        lo, hi = idx_range or (0, len(self.state.notes))
        for i in range(lo, hi):
            n = self.state.notes[i]
            new_pitch = max(21, min(108, n.pitch + semitones))
            self.state.notes[i] = Note(
                bar=n.bar, position=n.position, pitch=new_pitch,
                duration=n.duration, velocity_bin=n.velocity_bin,
            )
        return self._reconcile()

    # ---------- model continuation ----------

    @torch.no_grad()
    def continue_from(
        self,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.92,
    ) -> List[Note]:
        """Sample additional tokens after the current sequence and return the new notes."""
        # Strip a trailing EOS if present; we want to continue the music.
        ids = self.state.token_ids
        if ids and ids[-1] == EOS_ID:
            ids = ids[:-1]
            self.state.token_ids = ids
            # Cache for the dropped EOS is still valid up to len(ids)-? — easiest: re-sync.
            self.state.cache_valid_to = min(self.state.cache_valid_to, len(ids))
        self._sync_cache_to_tokens()

        new_ids: List[int] = []
        last = ids[-1] if ids else BOS_ID
        for _ in range(max_new_tokens):
            nxt = sample_next(self.model, last, self.state.caches,
                              temperature=temperature, top_p=top_p)
            self.state.token_ids.append(nxt)
            self.state.cache_valid_to = self.state.caches[0].length
            new_ids.append(nxt)
            last = nxt
            if nxt == EOS_ID:
                break

        # Update the note list to reflect the new tokens.
        new_notes = decode_tokens(new_ids)
        self.state.notes.extend(new_notes)
        # Re-tokenize once to canonicalize (handles bar tokens cleanly).
        self.state.token_ids = encode_notes(self.state.notes)
        self.state.cache_valid_to = min(self.state.cache_valid_to, len(self.state.token_ids))
        self._sync_cache_to_tokens()
        return new_notes

    # ---------- internals ----------

    def _reconcile(self) -> int:
        """Re-tokenize, diff against the cached token stream, truncate caches.

        Returns the truncation point (= first changed token index)."""
        new_ids = encode_notes(self.state.notes) if self.state.notes else [BOS_ID]
        diff_at = _first_diff(self.state.token_ids, new_ids)
        self.state.token_ids = new_ids

        # Cache validity cannot exceed the unchanged prefix.
        new_valid = min(self.state.cache_valid_to, diff_at)
        for c in self.state.caches:
            if new_valid < c.length:
                c.truncate(new_valid)
        self.state.cache_valid_to = new_valid

        # Replay only the missing portion through the model so caches stay populated.
        self._sync_cache_to_tokens()
        return diff_at

    @torch.no_grad()
    def _sync_cache_to_tokens(self) -> None:
        """Make sure the KV-caches cover [0, len(token_ids))."""
        if self.state.caches is None:
            self.state.caches = self.model.make_caches(self.device, torch.float32)
            self.state.cache_valid_to = 0
        target = len(self.state.token_ids)
        if self.state.cache_valid_to >= target:
            return
        gap_ids = self.state.token_ids[self.state.cache_valid_to : target]
        if not gap_ids:
            return
        x = torch.tensor([gap_ids], dtype=torch.long, device=self.device)
        self.model(x, caches=self.state.caches, offset=self.state.cache_valid_to)
        self.state.cache_valid_to = target
