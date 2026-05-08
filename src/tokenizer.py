"""REMI-like tokenizer.

Each note is represented by a 4-token group:
    BAR? POSITION_p PITCH_x DURATION_d VELOCITY_v
where BAR is emitted only at bar boundaries. This grouping is convenient for
edit-aware decoding because every visible note maps to a contiguous
4-or-5-token span.

Time grid: 16 positions per bar (sixteenth-note resolution, 4/4 assumed).
Pitch range: MIDI 21..108 (88-key piano).
Durations: quantized to {1,2,3,4,6,8,12,16} sixteenths.
Velocities: 8 bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

POSITIONS_PER_BAR = 16
PITCH_LO, PITCH_HI = 21, 108
DURATION_BINS = (1, 2, 3, 4, 6, 8, 12, 16)
VELOCITY_BINS = 8

SPECIAL = ["<pad>", "<bos>", "<eos>", "<bar>"]


def _build_vocab() -> List[str]:
    vocab = list(SPECIAL)
    vocab += [f"POS_{i}" for i in range(POSITIONS_PER_BAR)]
    vocab += [f"PITCH_{p}" for p in range(PITCH_LO, PITCH_HI + 1)]
    vocab += [f"DUR_{d}" for d in DURATION_BINS]
    vocab += [f"VEL_{v}" for v in range(VELOCITY_BINS)]
    return vocab


VOCAB: List[str] = _build_vocab()
TOKEN2ID = {tok: i for i, tok in enumerate(VOCAB)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)

PAD_ID = TOKEN2ID["<pad>"]
BOS_ID = TOKEN2ID["<bos>"]
EOS_ID = TOKEN2ID["<eos>"]
BAR_ID = TOKEN2ID["<bar>"]


@dataclass(frozen=True)
class Note:
    """A single note in the symbolic representation."""
    bar: int               # 0-indexed bar number
    position: int          # 0..POSITIONS_PER_BAR-1
    pitch: int             # MIDI pitch
    duration: int          # in sixteenths, snapped to DURATION_BINS
    velocity_bin: int      # 0..VELOCITY_BINS-1

    def to_tokens(self, emit_bar: bool) -> List[str]:
        toks = []
        if emit_bar:
            toks.append("<bar>")
        toks += [
            f"POS_{self.position}",
            f"PITCH_{self.pitch}",
            f"DUR_{self.duration}",
            f"VEL_{self.velocity_bin}",
        ]
        return toks


def _snap_duration(sixteenths: int) -> int:
    sixteenths = max(1, sixteenths)
    return min(DURATION_BINS, key=lambda d: abs(d - sixteenths))


def _vel_bin(vel: int) -> int:
    return max(0, min(VELOCITY_BINS - 1, vel * VELOCITY_BINS // 128))


def _vel_unbin(b: int) -> int:
    return int((b + 0.5) * 128 / VELOCITY_BINS)


def encode_notes(notes: Sequence[Note]) -> List[int]:
    """Notes (sorted by (bar, position, pitch)) -> token ids with <bos>/<eos>."""
    out = [BOS_ID]
    last_bar = -1
    for n in sorted(notes, key=lambda x: (x.bar, x.position, x.pitch)):
        emit_bar = n.bar != last_bar
        for tok in n.to_tokens(emit_bar=emit_bar):
            out.append(TOKEN2ID[tok])
        last_bar = n.bar
    out.append(EOS_ID)
    return out


def decode_tokens(ids: Iterable[int]) -> List[Note]:
    """Token ids -> Notes. Malformed groups are skipped silently."""
    notes: List[Note] = []
    cur_bar = -1   # so the FIRST <bar> token sets cur_bar to 0
    state = {"pos": None, "pitch": None, "dur": None}
    for tid in ids:
        if tid in (PAD_ID, BOS_ID, EOS_ID):
            continue
        tok = ID2TOKEN[tid]
        if tok == "<bar>":
            cur_bar += 1
            state = {"pos": None, "pitch": None, "dur": None}
        elif tok.startswith("POS_"):
            state = {"pos": int(tok[4:]), "pitch": None, "dur": None}
        elif tok.startswith("PITCH_") and state["pos"] is not None:
            state["pitch"] = int(tok[6:])
        elif tok.startswith("DUR_") and state["pitch"] is not None:
            state["dur"] = int(tok[4:])
        elif tok.startswith("VEL_") and state["dur"] is not None:
            vbin = int(tok[4:])
            notes.append(Note(
                bar=cur_bar,
                position=state["pos"],
                pitch=state["pitch"],
                duration=state["dur"],
                velocity_bin=vbin,
            ))
            state = {"pos": None, "pitch": None, "dur": None}
    return notes


def midi_to_notes(midi_path: str) -> List[Note]:
    """Load a MIDI file and quantize to the REMI grid."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    if not pm.instruments:
        return []
    # Use the longest non-drum instrument.
    inst = max(
        (i for i in pm.instruments if not i.is_drum),
        key=lambda i: len(i.notes),
        default=None,
    )
    if inst is None:
        return []

    tempo = pm.estimate_tempo() or 120.0
    sec_per_sixteenth = 60.0 / tempo / 4.0
    notes: List[Note] = []
    for n in inst.notes:
        if not (PITCH_LO <= n.pitch <= PITCH_HI):
            continue
        start_16 = round(n.start / sec_per_sixteenth)
        dur_16 = max(1, round((n.end - n.start) / sec_per_sixteenth))
        bar = start_16 // POSITIONS_PER_BAR
        position = start_16 % POSITIONS_PER_BAR
        notes.append(Note(
            bar=bar,
            position=position,
            pitch=n.pitch,
            duration=_snap_duration(dur_16),
            velocity_bin=_vel_bin(n.velocity),
        ))
    return notes


def velocity_from_bin(b: int) -> int:
    return _vel_unbin(b)
