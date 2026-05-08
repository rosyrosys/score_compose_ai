"""Conversion between Notes and music21 Streams / MusicXML strings."""

from __future__ import annotations

from typing import List, Sequence

from music21 import duration, note, stream

from .tokenizer import POSITIONS_PER_BAR, Note, velocity_from_bin


_QL_PER_SIXTEENTH = 0.25  # quarterLength per sixteenth


def notes_to_stream(notes: Sequence[Note], time_signature: str = "4/4") -> stream.Score:
    """Build a single-staff score from Note objects."""
    score = stream.Score()
    part = stream.Part()
    score.insert(0, part)

    if not notes:
        return score

    sorted_notes = sorted(notes, key=lambda n: (n.bar, n.position, n.pitch))

    last_offset = 0.0
    for n in sorted_notes:
        absolute_pos_16 = n.bar * POSITIONS_PER_BAR + n.position
        target_offset = absolute_pos_16 * _QL_PER_SIXTEENTH

        if target_offset > last_offset:
            r = note.Rest()
            r.duration = duration.Duration(target_offset - last_offset)
            part.insert(last_offset, r)

        m21_note = note.Note(midi=n.pitch)
        m21_note.duration = duration.Duration(n.duration * _QL_PER_SIXTEENTH)
        m21_note.volume.velocity = velocity_from_bin(n.velocity_bin)
        part.insert(target_offset, m21_note)
        last_offset = target_offset + n.duration * _QL_PER_SIXTEENTH

    part.makeMeasures(inPlace=True)
    return score


def stream_to_musicxml(score: stream.Score) -> str:
    from music21.musicxml import m21ToXml

    exporter = m21ToXml.GeneralObjectExporter(score)
    return exporter.parse().decode("utf-8")


def notes_to_musicxml(notes: Sequence[Note]) -> str:
    return stream_to_musicxml(notes_to_stream(notes))


def notes_to_dict_list(notes: Sequence[Note]) -> List[dict]:
    """Lightweight JSON-friendly view of the score, used by the web client."""
    return [
        {
            "i": i,
            "bar": n.bar,
            "position": n.position,
            "pitch": n.pitch,
            "duration": n.duration,
            "velocity_bin": n.velocity_bin,
        }
        for i, n in enumerate(notes)
    ]
