"""Export the current score to MIDI and rendered WAV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pretty_midi

from .tokenizer import POSITIONS_PER_BAR, Note, velocity_from_bin


def notes_to_pretty_midi(notes: Sequence[Note], tempo_bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="Generated")
    sec_per_sixteenth = 60.0 / tempo_bpm / 4.0
    for n in notes:
        start_16 = n.bar * POSITIONS_PER_BAR + n.position
        start = start_16 * sec_per_sixteenth
        end = start + n.duration * sec_per_sixteenth
        inst.notes.append(pretty_midi.Note(
            velocity=velocity_from_bin(n.velocity_bin),
            pitch=n.pitch,
            start=start,
            end=end,
        ))
    pm.instruments.append(inst)
    return pm


def export_midi(notes: Sequence[Note], path: str, tempo_bpm: float = 120.0) -> str:
    pm = notes_to_pretty_midi(notes, tempo_bpm=tempo_bpm)
    pm.write(path)
    return path


def export_wav(
    notes: Sequence[Note],
    path: str,
    soundfont_path: str,
    tempo_bpm: float = 120.0,
    sample_rate: int = 44100,
) -> str:
    """Render to WAV via FluidSynth. Requires a SoundFont (.sf2)."""
    import numpy as np
    import soundfile as sf
    import fluidsynth

    if not Path(soundfont_path).exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

    fs = fluidsynth.Synth(samplerate=sample_rate)
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)

    sec_per_sixteenth = 60.0 / tempo_bpm / 4.0
    events = []
    for n in notes:
        start = (n.bar * POSITIONS_PER_BAR + n.position) * sec_per_sixteenth
        end = start + n.duration * sec_per_sixteenth
        events.append((start, "on", n.pitch, velocity_from_bin(n.velocity_bin)))
        events.append((end, "off", n.pitch, 0))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    if not events:
        sf.write(path, np.zeros(int(sample_rate * 0.5), dtype="float32"), sample_rate)
        return path

    total_dur = events[-1][0] + 0.5
    buf = np.zeros((int(total_dur * sample_rate), 2), dtype="float32")
    cursor = 0.0
    cursor_idx = 0
    for t, kind, pitch, vel in events:
        n_samples = int((t - cursor) * sample_rate)
        if n_samples > 0:
            chunk = fs.get_samples(n_samples).reshape(-1, 2).astype("float32") / 32768.0
            end_idx = min(cursor_idx + n_samples, buf.shape[0])
            buf[cursor_idx:end_idx] = chunk[: end_idx - cursor_idx]
            cursor_idx = end_idx
            cursor = t
        if kind == "on":
            fs.noteon(0, pitch, vel)
        else:
            fs.noteoff(0, pitch)

    # tail
    tail = int(0.5 * sample_rate)
    chunk = fs.get_samples(tail).reshape(-1, 2).astype("float32") / 32768.0
    end_idx = min(cursor_idx + tail, buf.shape[0])
    buf[cursor_idx:end_idx] = chunk[: end_idx - cursor_idx]

    fs.delete()
    sf.write(path, buf, sample_rate)
    return path
