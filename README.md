# ScoreCompose-AI

🌐 **Project site**: https://rosyrosys.github.io/score_compose_ai/

Symbolic music generation with **notated output** and **edit-aware incremental
decoding**, supporting real-time score editing and export to MusicXML, MIDI,
and rendered audio.

This repository accompanies the paper:

> *ScoreCompose-AI: Edit-Aware Incremental Decoding for Notated Symbolic Music
> Generation with Real-Time Score Editing and Audio Synthesis*

## What it does

1. A small GPT-style Transformer trained on a REMI-like tokenization generates
   symbolic music.
2. Tokens are converted to **MusicXML** so the system shows real notation, not
   piano-roll.
3. The browser editor (OpenSheetMusicDisplay) lets the user click a note and
   change its pitch or duration. The edit is sent to the server.
4. The server runs **edit-aware incremental decoding**: it keeps the KV-cache
   for the unchanged prefix and only re-decodes the suffix when the user asks
   the model to *continue after the edit*. Local edits do not require any
   model recomputation.
5. The current state is exported to **MIDI** (`pretty_midi`) and rendered to
   **WAV** with FluidSynth.

## Project layout

```
score_compose_ai/
├── src/
│   ├── tokenizer.py        REMI-like tokenization (music21-based)
│   ├── model.py            Decoder-only Transformer
│   ├── dataset.py          MIDI -> token tensors
│   ├── train.py            Training loop (Colab-friendly)
│   ├── generate.py         Top-p sampling
│   ├── edit_engine.py      Edit ops + incremental decoding
│   ├── score_io.py         tokens <-> music21 stream <-> MusicXML
│   ├── export.py           MIDI / WAV export
│   ├── server.py           FastAPI + WebSocket backend
│   └── static/index.html   OSMD-based editor
├── paper/
│   ├── main.tex
│   └── refs.bib
├── scripts/
│   ├── train_colab.py
│   └── prepare_lakh_subset.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.server          # http://localhost:8000
```

Open the URL, click *Generate*, then drag a notehead to edit.
*Export MIDI* / *Export WAV* download the current score.

## Training (Google Colab T4)

Open `scripts/colab_train.ipynb` in Colab, set the runtime to **T4 GPU**, and
run the cells top to bottom. The notebook:

1. mounts Google Drive (so checkpoints survive runtime restarts),
2. downloads MAESTRO v3 (default; reliable, ~80 MB) or LMD-clean,
3. trains for 8 epochs with fp16 AMP, periodic checkpointing every 500 steps,
   and automatic resume from `weights.pt.last`,
4. generates a verification sample (`sample.mid`, `sample.musicxml`).

8 epochs on MAESTRO take ~4–6 h on a T4. After training, point the server at
the saved weights:

```bash
SCORECOMPOSE_WEIGHTS=/path/to/weights.pt python -m src.server
```
