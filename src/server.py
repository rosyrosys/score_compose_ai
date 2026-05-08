"""FastAPI backend.

Endpoints
---------
GET  /                         the editor (static HTML)
POST /api/session/new          start a new session, returns initial score
POST /api/generate             ask the model for `n` more notes
POST /api/edit                 apply an edit op (insert/delete/replace/transpose)
GET  /api/score                current score as MusicXML + JSON
POST /api/export/midi          download MIDI of the current score
POST /api/export/wav           download WAV (requires SOUNDFONT_PATH)
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .edit_engine import EditEngine
from .export import export_midi, export_wav
from .model import ModelConfig, ScoreLM
from .score_io import notes_to_dict_list, notes_to_musicxml
from .tokenizer import (DURATION_BINS, POSITIONS_PER_BAR, VELOCITY_BINS,
                        VOCAB_SIZE, Note)


STATIC_DIR = Path(__file__).parent / "static"
WEIGHTS_PATH = os.environ.get("SCORECOMPOSE_WEIGHTS", "")
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "")

app = FastAPI(title="ScoreCompose-AI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- model singleton ----------

def _load_model() -> ScoreLM:
    cfg = ModelConfig(vocab_size=VOCAB_SIZE)
    model = ScoreLM(cfg)
    if WEIGHTS_PATH and Path(WEIGHTS_PATH).exists():
        sd = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


_MODEL = _load_model()
_ENGINES: Dict[str, EditEngine] = {}


def _engine(session_id: str) -> EditEngine:
    if session_id not in _ENGINES:
        raise HTTPException(404, "session not found")
    return _ENGINES[session_id]


# ---------- request schemas ----------

class NewSessionReq(BaseModel):
    n_initial_notes: int = 64
    temperature: float = 1.0
    top_p: float = 0.92


class GenerateReq(BaseModel):
    session_id: str
    n_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.92


class EditReq(BaseModel):
    session_id: str
    op: str               # "insert" | "delete" | "replace" | "transpose"
    index: Optional[int] = None
    note: Optional[dict] = None
    semitones: Optional[int] = None
    range: Optional[List[int]] = None


class ExportReq(BaseModel):
    session_id: str
    tempo_bpm: float = 120.0


def _note_from_dict(d: dict) -> Note:
    return Note(
        bar=int(d["bar"]),
        position=int(d["position"]) % POSITIONS_PER_BAR,
        pitch=max(21, min(108, int(d["pitch"]))),
        duration=min(DURATION_BINS, key=lambda x: abs(x - int(d.get("duration", 4)))),
        velocity_bin=max(0, min(VELOCITY_BINS - 1, int(d.get("velocity_bin", 4)))),
    )


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/session/new")
def new_session(req: NewSessionReq):
    import uuid
    sid = uuid.uuid4().hex[:12]
    eng = EditEngine(_MODEL)
    eng.reset(notes=[])
    new_notes = eng.continue_from(
        max_new_tokens=req.n_initial_notes * 5,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    _ENGINES[sid] = eng
    return {
        "session_id": sid,
        "notes": notes_to_dict_list(eng.state.notes),
        "musicxml": notes_to_musicxml(eng.state.notes),
        "n_new_notes": len(new_notes),
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    eng = _engine(req.session_id)
    new_notes = eng.continue_from(
        max_new_tokens=req.n_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return {
        "notes": notes_to_dict_list(eng.state.notes),
        "musicxml": notes_to_musicxml(eng.state.notes),
        "n_new_notes": len(new_notes),
    }


@app.post("/api/edit")
def edit(req: EditReq):
    eng = _engine(req.session_id)
    diff_at = -1
    if req.op == "replace":
        diff_at = eng.replace_note(req.index, _note_from_dict(req.note))
    elif req.op == "insert":
        diff_at = eng.insert_note(req.index or len(eng.state.notes), _note_from_dict(req.note))
    elif req.op == "delete":
        diff_at = eng.delete_note(req.index)
    elif req.op == "transpose":
        rng = tuple(req.range) if req.range else None
        diff_at = eng.transpose(req.semitones or 0, idx_range=rng)
    else:
        raise HTTPException(400, f"unknown op: {req.op}")

    return {
        "diff_at": diff_at,
        "cache_valid_to": eng.state.cache_valid_to,
        "n_tokens": len(eng.state.token_ids),
        "notes": notes_to_dict_list(eng.state.notes),
        "musicxml": notes_to_musicxml(eng.state.notes),
    }


@app.get("/api/score")
def score(session_id: str):
    eng = _engine(session_id)
    return {
        "notes": notes_to_dict_list(eng.state.notes),
        "musicxml": notes_to_musicxml(eng.state.notes),
    }


@app.post("/api/export/midi")
def midi(req: ExportReq):
    eng = _engine(req.session_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    tmp.close()
    export_midi(eng.state.notes, tmp.name, tempo_bpm=req.tempo_bpm)
    return FileResponse(tmp.name, filename="score.mid", media_type="audio/midi")


@app.post("/api/export/wav")
def wav(req: ExportReq):
    if not SOUNDFONT_PATH:
        raise HTTPException(500, "set SOUNDFONT_PATH env var to a .sf2 file")
    eng = _engine(req.session_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    export_wav(eng.state.notes, tmp.name, soundfont_path=SOUNDFONT_PATH,
               tempo_bpm=req.tempo_bpm)
    return FileResponse(tmp.name, filename="score.wav", media_type="audio/wav")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
