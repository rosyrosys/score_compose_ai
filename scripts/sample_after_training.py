"""Quick sanity check after training: load weights, sample, save MIDI + MusicXML.

Usage:
    python -m scripts.sample_after_training \
        --weights /content/drive/MyDrive/score_compose_ai/weights.pt \
        --out_midi sample.mid --out_xml sample.musicxml --n_tokens 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as plain script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.checkpoint import load_for_inference
from src.export import export_midi
from src.generate import generate
from src.score_io import notes_to_musicxml
from src.tokenizer import BOS_ID, decode_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_midi", default="sample.mid")
    ap.add_argument("--out_xml", default="sample.musicxml")
    ap.add_argument("--n_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--tempo", type=float, default=110.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _cfg = load_for_inference(args.weights, device=device)

    ids = generate(model, [BOS_ID], max_new_tokens=args.n_tokens,
                   temperature=args.temperature, top_p=args.top_p)
    notes = decode_tokens(ids)
    print(f"generated {len(notes)} notes from {len(ids)} tokens")

    export_midi(notes, args.out_midi, tempo_bpm=args.tempo)
    Path(args.out_xml).write_text(notes_to_musicxml(notes), encoding="utf-8")
    print(f"wrote {args.out_midi}, {args.out_xml}")


if __name__ == "__main__":
    main()
