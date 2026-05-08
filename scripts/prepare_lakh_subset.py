"""Download and extract a MIDI dataset for training.

Two sources are supported:

  --source maestro   (default) MAESTRO v3.0.0 piano MIDI, ~80MB, hosted by
                     Google Magenta. Reliable, small, well-suited to
                     classical-piano-style sheet output.
                     URL: https://magenta.tensorflow.org/datasets/maestro

  --source lmd       LMD-clean (Lakh MIDI Dataset). ~250MB, multi-genre,
                     multi-instrument. Hosted on Columbia's hog server,
                     historically intermittent.
                     URL: https://colinraffel.com/projects/lmd/

Cite respectively:
  Hawthorne et al. ICLR 2019 (MAESTRO)
  Raffel, Columbia PhD thesis 2016 (Lakh MIDI Dataset)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/"
    "v3.0.0/maestro-v3.0.0-midi.zip"
)
LMD_URLS = [
    "https://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz",
    "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz",
]


def _download(url: str, out: Path, fallbacks=()) -> Path:
    urls = [url, *fallbacks]
    last_err = None
    for u in urls:
        try:
            print(f"downloading {u}")
            urllib.request.urlretrieve(u, out)
            return out
        except Exception as e:
            print(f"  failed: {e}")
            last_err = e
    raise RuntimeError(f"all download URLs failed; last error: {last_err}")


def _extract_zip(archive: Path, out_dir: Path, max_files: int) -> int:
    n = 0
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.endswith((".mid", ".midi", ".MID", ".MIDI"))]
        for m in members[:max_files]:
            target = out_dir / Path(m).name
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
            if n % 200 == 0:
                print(f"  extracted {n}/{min(max_files, len(members))}")
    return n


def _extract_tar(archive: Path, out_dir: Path, max_files: int) -> int:
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith((".mid", ".midi", ".MID", ".MIDI")):
                continue
            if n >= max_files:
                break
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                target = out_dir / f"{n:06d}.mid"
                with open(target, "wb") as dst:
                    shutil.copyfileobj(f, dst)
                n += 1
                if n % 500 == 0:
                    print(f"  extracted {n}/{max_files}")
            except Exception as e:
                print(f"  skip {m.name}: {e}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["maestro", "lmd"], default="maestro")
    ap.add_argument("--out", default="data/midi")
    ap.add_argument("--max_files", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "maestro":
        archive = out.parent / "maestro.zip"
        if not archive.exists():
            _download(MAESTRO_URL, archive)
        print("extracting MAESTRO ...")
        n = _extract_zip(archive, out, args.max_files)
    else:
        archive = out.parent / "clean_midi.tar.gz"
        if not archive.exists():
            _download(LMD_URLS[0], archive, fallbacks=LMD_URLS[1:])
        print("extracting LMD-clean ...")
        n = _extract_tar(archive, out, args.max_files)

    print(f"done. {n} files in {out}")
    if n == 0:
        raise SystemExit("no MIDI files extracted; check archive integrity")


if __name__ == "__main__":
    main()
