"""Run me on Google Colab.

   !git clone <this repo>
   %cd score_compose_ai
   !pip install -r requirements.txt
   !python scripts/prepare_lakh_subset.py --out data/midi
   !python -m src.train --midi_dir data/midi --epochs 8 --batch_size 16
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.train",
                "--midi_dir", "data/midi",
                "--out", "weights.pt",
                "--epochs", "8",
                "--batch_size", "16",
                "--seq_len", "1024"], check=True)
