"""
One-shot setup for phase 2 sentiment analysis.
Creates a venv, installs dependencies, downloads model weights.

Usage:
  python analyze/setup.py             # venv + deps + (skip models)
  python analyze/setup.py --with-models  # also pre-download HF model weights

After setup, run sentiment via:
  .venv/bin/python analyze/sentiment.py            # linux/git-bash
  .venv/Scripts/python.exe analyze/sentiment.py    # windows
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)
    if r.returncode != 0:
        print(f"!! command failed (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-models", action="store_true",
                   help="also pre-download BERT + RoBERTa emotion model weights (~500 MB)")
    args = p.parse_args()

    # 1. create venv if missing
    venv_dir = REPO_ROOT / ".venv"
    if not venv_dir.exists():
        print(f"[1/4] creating venv at {venv_dir}")
        run(["uv", "venv", str(venv_dir)])
    else:
        print(f"[1/4] venv exists at {venv_dir}")

    # 2. install deps
    print("[2/4] installing vaderSentiment, textblob ...")
    run(["uv", "pip", "install", "--quiet", "vaderSentiment", "textblob"])

    print("[2/4] installing torch (CPU) ...")
    run(["uv", "pip", "install", "--quiet", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"])

    print("[2/4] installing transformers ...")
    run(["uv", "pip", "install", "--quiet", "transformers"])

    # 3. textblob corpora
    print("[3/4] downloading textblob corpora ...")
    py = venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python"
    run([str(py), "-m", "textblob.download_corpora"])

    # 4. optionally pre-download HF models
    if args.with_models:
        print("[4/4] pre-downloading BERT + RoBERTa emotion models ...")
        run([str(py), "-c",
             "from transformers import AutoTokenizer, AutoModelForSequenceClassification; "
             "for name in ['distilbert-base-uncased-finetuned-sst-2-english', "
             "'j-hartmann/emotion-english-distilroberta-base']: "
             "AutoTokenizer.from_pretrained(name); "
             "AutoModelForSequenceClassification.from_pretrained(name); "
             "print('downloaded', name)"])

    print("\n[ok] setup complete")
    print(f"     next: {py} analyze/sentiment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
