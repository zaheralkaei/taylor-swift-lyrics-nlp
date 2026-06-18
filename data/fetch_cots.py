"""
Fetch the Corpus-of-Taylor-Swift (CoTS) v1.4 files.

Downloads CoTS TSVs and lyrics JSON from upstream (github.com/sagesolar)
into data/raw/cots/. CoTS is GPL-3.0 licensed — see THIRD_PARTY_LICENSES.md.
Files are NOT committed to this repo; they live only in your local
working tree (and are gitignored).

Usage:
  python data/fetch_cots.py            # downloads if missing
  python data/fetch_cots.py --force    # re-download even if present
  python data/fetch_cots.py --check    # check existence + size only, no download
"""

from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw" / "cots"

# CoTS v1.4 files we want.
# - tsv/cots-*-details.tsv: song/album/word metadata, easy to parse
# - lyrics/album-song-lyrics.json: section-tagged lyrics (verse, chorus, etc.)
FILES = [
    ("tsv/cots-album-details.tsv",      "album metadata (13 albums)"),
    ("tsv/cots-song-details.tsv",       "song metadata (~244 songs)"),
    ("tsv/cots-word-details.tsv",       "word metadata (pre-classified words)"),
    ("lyrics/album-song-lyrics.json",   "section-tagged lyrics"),
]

BASE_URL = "https://raw.githubusercontent.com/sagesolar/Corpus-of-Taylor-Swift/main"

# GPL-3.0 banner shown to anyone running the script.
LICENSE_BANNER = """\
NOTICE: Corpus of Taylor Swift (CoTS) v1.4 by sagesolar is licensed under
        GNU General Public License v3.0 (GPL-3.0).

By running this script you are downloading GPL-3.0 licensed material from
the upstream repository. CoTS files will be stored in data/raw/cots/ and are
NOT committed to this repository. See THIRD_PARTY_LICENSES.md for details.
"""


def expected_size(path: str) -> int:
    """HEAD request to fetch the upstream file size for verification."""
    url = f"{BASE_URL}/{path}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"  [warn] could not fetch size for {path}: {e}", file=sys.stderr)
        return 0


def download(path: str, dest: Path) -> int:
    """Download a single file. Returns bytes downloaded."""
    url = f"{BASE_URL}/{path}"
    expected = expected_size(path)
    print(f"  fetching {path} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    actual = len(data)
    sha = hashlib.sha256(data).hexdigest()[:16]
    status = "ok" if expected == 0 or expected == actual else f"size mismatch (expected {expected})"
    print(f"{actual:>10,} bytes  sha256:{sha}  {status}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if file exists")
    parser.add_argument("--check", action="store_true", help="check existence + size only")
    args = parser.parse_args()

    print(LICENSE_BANNER)
    print(f"destination: {DEST_DIR.relative_to(REPO_ROOT)}/\n")

    if args.check:
        print("[check mode — no downloads]")
        for path, desc in FILES:
            dest = DEST_DIR / Path(path).name
            if dest.exists():
                size = dest.stat().st_size
                print(f"  ✓ {dest.name:<40} {size:>10,} bytes  ({desc})")
            else:
                print(f"  ✗ {dest.name:<40} MISSING  ({desc})")
        return 0

    total = 0
    for path, desc in FILES:
        dest = DEST_DIR / Path(path).name
        if dest.exists() and not args.force:
            size = dest.stat().st_size
            print(f"  ✓ {dest.name:<40} {size:>10,} bytes  (cached, use --force to re-fetch)")
            total += size
            continue
        total += download(path, dest)

    print(f"\n[ok] {total:,} bytes total in {DEST_DIR.relative_to(REPO_ROOT)}/")
    print(f"[next] run `python data/build_pipeline.py` to construct the analysis-ready dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
