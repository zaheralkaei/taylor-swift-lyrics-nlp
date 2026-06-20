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

# CoTS v1.4 SHA256 hashes — pin to upstream's main branch as of 2026-06-20.
# Round 9 audit fix: prevents silent data drift if upstream changes.
# Re-generate with `python data/fetch_cots.py --print-sha256` after a
# deliberate upstream update.
EXPECTED_SHA256 = {
    "tsv/cots-album-details.tsv":    "6f5386b58e0cf311745bbc4eb0d27be271a6337ab88a30a746ca8ebfac16a5be",
    "tsv/cots-song-details.tsv":     "935823b8b008b81d383a1c22b4b1fa8e38fb400051d30b7dc7ab24ee5c044444",
    "tsv/cots-word-details.tsv":     "a224ef8f0fe873c6d558508a0171b4d417132bae171b52a13ee11737e097b16b",
    "lyrics/album-song-lyrics.json": "7da321db60722143a1241f836d75fc218008b023e150fdb75333560555c9eb62",
}

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
    """Download a single file. Returns bytes downloaded.

    Verifies SHA256 against EXPECTED_SHA256[path] after download (round 9
    audit fix). Refuses to write a file whose hash doesn't match the pinned
    upstream snapshot — prevents silent data drift.
    """
    url = f"{BASE_URL}/{path}"
    expected = expected_size(path)
    print(f"  fetching {path} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    actual = len(data)
    sha = hashlib.sha256(data).hexdigest()
    expected_sha = EXPECTED_SHA256.get(path)
    if expected_sha and sha != expected_sha:
        print(f"\n  [FAIL] SHA mismatch for {path}", file=sys.stderr)
        print(f"         expected: {expected_sha}", file=sys.stderr)
        print(f"         got:      {sha}", file=sys.stderr)
        print(f"         data has changed since 2026-06-20 snapshot.", file=sys.stderr)
        print(f"         Inspect the diff upstream. To override, edit EXPECTED_SHA256 in fetch_cots.py.", file=sys.stderr)
        return -1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    size_status = "ok" if expected == 0 or expected == actual else f"size mismatch (expected {expected})"
    print(f"{actual:>10,} bytes  sha256:{sha[:16]}  {size_status}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if file exists")
    parser.add_argument("--check", action="store_true", help="check existence + size only")
    parser.add_argument("--verify-sha256", action="store_true",
                       help="check existing files against pinned EXPECTED_SHA256 (no downloads)")
    parser.add_argument("--print-sha256", action="store_true",
                       help="print current upstream file SHAs (one-time, for pinning)")
    args = parser.parse_args()

    if args.print_sha256:
        # print full SHAs for the current upstream snapshot — use this output
        # to update EXPECTED_SHA256 above after a deliberate upstream bump
        for path, desc in FILES:
            url = f"{BASE_URL}/{path}"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                sha = hashlib.sha256(data).hexdigest()
                print(f"  \"{path}\": \"{sha}\",")
            except Exception as e:
                print(f"  [error] could not fetch {path}: {e}", file=sys.stderr)
        return 0

    print(LICENSE_BANNER)
    print(f"destination: {DEST_DIR.relative_to(REPO_ROOT)}/\n")

    if args.verify_sha256:
        print("[verify-sha256 mode — comparing cached files against pinned SHAs]\n")
        ok = True
        for path, desc in FILES:
            dest = DEST_DIR / Path(path).name
            expected_sha = EXPECTED_SHA256.get(path)
            if not dest.exists():
                print(f"  ✗ {dest.name:<40} MISSING")
                ok = False
                continue
            actual_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            if expected_sha and actual_sha == expected_sha:
                print(f"  ✓ {dest.name:<40} sha256 matches")
            else:
                print(f"  ✗ {dest.name:<40} sha256 MISMATCH")
                print(f"      expected: {expected_sha}")
                print(f"      got:      {actual_sha}")
                ok = False
        return 0 if ok else 1

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
        result = download(path, dest)
        if result < 0:
            return 1
        total += result

    print(f"\n[ok] {total:,} bytes total in {DEST_DIR.relative_to(REPO_ROOT)}/")
    print(f"[next] run `python data/build_pipeline.py` to construct the analysis-ready dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
