"""
Build the era-aware dataset from the original (year-only) CSV and a manual album mapping.

Reads:
  - ../final_lyrics_with_year.csv (original, year-only)
  - song_to_album.json            (manual mapping: song title -> album)

Writes:
  - processed/final_lyrics_with_era.csv (new dataset with Year, Album, Era columns)

Re-run this whenever:
  - a new Taylor's Version is released
  - a song needs reclassification
  - the original CSV is updated with new songs

Usage:
  python data/build_dataset.py
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "final_lyrics_with_year.csv"
MAPPING_JSON = REPO_ROOT / "data" / "song_to_album.json"
OUT_CSV = REPO_ROOT / "data" / "processed" / "final_lyrics_with_era.csv"


def main() -> int:
    with MAPPING_JSON.open(encoding="utf-8") as f:
        mapping_doc = json.load(f)

    albums = mapping_doc["_albums"]
    song_to_album = mapping_doc["_mappings"]

    # sanity-check the mapping covers all albums
    for album_name, album_info in albums.items():
        if album_info.get("_note"):
            print(f"[note] {album_name}: {album_info['_note']}")

    if not RAW_CSV.exists():
        print(f"[error] missing raw CSV: {RAW_CSV}")
        return 1

    # read original CSV
    with RAW_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[info] loaded {len(rows)} rows from data/raw/{RAW_CSV.name}")

    # validate + transform
    unmapped: list[tuple[str, int]] = []
    out_rows: list[dict] = []
    for row in rows:
        song = row["Song"]
        year = int(row["Year"])
        album = song_to_album.get(song)
        if album is None:
            unmapped.append((song, year))
            continue
        if album not in albums:
            print(f"[error] mapping for '{song}' points to unknown album '{album}'")
            return 2
        era = albums[album]["era"]
        out_rows.append({
            "Song": song,
            "Lyrics": row["Lyrics"],
            "Year": year,
            "Album": album,
            "Era": era,
        })

    # report unmapped songs (if any)
    if unmapped:
        print(f"\n[warning] {len(unmapped)} song(s) not in song_to_album.json:")
        for s, y in unmapped:
            print(f"  {y}: {s}")
        print("\n[fix] edit data/song_to_album.json and add the missing entries,")
        print("       then re-run this script.")
        return 3

    # write output CSV (sorted by album year then song for stable diffs)
    album_year = {a: info["year"] for a, info in albums.items()}
    out_rows.sort(key=lambda r: (album_year[r["Album"]], r["Album"], r["Song"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Song", "Lyrics", "Year", "Album", "Era"])
        writer.writeheader()
        writer.writerows(out_rows)

    # summary
    print(f"\n[ok] wrote {len(out_rows)} rows to {OUT_CSV.relative_to(REPO_ROOT)}")

    from collections import Counter
    print("\n[ok] per-album counts:")
    for album in sorted(albums.keys(), key=lambda a: album_year[a]):
        n = sum(1 for r in out_rows if r["Album"] == album)
        if n > 0:
            print(f"  {album} ({album_year[album]}): {n} songs")
        else:
            print(f"  {album} ({album_year[album]}): 0 songs  <-- gap in dataset")

    print("\n[ok] per-era counts:")
    for era in albums["Debut"]["era"], albums["Red"]["era"], albums["folklore"]["era"], albums["Lover"]["era"], albums["TTPD"]["era"]:
        n = sum(1 for r in out_rows if r["Era"] == era)
        print(f"  {era}: {n} songs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
