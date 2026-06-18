"""
Build the analysis-ready dataset from Corpus-of-Taylor-Swift (CoTS).

Inputs (downloaded by data/fetch_cots.py, gitignored):
  - data/raw/cots/cots-album-details.tsv
  - data/raw/cots/cots-song-details.tsv
  - data/raw/cots/album-song-lyrics.json
  - data/album_to_era.json (era taxonomy)

Outputs (gitignored, regenerated on every run):
  - data/processed/songs.csv        - one row per song with all metadata
  - data/processed/albums.csv       - one row per album with all metadata
  - data/processed/lyrics_by_section.json - section-tagged lyrics, keyed by song

Run after `python data/fetch_cots.py`.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COTS_DIR = REPO_ROOT / "data" / "raw" / "cots"
ALBUM_TO_ERA = REPO_ROOT / "data" / "album_to_era.json"
OUT_DIR = REPO_ROOT / "data" / "processed"


def read_tsv(path: Path) -> list[dict]:
    """Read a TSV with a quoted header. Returns list of dicts."""
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        return [dict(r) for r in reader]


def coerce_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def main() -> int:
    # sanity: cots files must exist
    needed = [
        COTS_DIR / "cots-album-details.tsv",
        COTS_DIR / "cots-song-details.tsv",
        COTS_DIR / "album-song-lyrics.json",
    ]
    for p in needed:
        if not p.exists():
            print(f"[error] missing {p.relative_to(REPO_ROOT)}")
            print(f"        run: python data/fetch_cots.py")
            return 1

    with ALBUM_TO_ERA.open(encoding="utf-8") as f:
        era_doc = json.load(f)
    album_to_era = era_doc["_albums"]

    # ---- albums ----
    print(f"[info] reading {COTS_DIR.name}/cots-album-details.tsv ...")
    raw_albums = read_tsv(COTS_DIR / "cots-album-details.tsv")
    albums_out = []
    for row in raw_albums:
        code = row["Code"]
        meta = album_to_era.get(code, {"album": code, "year": None, "era": "Other"})
        albums_out.append({
            "AlbumCode": code,
            "Album":     meta["album"],
            "Year":      coerce_int(row.get("Year")) or meta.get("year"),
            "Era":       meta["era"],
            "Songs":     coerce_int(row.get("Songs")),
            "Lines":     coerce_int(row.get("Lines")),
            "Words":     coerce_int(row.get("Words")),
            "LowestFreqWord":        row.get("LowestFqWord", ""),
            "PrevalentVerb":         row.get("PrevalentVerb", ""),
            "PrevalentAdjective":    row.get("PrevalentAdjective", ""),
            "PrevalentNoun":         row.get("PrevalentNoun", ""),
        })

    # ---- songs ----
    print(f"[info] reading {COTS_DIR.name}/cots-song-details.tsv ...")
    raw_songs = read_tsv(COTS_DIR / "cots-song-details.tsv")
    songs_out = []
    for row in raw_songs:
        code = row["Album"]
        meta = album_to_era.get(code, {"album": code, "year": None, "era": "Other"})
        songs_out.append({
            "AlbumCode":    code,
            "Album":        meta["album"],
            "Year":         meta.get("year"),
            "Era":          meta["era"],
            "TrackNumber":  coerce_int(row.get("Track")),
            "Title":        row.get("Title", ""),
            "FeaturedArtists": row.get("FeaturedArtists", ""),
            "FromTheVault": row.get("FromTheVault", ""),
            "Lines":        coerce_int(row.get("Lines")),
            "Verses":       coerce_int(row.get("Verses")),
            "Bridges":      coerce_int(row.get("Bridges")),
            "Choruses":     coerce_int(row.get("Choruses")),
            "Refrains":     coerce_int(row.get("Refrains")),
            "InOuts":       coerce_int(row.get("InOuts")),
            "Words":        coerce_int(row.get("Words")),
            "LowestFreqWord":     row.get("LowestFqWord", ""),
            "PrevalentVerb":      row.get("PrevalentVerb", ""),
            "PrevalentAdjective": row.get("PrevalentAdjective", ""),
            "PrevalentNoun":      row.get("PrevalentNoun", ""),
            "GeniusUrl":          row.get("GeniusUrl", ""),
        })

    # ---- lyrics ----
    print(f"[info] reading {COTS_DIR.name}/album-song-lyrics.json ...")
    with (COTS_DIR / "album-song-lyrics.json").open(encoding="utf-8") as f:
        lyrics_raw = json.load(f)

    lyrics_by_song: dict[str, list[dict]] = {}
    lyric_section_counts: dict[str, dict[str, int]] = {}
    for album in lyrics_raw:
        code = album["Code"]
        for song in album.get("Songs", []):
            key = f"{code}:{song['TrackNumber']:02d}:{song['Title']}"
            lines = []
            section_counts: dict[str, int] = {}
            for lyric in song.get("Lyrics", []):
                part = lyric.get("SongPart", "Unknown")
                section_counts[part] = section_counts.get(part, 0) + 1
                lines.append({
                    "Order":    lyric.get("Order"),
                    "Text":     lyric.get("Text", ""),
                    "SongPart": part,
                })
            lyrics_by_song[key] = lines
            lyric_section_counts[key] = section_counts

    # augment songs_out with lyric-derived counts
    for s in songs_out:
        key = f"{s['AlbumCode']}:{s['TrackNumber']:02d}:{s['Title']}"
        sc = lyric_section_counts.get(key, {})
        s["LyricLines"]     = sum(sc.values())
        s["VerseLines"]     = sc.get("Verse", 0)
        s["ChorusLines"]    = sc.get("Chorus", 0)
        s["BridgeLines"]    = sc.get("Bridge", 0)
        s["RefrainLines"]   = sc.get("Refrain", 0)
        s["InOutLines"]     = sc.get("Intro", 0) + sc.get("Outro", 0) + sc.get("Spoken Outro", 0)
        s["OtherLines"]     = s["LyricLines"] - s["VerseLines"] - s["ChorusLines"] - s["BridgeLines"] - s["RefrainLines"] - s["InOutLines"]
        s["HasSectionTags"] = s["LyricLines"] > 0

    # ---- write outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    albums_csv = OUT_DIR / "albums.csv"
    fieldnames_a = ["AlbumCode", "Album", "Year", "Era", "Songs", "Lines", "Words",
                    "LowestFreqWord", "PrevalentVerb", "PrevalentAdjective", "PrevalentNoun"]
    albums_out.sort(key=lambda r: (r["Year"] or 9999, r["AlbumCode"]))
    with albums_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_a)
        w.writeheader()
        w.writerows(albums_out)
    print(f"[ok] wrote {len(albums_out)} albums to {albums_csv.relative_to(REPO_ROOT)}")

    songs_csv = OUT_DIR / "songs.csv"
    fieldnames_s = ["AlbumCode", "Album", "Year", "Era", "TrackNumber", "Title",
                    "FeaturedArtists", "FromTheVault",
                    "Lines", "Verses", "Bridges", "Choruses", "Refrains", "InOuts",
                    "Words",
                    "LyricLines", "VerseLines", "ChorusLines", "BridgeLines",
                    "RefrainLines", "InOutLines", "OtherLines", "HasSectionTags",
                    "LowestFreqWord", "PrevalentVerb", "PrevalentAdjective", "PrevalentNoun",
                    "GeniusUrl"]
    songs_out.sort(key=lambda r: ((r["Year"] or 9999), r["AlbumCode"], r["TrackNumber"] or 0))
    with songs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_s)
        w.writeheader()
        w.writerows(songs_out)
    print(f"[ok] wrote {len(songs_out)} songs to {songs_csv.relative_to(REPO_ROOT)}")

    lyrics_json = OUT_DIR / "lyrics_by_section.json"
    with lyrics_json.open("w", encoding="utf-8") as f:
        json.dump(lyrics_by_song, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {len(lyrics_by_song)} songs' section-tagged lyrics to {lyrics_json.relative_to(REPO_ROOT)}")

    # ---- summary ----
    print("\n=== summary ===")
    print(f"  albums: {len(albums_out)}")
    print(f"  songs:  {len(songs_out)}")
    tagged = sum(1 for s in songs_out if s["HasSectionTags"])
    print(f"  songs with section tags: {tagged}/{len(songs_out)}")

    print("\n=== per-album counts ===")
    for a in albums_out:
        marker = "" if a["Songs"] else "  <-- no songs?"
        print(f"  {a['AlbumCode']:<4} {a['Album']:<22} ({a['Year']}) [{a['Era']:<15}] {a['Songs']:>3} songs{marker}")

    print("\n=== per-era counts ===")
    from collections import Counter
    era_counts = Counter(a["Era"] for a in albums_out)
    era_song_counts = Counter(s["Era"] for s in songs_out)
    for era in era_doc["_eras"]:
        print(f"  {era:<18} {era_counts.get(era, 0):>2} albums  {era_song_counts.get(era, 0):>3} songs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
