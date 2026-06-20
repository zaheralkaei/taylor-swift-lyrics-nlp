"""
Phase 6 (full) — LLM pass with ollama.

For each of the 244 songs, calls an ollama-served model with the
lyrics and asks for:
  1. A one-line summary (max 25 words)
  2. A vibe label (3-5 words, comma-separated tags)

Output as JSON for easy parsing.

Uses the local ollama HTTP API (http://localhost:11434) — the
service is already running on ai-laptop.

Outputs:
  - reports/llm_vibes.csv (244 rows × summary, vibe, model, latency)
  - reports/llm_summary.md (human-readable)

Usage:
  python analyze/llm_vibes.py
  python analyze/llm_vibes.py --model qwen3:4b --limit 5   # pilot
  python analyze/llm_vibes.py --model gpt-oss:20b          # slower, higher quality
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LYRICS_JSON = REPO_ROOT / "data" / "processed" / "lyrics_by_section.json"
SONGS_CSV = REPO_ROOT / "data" / "processed" / "songs.csv"

OUT_CSV = REPO_ROOT / "reports" / "llm_vibes.csv"
OUT_MD = REPO_ROOT / "reports" / "llm_summary.md"

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


SYSTEM_PROMPT = """You are a careful music analyst. Given song lyrics, produce a brief summary and vibe label as JSON.

Rules:
- Summary: max 25 words, captures the song's emotional core and central image.
- Vibe: 3-5 comma-separated words capturing mood/genre/imagery.
- Output ONLY the JSON object, no commentary, no markdown fences.
- Schema: {"summary": "...", "vibe": "tag1, tag2, tag3"}"""


USER_TEMPLATE = """Song: {title} ({album}, {year})

Lyrics:
{lyrics}

Respond with JSON only."""


def call_ollama(model: str, prompt: str, timeout: int = 180) -> tuple[str, float]:
    """Call ollama /api/generate and return (response_text, latency_seconds)."""
    payload = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 120,
            "top_p": 0.9,
        },
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    latency = time.time() - t0
    return body.get("response", "").strip(), latency


def parse_response(text: str) -> tuple[str, str]:
    """Best-effort parse of model JSON output. Returns (summary, vibe)."""
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # try direct JSON
    try:
        obj = json.loads(text)
        return str(obj.get("summary", "")).strip(), str(obj.get("vibe", "")).strip()
    except json.JSONDecodeError:
        pass
    # try to find a JSON object within the text
    m = re.search(r"\{[^{}]*\"summary\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return str(obj.get("summary", "")).strip(), str(obj.get("vibe", "")).strip()
        except json.JSONDecodeError:
            pass
    # last resort: pull out summary and vibe fields with regex
    sm = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
    vm = re.search(r'"vibe"\s*:\s*"([^"]+)"', text)
    if sm and vm:
        return sm.group(1).strip(), vm.group(1).strip()
    return "", text[:200]  # return raw as fallback


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"ollama model name (default {DEFAULT_MODEL})")
    p.add_argument("--limit", type=int, default=None, help="process only first N songs (pilot)")
    p.add_argument("--max-lyrics-chars", type=int, default=4000,
                   help="truncate lyrics at this many chars (default 4000)")
    args = p.parse_args()

    songs = list(csv.DictReader(SONGS_CSV.open(encoding="utf-8")))
    with LYRICS_JSON.open(encoding="utf-8") as f:
        lyrics_by_song = json.load(f)

    song_records = []
    for s in songs:
        try:
            track_int = int(s["TrackNumber"])
        except (ValueError, TypeError):
            track_int = 0
        key = f"{s['AlbumCode']}:{track_int:02d}:{s['Title']}"
        lines = lyrics_by_song.get(key, [])
        text = " ".join(ln.get("Text", "") for ln in lines)
        song_records.append({
            "AlbumCode": s["AlbumCode"], "Album": s["Album"], "Year": s["Year"],
            "TrackNumber": track_int, "Title": s["Title"],
            "WordCount": s["Words"], "lyrics": text, "key": key,
        })

    if args.limit:
        song_records = song_records[:args.limit]
    print(f"[info] processing {len(song_records)} songs with model={args.model}")

    # ----- run inference -----
    rows = []
    total_latency = 0.0
    fails = 0
    t_start = time.time()
    for i, s in enumerate(song_records):
        lyrics = s["lyrics"]
        if len(lyrics) > args.max_lyrics_chars:
            lyrics = lyrics[:args.max_lyrics_chars] + "..."
        prompt = USER_TEMPLATE.format(
            title=s["Title"], album=s["Album"], year=s["Year"], lyrics=lyrics,
        )
        try:
            raw, lat = call_ollama(args.model, prompt)
            summary, vibe = parse_response(raw)
            total_latency += lat
            ok = bool(summary)
            # additional warning: non-Latin content
            non_latin_chars = sum(1 for c in (summary + vibe)
                                  if 0x3000 <= ord(c) <= 0x9FFF or 0x0400 <= ord(c) <= 0x04FF)
            if non_latin_chars > 0:
                print(f"  [warn] {s['Title']:<30} ({s['Album']}): {non_latin_chars} non-Latin chars in output")
        except Exception as e:
            raw, lat, summary, vibe = str(e), 0.0, "", ""
            ok = False
            fails += 1

        rows.append({
            "AlbumCode": s["AlbumCode"], "Album": s["Album"], "Year": s["Year"],
            "TrackNumber": s["TrackNumber"], "Title": s["Title"],
            "summary": summary, "vibe": vibe,
            "model": args.model, "latency_sec": round(lat, 2),
            "raw_response": raw[:500], "ok": ok,
        })
        if (i + 1) % 5 == 0 or i + 1 == len(song_records):
            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            eta = avg * (len(song_records) - i - 1)
            print(f"  [{i+1:>3}/{len(song_records)}] {s['Title'][:30]:<30}  lat={lat:.1f}s  avg={avg:.1f}s  ETA={eta/60:.1f}min  fails={fails}")

    # ----- write CSV -----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(rows)} rows, {fails} failed)")

    # ----- write summary markdown -----
    avg_lat = total_latency / max(1, len(rows) - fails)
    ok_rows = [r for r in rows if r["ok"]]

    L = []
    L.append(f"# LLM pass — vibe + summary ({args.model})")
    L.append("")
    L.append(f"Generated by `python analyze/llm_vibes.py --model {args.model}`.")
    L.append("")
    L.append("**What this is**: each song's lyrics are sent to a local ollama-")
    L.append(f"served LLM ({args.model}). The model is asked for a one-line")
    L.append("summary (≤25 words) and a 3-5-word vibe label, returned as JSON.")
    L.append("")
    L.append("**Stats**:")
    L.append(f"- Songs processed: {len(rows)} ({fails} parse failures)")
    L.append(f"- Avg inference latency: {avg_lat:.1f} sec/song")
    L.append(f"- Total wall time: {(time.time()-t_start)/60:.1f} min")
    L.append("")

    # per-album vibe table
    by_album = {}
    for r in ok_rows:
        by_album.setdefault(r["Album"], []).append(r)

    L.append("## Per-album vibe word cloud")
    L.append("")
    L.append("Most common vibe tags per album (top 8 each).")
    L.append("")
    for album in sorted(by_album, key=lambda a: int(by_album[a][0]["Year"]) if by_album[a][0]["Year"].isdigit() else 9999):
        tags = []
        for r in by_album[album]:
            for t in r["vibe"].split(","):
                t = t.strip().lower()
                if t: tags.append(t)
        from collections import Counter
        top = Counter(tags).most_common(8)
        tag_str = ", ".join(f"{t} ({n})" for t, n in top)
        L.append(f"- **{album}** ({by_album[album][0]['Year']}): {tag_str}")
    L.append("")

    # sample summaries
    L.append(f"## Sample summaries (first 10 songs)")
    L.append("")
    L.append("| Song | Album | Summary | Vibe |")
    L.append("|------|-------|---------|------|")
    for r in ok_rows[:10]:
        L.append(f"| {r['Title']} | {r['Album']} | {r['summary']} | {r['vibe']} |")
    L.append("")

    L.append("## Failed parses")
    L.append("")
    if fails:
        L.append(f"{fails} songs failed to parse cleanly. Sample raw responses:")
        L.append("")
        for r in rows:
            if not r["ok"]:
                L.append(f"- **{r['Title']}** ({r['Album']}): `{r['raw_response'][:150]}...`")
    else:
        L.append("None.")
    L.append("")

    # non-Latin content detection (script-generated)
    L.append("## Honest caveats")
    L.append("")
    L.append(f"- The model is {args.model} (q4_K_M), a small general-purpose")
    L.append("  chat model. Outputs are plausible but not authoritative.")
    non_latin_summary = [r for r in ok_rows
                         if any(0x3000 <= ord(c) <= 0x9FFF or 0x0400 <= ord(c) <= 0x04FF
                                for c in r["summary"])]
    non_latin_vibe = [r for r in ok_rows
                      if any(0x3000 <= ord(c) <= 0x9FFF or 0x0400 <= ord(c) <= 0x04FF
                             for c in r["vibe"])]
    if non_latin_summary:
        names = ", ".join(f"**{r['Title']}** ({r['Album']})" for r in non_latin_summary)
        L.append(f"- {len(non_latin_summary)} song(s) returned a non-Latin summary that")
        L.append(f"  may be unusable for English readers: {names}.")
    if non_latin_vibe:
        names = ", ".join(f"**{r['Title']}** ({r['Album']})" for r in non_latin_vibe)
        L.append(f"- {len(non_latin_vibe)} song(s) had non-Latin tokens in the vibe")
        L.append(f"  field (sampling artifact at temperature=0.3): {names}.")
    L.append("- 'Other' bucket has only 2 songs; the per-album vibe word cloud for")
    L.append("  'Other' is noise and not interpretable.")
    L.append("")
    L.append("- The Year column was wrong for 4 albums in the previous version of this")
    L.append("  table (Fearless 2008, Red 2012, 1989 2014, Speak Now 2010 — but CoTS")
    L.append("  reported the Taylor's Version re-release years 2021, 2021, 2023, 2023).")
    L.append("  build_pipeline.py now prefers album_meta.json's canonical years.")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```bash")
    L.append("# ollama service must be running on :11434")
    L.append(f"python analyze/llm_vibes.py --model {args.model}")
    L.append("```")
    L.append("")
    L.append("Output files (gitignored, regenerable):")
    L.append("- `reports/llm_vibes.csv` — 244 rows × summary, vibe, latency")
    L.append("- `reports/llm_summary.md` — this file")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_MD.relative_to(REPO_ROOT)} ({len(L)} lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())