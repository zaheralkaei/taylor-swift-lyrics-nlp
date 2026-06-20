"""
Compute per-song and per-section sentiment scores for every song
in data/processed/songs.csv.

Models (loaded one at a time to minimize peak memory):
  - VADER: rule-based sentiment, range [-1, 1]
  - TextBlob: rule-based polarity + subjectivity
  - DistilBERT (distilbert-base-uncased-finetuned-sst-2-english): positive/negative classifier

Note: the earlier pipeline also scored a 7-way emotion classifier
(j-hartmann/emotion-english-distilroberta-base). It was removed because
the model is trained on short-form reviews/dialogue and produces
near-uniform distributions on abstract lyrical text — not a reliable
signal. DistilBERT (positive/negative) is kept because it transfers well.

Inputs:
  - data/processed/songs.csv
  - data/processed/lyrics_by_section.json

Outputs:
  - reports/sentiment_per_song.csv       (244 rows × scores)
  - reports/sentiment_per_section.csv   (one row per song × section)

Usage:
  python analyze/sentiment.py                 # full run
  python analyze/sentiment.py --limit 10      # first 10 songs only (smoke test)
  python analyze/sentiment.py --no-bert       # rule-based only (fast, no model download)
"""

from __future__ import annotations
import argparse
import csv
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SONGS_CSV = REPO_ROOT / "data" / "processed" / "songs.csv"
LYRICS_JSON = REPO_ROOT / "data" / "processed" / "lyrics_by_section.json"
OUT_DIR = REPO_ROOT / "reports"


SECTION_GROUPS = {
    "verse":   ["Verse"],
    "chorus":  ["Chorus"],
    "bridge":  ["Bridge"],
    "refrain": ["Refrain"],
    # CoTS uses the merged key 'IntroOutro' (not separate 'Intro'/'Outro' tags),
    # so we map all three possible names to keep working across CoTS versions.
    "in_out":  ["IntroOutro", "Intro", "Outro", "Spoken Outro"],
}


def clean_for_sentiment(text: str) -> str:
    """Strip structural markers for sentiment analysis.

    Drops bracketed structural labels like [Verse], [Bridge], [Spoken] (these
    are CoTS metadata, not sung lyrics). Keeps parenthesised vocalisations
    like (Ah-ah-ah), (Hey, hey, hey) — these are sung content, not metadata,
    and dropping them silently wipes intros for songs whose intro is purely
    vocalisation (e.g. Castles Crumbling, Invisible String).
    """
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lyrics_for_song(key: str, lyrics_by_song: dict) -> list[dict]:
    return lyrics_by_song.get(key, [])


def section_text(lines: list[dict], section: str) -> str:
    """Concatenate lines belonging to a single section group."""
    parts = [ln["Text"] for ln in lines if ln.get("SongPart") in SECTION_GROUPS[section]]
    return clean_for_sentiment(" ".join(parts))


def all_text(lines: list[dict]) -> str:
    return clean_for_sentiment(" ".join(ln["Text"] for ln in lines))


def compute_vader(text: str, sia) -> dict:
    if not text:
        return {"vader_neg": None, "vader_neu": None, "vader_pos": None, "vader_compound": None}
    s = sia.polarity_scores(text)
    return {"vader_neg": s["neg"], "vader_neu": s["neu"], "vader_pos": s["pos"], "vader_compound": s["compound"]}


def compute_textblob(text: str) -> dict:
    if not text:
        return {"tb_polarity": None, "tb_subjectivity": None}
    from textblob import TextBlob
    b = TextBlob(text).sentiment
    return {"tb_polarity": b.polarity, "tb_subjectivity": b.subjectivity}


def compute_bert(text: str, tokenizer, model, device, chunk_tokens: int = 480) -> dict:
    """distilbert-sst2: returns positive/negative probabilities.

    For texts longer than 512 tokens (the model's max), use chunk-and-mean-pool:
    split into ~480-token overlapping chunks and average the per-chunk softmax
    probabilities. This is methodologically cleaner than silent truncation
    (which biases bert_pos toward verse/intro sentiment for long songs).
    """
    if not text:
        return {"bert_pos": None, "bert_neg": None}
    import torch
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    n_tokens = len(ids)
    if n_tokens <= 512:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return {"bert_neg": float(probs[0]), "bert_pos": float(probs[1])}
    # mean-pool over chunks
    all_probs = []
    stride = chunk_tokens  # non-overlapping chunks
    for start in range(0, n_tokens, stride):
        end = min(start + 512, n_tokens)
        chunk_ids = ids[start:end].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids=chunk_ids).logits
        all_probs.append(torch.softmax(logits, dim=-1)[0].cpu())
    mean_probs = torch.stack(all_probs).mean(dim=0).tolist()
    return {"bert_neg": float(mean_probs[0]), "bert_pos": float(mean_probs[1])}


# (compute_roberta_emotion was removed 2026-06-18 — model was too noisy on lyrics.
# See the module docstring for details.)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="process only the first N songs (smoke test)")
    p.add_argument("--no-bert", action="store_true", help="skip BERT and RoBERTa (rule-based only)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading {SONGS_CSV.relative_to(REPO_ROOT)} ...")
    with SONGS_CSV.open(encoding="utf-8") as f:
        songs = list(csv.DictReader(f))
    print(f"[info] {len(songs)} songs")

    with LYRICS_JSON.open(encoding="utf-8") as f:
        lyrics_by_song = json.load(f)
    print(f"[info] {len(lyrics_by_song)} songs have section-tagged lyrics\n")

    if args.limit:
        songs = songs[:args.limit]
        print(f"[info] --limit {args.limit}: processing first {len(songs)} songs only\n")

    # ----- phase A: rule-based (vader + textblob) -----
    print("=" * 60)
    print("PHASE A — rule-based (VADER + TextBlob)")
    print("=" * 60)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    per_song_rows = []
    per_section_rows = []

    for i, s in enumerate(songs):
        key = f"{s['AlbumCode']}:{int(s['TrackNumber']):02d}:{s['Title']}"
        lines = lyrics_for_song(key, lyrics_by_song)
        full_text = all_text(lines)

        row = {
            "AlbumCode": s["AlbumCode"], "Album": s["Album"], "Year": s["Year"],
            "Title": s["Title"], "TrackNumber": s["TrackNumber"],
            "WordCount": s["Words"],
        }
        row.update(compute_vader(full_text, sia))
        row.update(compute_textblob(full_text))

        # per-section scores (only if we have sections for this song)
        for section in SECTION_GROUPS:
            sec_text = section_text(lines, section)
            if not sec_text:
                continue
            sec_row = {
                "AlbumCode": s["AlbumCode"], "Album": s["Album"],
                "Title": s["Title"], "TrackNumber": s["TrackNumber"],
                "Section": section,
                "SectionCharCount": len(sec_text),
            }
            sec_row.update(compute_vader(sec_text, sia))
            sec_row.update(compute_textblob(sec_text))
            per_section_rows.append(sec_row)

        per_song_rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(songs):
            print(f"  [{i+1:>3}/{len(songs)}] {s['Title'][:40]:<40}  vader={row['vader_compound']:+.3f}  tb={row['tb_polarity']:+.3f}")

    # write rule-based results immediately
    out_a = OUT_DIR / "sentiment_per_song.csv"
    fieldnames_a = list(per_song_rows[0].keys())
    with out_a.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_a)
        w.writeheader()
        w.writerows(per_song_rows)
    print(f"\n[ok] wrote {len(per_song_rows)} rows to {out_a.relative_to(REPO_ROOT)}")

    if per_section_rows:
        out_s = OUT_DIR / "sentiment_per_section.csv"
        fieldnames_s = list(per_section_rows[0].keys())
        with out_s.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_s)
            w.writeheader()
            w.writerows(per_section_rows)
        print(f"[ok] wrote {len(per_section_rows)} rows to {out_s.relative_to(REPO_ROOT)}")

    if args.no_bert:
        print("\n[info] --no-bert: skipping transformer models")
        return 0

    # ----- phase B: BERT (distilbert-sst2) -----
    print("\n" + "=" * 60)
    print("PHASE B — BERT (distilbert-sst2, positive/negative)")
    print("=" * 60)
    print("[info] loading model (downloads ~250 MB on first run) ...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    bert_name = "distilbert-base-uncased-finetuned-sst-2-english"
    bert_tok = AutoTokenizer.from_pretrained(bert_name)
    bert_mod = AutoModelForSequenceClassification.from_pretrained(bert_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_mod.to(device)
    bert_mod.eval()
    print(f"[ok] loaded in {time.time()-t0:.1f}s, device={device}")

    for i, s in enumerate(songs):
        key = f"{s['AlbumCode']}:{int(s['TrackNumber']):02d}:{s['Title']}"
        lines = lyrics_for_song(key, lyrics_by_song)
        full_text = all_text(lines)
        row = per_song_rows[i]
        row.update(compute_bert(full_text, bert_tok, bert_mod, device))
        if (i + 1) % 25 == 0 or i + 1 == len(songs):
            print(f"  [{i+1:>3}/{len(songs)}] {s['Title'][:40]:<40}  bert_pos={row['bert_pos']:.3f}")

    # rewrite the per_song csv with bert columns added
    with out_a.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_song_rows[0].keys()))
        w.writeheader()
        w.writerows(per_song_rows)
    print(f"\n[ok] updated {out_a.relative_to(REPO_ROOT)} with BERT columns")

    # also score each section with BERT so per-section analysis (verse vs chorus,
    # verse->chorus jump) has the same neural signal we trust at the song level.
    print("\n[info] scoring each section with DistilBERT (per-section) ...")
    # build (song_idx, section_name) -> sec_row index once (avoids O(n^2) inner search)
    sec_index: dict[tuple[int, str], dict] = {}
    for sr in per_section_rows:
        # find the song_idx by matching AlbumCode + TrackNumber + Title
        # (search songs list once per row, but only as a one-time cost)
        for i, s in enumerate(songs):
            if (sr["AlbumCode"] == s["AlbumCode"]
                and sr["TrackNumber"] == s["TrackNumber"]
                and sr["Title"] == s["Title"]):
                sec_index[(i, sr["Section"])] = sr
                break

    for i, s in enumerate(songs):
        key = f"{s['AlbumCode']}:{int(s['TrackNumber']):02d}:{s['Title']}"
        lines = lyrics_for_song(key, lyrics_by_song)
        for section in SECTION_GROUPS:
            sec_text = section_text(lines, section)
            if not sec_text:
                continue
            sec_row = sec_index.get((i, section))
            if sec_row is None:
                continue
            sec_row.update(compute_bert(sec_text, bert_tok, bert_mod, device))
        if (i + 1) % 25 == 0 or i + 1 == len(songs):
            print(f"  [{i+1:>3}/{len(songs)}] sections scored")

    # rewrite the per_section csv with bert columns added
    with out_s.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_section_rows[0].keys()))
        w.writeheader()
        w.writerows(per_section_rows)
    print(f"[ok] updated {out_s.relative_to(REPO_ROOT)} with BERT columns")

    # free BERT
    del bert_tok, bert_mod
    gc.collect()

    print("\n[done] sentiment analysis complete (RoBERTa-emotion skipped — too noisy on lyrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
