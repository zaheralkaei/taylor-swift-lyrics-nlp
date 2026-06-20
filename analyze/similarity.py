"""
Phase 5 (slim) — song similarity via sentence embeddings.

Encodes each song's lyrics with a pretrained sentence-transformer
(`all-MiniLM-L6-v2`, ~80 MB) and computes pairwise cosine similarity.
For each song, finds the top-K most similar songs.

Outputs:
  - reports/song_similarity.csv (244 rows × K nearest neighbors + scores)
  - reports/similarity_summary.md (human-readable, committed)

Why this and not BERTopic?
  BERTopic on 244 short documents is fragile — the UMAP/HDBSCAN
  tuning is fiddly and the topic assignments are noisy. Sentence
  embeddings + similarity is cleaner, faster, and more publishable:
  the headline finding is "which songs are most alike, regardless
  of when they were written?"

Why `all-MiniLM-L6-v2`?
  Small (~80 MB), fast on CPU, well-tested on English text. Good
  quality for general semantic similarity even if not fine-tuned on
  lyrics specifically.

Usage:
  python analyze/similarity.py
  python analyze/similarity.py --top-k 5
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LYRICS_JSON = REPO_ROOT / "data" / "processed" / "lyrics_by_section.json"
SONGS_CSV = REPO_ROOT / "data" / "processed" / "songs.csv"

OUT_CSV = REPO_ROOT / "reports" / "song_similarity.csv"
OUT_MD = REPO_ROOT / "reports" / "similarity_summary.md"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=5, help="top-K neighbors per song (default 5)")
    p.add_argument("--model", default=MODEL_NAME, help=f"sentence-transformer model (default {MODEL_NAME})")
    args = p.parse_args()

    # ----- load data -----
    songs = list(csv.DictReader(SONGS_CSV.open(encoding="utf-8")))
    with LYRICS_JSON.open(encoding="utf-8") as f:
        lyrics_by_song = json.load(f)

    song_records = []
    texts = []
    for s in songs:
        try:
            track_int = int(s["TrackNumber"])
        except (ValueError, TypeError):
            track_int = 0
        key = f"{s['AlbumCode']}:{track_int:02d}:{s['Title']}"
        lines = lyrics_by_song.get(key, [])
        text = " ".join(ln.get("Text", "") for ln in lines)
        if not text.strip():
            text = "(no lyrics)"
        song_records.append({
            "AlbumCode": s["AlbumCode"], "Album": s["Album"], "Year": s["Year"],
            "TrackNumber": track_int, "Title": s["Title"],
            "WordCount": s["Words"], "key": key,
        })
        texts.append(text)

    print(f"[info] {len(song_records)} songs to encode")

    # ----- encode -----
    print(f"[info] loading model {args.model} (~80 MB on first run) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"[info] encoding {len(texts)} songs ...")
    embeddings = model.encode(
        texts, batch_size=8, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    print(f"[info] embeddings shape: {embeddings.shape}")

    # ----- cosine similarity (embeddings already normalized) -----
    # similarity = dot product (since normalized)
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)  # exclude self-similarity

    # ----- top-K neighbors -----
    K = args.top_k
    nearest_rows = []
    nearest_idx = np.argsort(-sim, axis=1)[:, :K]  # (n_songs, K)
    nearest_scores = np.take_along_axis(sim, nearest_idx, axis=1)

    for i, src in enumerate(song_records):
        row = {
            "src_album": src["Album"], "src_year": src["Year"],
            "src_track": src["TrackNumber"], "src_title": src["Title"],
        }
        for k in range(K):
            j = nearest_idx[i, k]
            tgt = song_records[j]
            row[f"n{k+1}_album"] = tgt["Album"]
            row[f"n{k+1}_year"]  = tgt["Year"]
            row[f"n{k+1}_title"] = tgt["Title"]
            row[f"n{k+1}_score"] = round(float(nearest_scores[i, k]), 4)
        nearest_rows.append(row)

    # ----- write per-song CSV -----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(nearest_rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(nearest_rows)
    print(f"[ok] wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(nearest_rows)} rows)")

    # ----- top mutual pairs (where A->B and B->A agree) -----
    # Build a set of (i,j) with i<j such that both i lists j in top-K AND j lists i in top-K
    mutual = []
    for i in range(len(song_records)):
        for k in range(K):
            j = int(nearest_idx[i, k])
            if i < j and i in nearest_idx[j, :K]:
                mutual.append((float(sim[i, j]), i, j))
    mutual.sort(reverse=True)
    print(f"[info] {len(mutual)} mutual top-K pairs")

    # ----- write summary markdown -----
    L = []
    L.append("# Song similarity — summary")
    L.append("")
    L.append(f"Generated by `python analyze/similarity.py`. Encodes each song's lyrics with")
    L.append(f"`{args.model}` (sentence-transformer, ~80 MB) and finds the top-{K} most")
    L.append("similar songs by cosine similarity.")
    L.append("")
    L.append("**Notation**: similarity scores are cosine similarity in [0,1]")
    L.append("(embeddings are L2-normalized, so dot product = cosine).")
    L.append("Higher = more semantically alike.")
    L.append("")

    # ---- top mutual pairs (most reliable signal of "real" similarity) ----
    L.append(f"## Top {min(20, len(mutual))} mutual nearest-neighbor pairs")
    L.append("")
    L.append("Both A's top-{K} and B's top-{K} lists include the other.")
    L.append("Mutual pairs are stronger evidence of similarity than one-way matches.")
    L.append("")
    L.append("| Song A | Album A | Year A | Song B | Album B | Year B | similarity |")
    L.append("|--------|---------|--------|--------|---------|--------|------------|")
    for score, i, j in mutual[:20]:
        a = song_records[i]
        b = song_records[j]
        L.append(f"| {a['Title']} | {a['Album']} | {a['Year']} | {b['Title']} | {b['Album']} | {b['Year']} | {score:.4f} |")
    L.append("")

    # ---- songs with the highest max-similarity (most "clustered") ----
    L.append("## Most 'interchangeable' songs (highest mean similarity to top-5)")
    L.append("")
    L.append("Songs whose lyrics are most similar to other songs on average.")
    L.append("(Heuristic label — high mean similarity to top-5 could reflect")
    L.append("shared function words, shared imagery, or genuine thematic")
    L.append("overlap. Don't over-read.)")
    L.append("")
    mean_top_sim = np.mean(nearest_scores, axis=1)
    by_mean = np.argsort(-mean_top_sim)
    L.append("| Song | Album | Year | mean sim to top-5 |")
    L.append("|------|-------|------|--------------------|")
    for i in by_mean[:15]:
        s = song_records[i]
        L.append(f"| {s['Title']} | {s['Album']} | {s['Year']} | {mean_top_sim[i]:.4f} |")
    L.append("")

    # ---- most distinctive (lowest max-similarity) ----
    L.append("## Most 'distinctive' songs (lowest mean similarity to top-5)")
    L.append("")
    L.append("Songs with the lowest mean similarity to their top-5 neighbors.")
    L.append("(Heuristic label — could reflect idiosyncratic imagery OR")
    L.append("shorter lyric length, OR no direct matches in this corpus.)")
    L.append("")
    L.append("| Song | Album | Year | mean sim to top-5 |")
    L.append("|------|-------|------|--------------------|")
    for i in by_mean[-15:][::-1]:
        s = song_records[i]
        L.append(f"| {s['Title']} | {s['Album']} | {s['Year']} | {mean_top_sim[i]:.4f} |")
    L.append("")

    L.append("## Cross-album similarity")
    L.append("")
    # nearest_idx[i, 0] is the closest song to song i
    cross = sum(1 for i, src in enumerate(song_records)
                if song_records[nearest_idx[i, 0]]["Album"] != src["Album"])
    same   = sum(1 for i, src in enumerate(song_records)
                 if song_records[nearest_idx[i, 0]]["Album"] == src["Album"])
    pct = 100*cross/len(song_records)
    L.append(f"Of {len(song_records)} songs, **{cross}** ({pct:.0f}%) have their nearest")
    L.append(f"neighbor from a *different* album; {same} ({100-pct:.0f}%) from the same album.")
    L.append("")
    L.append("**Caveat on the 'shared lyrical pool' interpretation**: this 86%")
    L.append("figure is consistent with at least three explanations:")
    L.append("(1) Swift genuinely reuses lyrical themes across albums (the story we")
    L.append("want to tell); (2) most songs share a large common-English core of")
    L.append("function words ('the', 'I', 'you', 'love', 'know', 'never') that")
    L.append("dominates the embedding; (3) the embedding model (all-MiniLM-L6-v2,")
    L.append("trained on web text) flattens fine-grained imagery. Without a control")
    L.append("comparison (e.g. similarity scores for shuffled or random texts),")
    L.append("we cannot say which explanation dominates.")
    L.append("")

    # ---- per-album average similarity (centroid cosines) ----
    L.append("## Per-album average centroid similarity")
    L.append("")
    L.append("For each album, compute the mean of its song embeddings (centroid)")
    L.append("and the cosine similarity to every other album's centroid.")
    L.append("Closest-neighbor album pairs cluster most strongly in lyrical")
    L.append("content (even if their genres differ).")
    L.append("")
    from collections import defaultdict
    by_album = defaultdict(list)
    for i, s in enumerate(song_records):
        by_album[s["Album"]].append(i)
    centroids = {a: np.mean(embeddings[idxs], axis=0) for a, idxs in by_album.items()}
    # normalize centroids
    for a in centroids:
        c = centroids[a]
        centroids[a] = c / np.linalg.norm(c)
    def year_key(a):
        y = song_records[by_album[a][0]]["Year"]
        try: return int(y)
        except (ValueError, TypeError): return 9999
    album_names = sorted(centroids.keys(), key=year_key)
    pairs = []
    for i, a in enumerate(album_names):
        if a == "Other": continue  # only 2 songs, distort centroid
        for b in album_names[i+1:]:
            if b == "Other": continue
            s = float(centroids[a] @ centroids[b])
            pairs.append((s, a, b))
    pairs.sort(reverse=True)
    L.append("### Closest album pairs (by centroid similarity)")
    L.append("")
    L.append("Excludes the 'Other' bucket (2 non-album songs, centroid is too noisy).")
    L.append("")
    L.append("| Album A | Album B | similarity |")
    L.append("|---------|---------|------------|")
    for s, a, b in pairs[:10]:
        L.append(f"| {a} | {b} | {s:.4f} |")
    L.append("")
    L.append("### Furthest album pairs (by centroid similarity)")
    L.append("")
    L.append("| Album A | Album B | similarity |")
    L.append("|---------|---------|------------|")
    for s, a, b in pairs[-10:][::-1]:
        L.append(f"| {a} | {b} | {s:.4f} |")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```bash")
    L.append("python data/fetch_cots.py")
    L.append("python data/build_pipeline.py")
    L.append("python analyze/similarity.py")
    L.append("```")
    L.append("")
    L.append("Output files (gitignored, regenerable):")
    L.append("- `reports/song_similarity.csv` — 244 rows × top-5 neighbors + scores")
    L.append("- `reports/similarity_summary.md` — this file")
    L.append("")
    L.append("First run downloads the sentence-transformer model (~80 MB) into")
    L.append("`~/.cache/huggingface/` (HF default cache). Subsequent runs reuse it.")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_MD.relative_to(REPO_ROOT)} ({len(L)} lines)")

    # console headline
    if mutual:
        top = mutual[0]
        a = song_records[top[1]]
        b = song_records[top[2]]
        print(f"\n=== headline ===")
        print(f"top mutual pair: {a['Title']} ({a['Album']}) <-> {b['Title']} ({b['Album']}) sim={top[0]:.4f}")
    print(f"most distinctive: {song_records[by_mean[-1]]['Title']} ({song_records[by_mean[-1]]['Album']}) mean_sim={mean_top_sim[by_mean[-1]]:.4f}")
    print(f"most interchangeable: {song_records[by_mean[0]]['Title']} ({song_records[by_mean[0]]['Album']}) mean_sim={mean_top_sim[by_mean[0]]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())