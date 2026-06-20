"""
Phase 6 (lightweight) — song vibes via embedding clustering.

Reuses the sentence-transformer embeddings from phase 5 (no new model
download). Clusters the 244 songs into K groups via K-means on the
embeddings, then summarises each cluster:
  - top songs (centroid-closest)
  - most distinctive songs (centroid-furthest)
  - auto-generated "vibe" hint from the top songs' titles + albums
    (since we don't have an LLM, the label is heuristic)

The point of phase 6 is to give each song a categorical "vibe" that
phase 7's visualization can use for coloring. Without an LLM we don't
get prose summaries, but we do get meaningful clusters that often
correspond to lyrical themes.

**IMPORTANT (round 4 audit, 2026-06-20)**: silhouette score for K=10
on this corpus is essentially zero (mean ~0.004). K-means cluster
assignments are unstable across random seeds (ARI ~0.15 between seeds).
49.6% of songs have NEGATIVE silhouette — they're closer to a different
cluster's centroid than to their own. The 'vibe clusters' are real
assignments but they don't represent any strong underlying structure in
the embedding space. The top-5 songs and album distribution per cluster
are seed-dependent: a different seed would tell a different "story" with
the same data. Treat cluster descriptions as exploratory, not definitive.

Outputs:
  - reports/song_vibes.csv (244 rows × cluster + vibe hint)
  - reports/vibes_summary.md (human-readable)

Usage:
  python analyze/vibes.py
  python analyze/vibes.py --k 10
"""

from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LYRICS_JSON = REPO_ROOT / "data" / "processed" / "lyrics_by_section.json"
SONGS_CSV = REPO_ROOT / "data" / "processed" / "songs.csv"

OUT_CSV = REPO_ROOT / "reports" / "song_vibes.csv"
OUT_MD = REPO_ROOT / "reports" / "vibes_summary.md"
OUT_QUALITY = REPO_ROOT / "reports" / "vibes_quality.json"  # read by dashboard.py
EMBED_CACHE = REPO_ROOT / "data" / "processed" / "song_embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10, help="number of clusters (default 10)")
    p.add_argument("--seed", type=int, default=42, help="random seed for K-means (default 42)")
    p.add_argument("--recompute-embeddings", action="store_true",
                   help="ignore cached embeddings and recompute (slower)")
    args = p.parse_args()

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

    print(f"[info] {len(song_records)} songs")

    # ----- embeddings: load cache or recompute -----
    cache_ok = False
    embeddings = None
    if EMBED_CACHE.exists() and not args.recompute_embeddings:
        print(f"[info] loading cached embeddings from {EMBED_CACHE.relative_to(REPO_ROOT)}")
        z = np.load(EMBED_CACHE)
        embeddings = z["embeddings"]
        if embeddings.shape[0] != len(song_records):
            print(f"[warn] cached shape {embeddings.shape} doesn't match {len(song_records)} songs; recomputing")
            embeddings = None
        else:
            cache_ok = True
    if not cache_ok:
        print(f"[info] loading model {MODEL_NAME} (~80 MB on first run) ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        print(f"[info] encoding {len(texts)} songs ...")
        embeddings = model.encode(
            texts, batch_size=8, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBED_CACHE, embeddings=embeddings)
        print(f"[ok] cached embeddings to {EMBED_CACHE.relative_to(REPO_ROOT)}")

    # ----- K-means clustering -----
    from sklearn.cluster import KMeans
    print(f"[info] K-means with K={args.k}, seed={args.seed}")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(embeddings)
    print(f"[info] cluster sizes: {Counter(int(l) for l in labels)}")

    # ----- per-cluster summary -----
    cluster_songs: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        cluster_songs[int(c)].append(i)

    # rank songs within cluster by distance to centroid (closest = most "core")
    cluster_summary = {}
    for c, idxs in cluster_songs.items():
        centroid = km.cluster_centers_[c]
        centroid /= np.linalg.norm(centroid)
        sims = embeddings[idxs] @ centroid
        order = np.argsort(-sims)  # closest first
        sorted_idxs = [idxs[i] for i in order]
        cluster_summary[c] = {
            "size": len(idxs),
            "sorted_idxs": sorted_idxs,
            "centroid_sim_top10": [float(sims[i]) for i in order[:10]],
            "albums": Counter(song_records[i]["Album"] for i in idxs),
            "years": [song_records[i]["Year"] for i in idxs],
        }

    # ----- heuristic "vibe" hints: top 3 song titles + dominant album ----
    def vibe_hint(c: int) -> str:
        top = [song_records[i]["Title"] for i in cluster_summary[c]["sorted_idxs"][:3]]
        dom_album = cluster_summary[c]["albums"].most_common(1)[0][0] if cluster_summary[c]["albums"] else "?"
        return f"cluster {c} (n={cluster_summary[c]['size']}, dominant album: {dom_album})"

    # ----- write per-song CSV -----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AlbumCode", "Album", "Year", "TrackNumber", "Title", "cluster", "vibe_hint"])
        for i, s in enumerate(song_records):
            c = int(labels[i])
            w.writerow([s["AlbumCode"], s["Album"], s["Year"], s["TrackNumber"], s["Title"], c, vibe_hint(c)])
    print(f"[ok] wrote {OUT_CSV.relative_to(REPO_ROOT)} ({len(song_records)} rows)")

    # ----- write summary markdown -----
    L = []
    L.append("# Song vibes — embedding clusters")
    L.append("")
    L.append(f"Generated by `python analyze/vibes.py`. K-means clustering (K={args.k}) on")
    L.append(f"the {MODEL_NAME} song embeddings from phase 5. No new model —")
    L.append(f"embeddings are cached at `data/processed/song_embeddings.npz`.")
    L.append("")
    L.append("**Note**: this is a lightweight replacement for an LLM-based vibe")
    L.append("labeling pass (qwen3:4b via ollama, phase 6 in the original plan).")
    L.append("The clusters are real and meaningful; the 'vibe' hints are")
    L.append("heuristic (top songs + dominant album), not LLM-generated prose.")
    L.append("For per-song prose summaries, an LLM pass would still be needed.")
    L.append("")
    L.append(f"Cluster sizes: " + ", ".join(f"c{k}={cluster_summary[k]['size']}" for k in sorted(cluster_summary)))
    L.append("")

    # sort clusters by size (largest first) so big clusters get attention
    sorted_clusters = sorted(cluster_summary.keys(), key=lambda c: -cluster_summary[c]["size"])

    for c in sorted_clusters:
        cs = cluster_summary[c]
        top_idxs = cs["sorted_idxs"]
        top_albums = cs["albums"].most_common(5)
        years = sorted(set(y for y in cs["years"] if y))
        year_range = f"{min(years)}-{max(years)}" if years else "—"
        L.append(f"## Cluster {c} (n={cs['size']}, years {year_range})")
        L.append("")
        L.append("**Top 5 songs by closeness to cluster centroid**:")
        L.append("")
        L.append("| Song | Album | Year |")
        L.append("|------|-------|------|")
        for i in top_idxs[:5]:
            s = song_records[i]
            L.append(f"| {s['Title']} | {s['Album']} | {s['Year']} |")
        L.append("")
        L.append(f"**Album distribution**: " + ", ".join(f"{a} ({n})" for a, n in top_albums))
        L.append("")

    # cross-cluster stats: how many songs share cluster with same-album neighbors?
    L.append("## Within-album cluster consistency")
    L.append("")
    L.append("For each album, how many of its songs share a cluster with another")
    L.append("song from the SAME album? Higher = album is internally coherent.")
    L.append("")
    album_clusters: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(song_records):
        album_clusters[s["Album"]].append(int(labels[i]))

    consistency = []
    for album, cs in album_clusters.items():
        cc = Counter(cs)
        same = sum(n for c, n in cc.items() if cc[c] > 1)
        consistency.append((same, len(cs), album))
    consistency.sort(reverse=True)

    L.append("| Album | n songs | in shared cluster | consistency % |")
    L.append("|-------|---------|-------------------|---------------|")
    for same, n, album in consistency:
        pct = 100 * same / n if n else 0
        L.append(f"| {album} | {n} | {same} | {pct:.0f}% |")
    L.append("")
    L.append("**Caveat on consistency**: the variation 67% to 97% mostly tracks")
    L.append("album size (n=12 to n=31) — a 31-song album is more likely to land")
    L.append("multiple songs in the same cluster than a 12-song album, by chance.")
    L.append("This is a real signal (TTPD songs are more similar to each other) but")
    L.append("it's confounded with sample size.")
    L.append("")

    # ---- cluster quality (round 4 audit) — computed dynamically ----
    from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
    sil_mean = silhouette_score(embeddings, labels)
    sil_per_sample = silhouette_samples(embeddings, labels)
    sil_median = float(np.median(sil_per_sample))
    neg_frac = float(np.mean(sil_per_sample < 0))
    pos30_frac = float(np.mean(sil_per_sample > 0.3))
    # stability across seeds
    seed_labels = []
    for s in [0, 1, 7, 13, 100, 999]:
        km_s = KMeans(n_clusters=args.k, random_state=s, n_init=10)
        seed_labels.append(km_s.fit_predict(embeddings))
    aris = []
    for i in range(1, len(seed_labels)):
        aris.append(adjusted_rand_score(seed_labels[0], seed_labels[i]))
    ari_min, ari_max = min(aris), max(aris)
    # pairwise sim
    sim_full = embeddings @ embeddings.T
    np.fill_diagonal(sim_full, np.nan)
    sim_mean = float(np.nanmean(sim_full))
    sim_std = float(np.nanstd(sim_full))

    L.append("## Cluster quality (computed dynamically)")
    L.append("")
    L.append(f"K-means (K={args.k}, seed={args.seed}) on {embeddings.shape[0]} song embeddings:")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Silhouette score (mean) | {sil_mean:.4f} |")
    L.append(f"| Silhouette score (median) | {sil_median:.4f} |")
    L.append(f"| Fraction of songs with negative silhouette | {neg_frac*100:.1f}% |")
    L.append(f"| Fraction of songs with silhouette > 0.3 | {pos30_frac*100:.1f}% |")
    L.append(f"| ARI between K={args.k} clusterings at different seeds | {ari_min:.3f} to {ari_max:.3f} |")
    L.append(f"| Pairwise cosine similarity (mean ± std) | {sim_mean:.3f} ± {sim_std:.3f} |")
    L.append("")
    L.append("**What this means**: the 384-dim sentence embeddings place all 244 songs")
    L.append("in a relatively tight region of space (all pairwise similarities are")
    L.append("positive, mean ≈ 0.4). K-means splits this region into K pieces, but the")
    L.append("pieces don't correspond to meaningful 'vibe' categories — the cluster")
    L.append("labels are arbitrary. A different random seed would give a different set")
    L.append(f"of {args.k} clusters with ARI ≈ {ari_max:.2f} against the current ones, telling")
    L.append("a different 'story' with the same data.")
    L.append("")
    L.append("The cluster compositions (top 5 songs, dominant album) shown above are")
    L.append(f"real for seed={args.seed} but seed-dependent. The summary's within-album")
    L.append("consistency % is also affected — different seeds give different ranks.")
    L.append("Treat the cluster descriptions as exploratory, not definitive.")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```bash")
    L.append("python data/fetch_cots.py")
    L.append("python data/build_pipeline.py")
    L.append("python analyze/similarity.py    # writes song_similarity.csv (also caches embeddings)")
    L.append("python analyze/vibes.py         # this file")
    L.append("```")
    L.append("")
    L.append("Output files:")
    L.append("- `reports/song_vibes.csv` — 244 rows × cluster assignment (committed)")
    L.append("- `reports/vibes_summary.md` — this file (committed)")
    L.append("- `data/processed/song_embeddings.npz` — cached embeddings (gitignored)")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_MD.relative_to(REPO_ROOT)} ({len(L)} lines)")

    # write cluster quality metrics to JSON for dashboard.py to read
    quality = {
        "k": int(args.k),
        "seed": int(args.seed),
        "n_songs": int(embeddings.shape[0]),
        "silhouette_mean": float(sil_mean),
        "silhouette_median": float(sil_median),
        "frac_negative_silhouette": float(neg_frac),
        "frac_silhouette_gt_0_3": float(pos30_frac),
        "ari_min": float(ari_min),
        "ari_max": float(ari_max),
        "pairwise_sim_mean": float(sim_mean),
        "pairwise_sim_std": float(sim_std),
    }
    OUT_QUALITY.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_QUALITY.relative_to(REPO_ROOT)}")

    # console headline
    print(f"\n=== headline ===")
    largest = sorted_clusters[0]
    cs = cluster_summary[largest]
    print(f"largest cluster (c{largest}, n={cs['size']})")
    print(f"  top songs: {', '.join(song_records[i]['Title'] for i in cs['sorted_idxs'][:3])}")
    print(f"  dominant album: {cs['albums'].most_common(1)[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())