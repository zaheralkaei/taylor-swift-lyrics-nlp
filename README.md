# Taylor Swift's Songs — NLP Analysis

A natural-language analysis of Taylor Swift's lyrics across all 13 studio
albums, exploring word usage, sentiment, emotion, and lyrical structure
across her discography. Built on the [Corpus of Taylor Swift (CoTS)
v1.4](https://github.com/sagesolar/Corpus-of-Taylor-Swift).

## What's in this repo

| File / Dir | Description |
|---|---|
| `data/fetch_cots.py` | Downloads the CoTS files from upstream into `data/raw/cots/` (gitignored). |
| `data/build_pipeline.py` | Reads CoTS + `data/album_meta.json`, produces the analysis-ready CSVs. |
| `data/album_meta.json` | Maps CoTS album codes to canonical Album names + Year (for the OTH bucket where CoTS leaves it blank). |
| `data/raw/cots/` | CoTS files (downloaded, gitignored). |
| `data/processed/albums.csv` | 13 albums with structural and linguistic metadata. |
| `data/processed/songs.csv` | 244 songs with structural counts, per-word pre-classification, and section-tagged lyric counts. |
| `data/processed/lyrics_by_section.json` | Section-tagged lyrics (verse / chorus / bridge / refrain / intro-outro). |
| `data/archive/` | Previous dataset (199 songs, MIT-licensed Kaggle-derived). Kept for reference, not used by the active pipeline. |
| `analyze/sentiment.py` | Phase 2 — runs VADER + TextBlob + DistilBERT (SST-2) across all 244 songs. Produces `reports/sentiment_per_song.csv` and `reports/sentiment_per_section.csv`. |
| `analyze/section_analysis.py` | Phase 3 — reads the per-section CSV, summarises section-level patterns. Produces `reports/section_summary.md`. |
| `analyze/vocabulary.py` | Phase 4 — re-tokenizes lyrics and joins with CoTS word-details (CEFR, OEC rank, frequency band). Produces `reports/vocabulary_per_song.csv` and `reports/vocabulary_summary.md`. |
| `analyze/similarity.py` | Phase 5 — encodes lyrics with `all-MiniLM-L6-v2` and finds top-K most similar songs by cosine similarity. Produces `reports/song_similarity.csv` and `reports/similarity_summary.md`. |
| `analyze/vibes.py` | Phase 6a (lightweight) — K-means clusters on the cached embeddings. Produces `reports/song_vibes.csv` and `reports/vibes_summary.md`. |
| `analyze/llm_vibes.py` | Phase 6b (full LLM pass) — calls ollama (`qwen2.5:3b` default) for per-song summary + vibe label. Produces `reports/llm_vibes.csv` and `reports/llm_summary.md`. |
| `analyze/dashboard.py` | Phase 7 — combines phases 2-6 into a self-contained interactive HTML report (`reports/dashboard.html`, ~100 KB, plotly.js via CDN). |
| `reports/dashboard.html` | Phase 7 — interactive dashboard (committed). Open in any browser. |
| `reports/sentiment_summary.md` | Human-readable phase-2 findings (per-album, model disagreements, top/bottom songs). **Committed.** |
| `reports/section_summary.md` | Human-readable phase-3 findings (per-section averages, verse→chorus jumps, bridge analysis). **Committed.** |
| `reports/vocabulary_summary.md` | Human-readable phase-4 findings (per-album CEFR, OEC, MATTR; album-pair Jaccard similarity). **Committed.** |
| `reports/similarity_summary.md` | Human-readable phase-5 findings (mutual nearest pairs, most distinctive/interchangeable songs, per-album centroid similarity). **Committed.** |
| `reports/vibes_summary.md` | Human-readable phase-6 findings (10 K-means clusters, top songs per cluster, within-album consistency). **Committed.** |
| `notebooks/legacy/Swift_NLP_2025.ipynb` | Original 2025-era notebook. Historical reference; uses the archived dataset. |
| `THIRD_PARTY_LICENSES.md` | Documents the GPL-3.0 dependency on CoTS. |
| `LICENSE` | MIT (this project's license). |

## How to run

Two commands. The CoTS data is downloaded at runtime (not committed):

```bash
python data/fetch_cots.py
python data/build_pipeline.py
```

Outputs land in `data/processed/`:

- `albums.csv` — 13 rows, one per studio album (+ 1 OTH bucket)
- `songs.csv` — 244 rows, one per song
- `lyrics_by_section.json` — section-tagged lyrics, keyed by `AlbumCode:TrackNumber:Title`

Run from the repo root.

## Data architecture

**Source**: [Corpus of Taylor Swift v1.4](https://github.com/sagesolar/Corpus-of-Taylor-Swift)
by [sagesolar](https://github.com/sagesolar), GPL-3.0 licensed.

CoTS provides 244 songs across 13 studio albums (and a small "Other Songs"
bucket), with:

- Section-tagged lyrics (verse / chorus / bridge / refrain / intro-outro)
- Pre-classified per-word linguistic metadata (PoS, frequency band, CEFR level, OEC rank)
- Song-level structural counts (lines, verses, bridges, choruses, refrains, intros/outros)
- Album-level aggregates (lines, words, prevalent verb / adjective / noun, lowest-frequency word)

**This project adds**: a clean Album display-name mapping (in
`data/album_meta.json`) so the CoTS codes (TSW, FER, SPN, etc.) become
the canonical release names (Taylor Swift, Fearless, Speak Now, etc.).
Year and structural counts come straight from CoTS.

### Album naming

CoTS uses short codes (`TSW`, `FER`, `SPN`, `RED`, `NEN`, `REP`, `LVR`,
`FOL`, `EVE`, `MID`, `TPD`, `LSG`, `OTH`) and a mix of display names. We
map these to the canonical release names:

| Code | Album | Year |
|---|---|---|
| TSW | Taylor Swift | 2006 |
| FER | Fearless | 2008 |
| SPN | Speak Now | 2010 |
| RED | Red | 2012 |
| NEN | 1989 | 2014 |
| REP | Reputation | 2017 |
| LVR | Lover | 2019 |
| FOL | Folklore | 2020 |
| EVE | Evermore | 2020 |
| MID | Midnights | 2022 |
| TPD | The Tortured Poets Department | 2024 |
| LSG | The Life of a Showgirl | 2025 |
| OTH | Other (non-album) | — |

To refine album names, edit `data/album_meta.json` and re-run
`data/build_pipeline.py`.

## License

This project is licensed under **MIT** (see `LICENSE`).

It depends on the [Corpus of Taylor Swift](https://github.com/sagesolar/Corpus-of-Taylor-Swift),
which is **GPL-3.0**. CoTS files are downloaded at runtime and not
committed to this repo, so this project's MIT license is preserved.
See `THIRD_PARTY_LICENSES.md` for the full explanation.

## What's coming

All seven phases are in place. The combined interactive dashboard is at
[`reports/dashboard.html`](reports/dashboard.html) — open in any browser.

Per-phase summaries (committed):
- [`reports/sentiment_summary.md`](reports/sentiment_summary.md) — phase 2
- [`reports/section_summary.md`](reports/section_summary.md) — phase 3
- [`reports/vocabulary_summary.md`](reports/vocabulary_summary.md) — phase 4
- [`reports/similarity_summary.md`](reports/similarity_summary.md) — phase 5
- [`reports/vibes_summary.md`](reports/vibes_summary.md) — phase 6

Reproducing everything from scratch (assumes data/raw/cots/ is empty):

```bash
python data/fetch_cots.py           # downloads CoTS (~2.3 MB)
python data/build_pipeline.py       # builds songs.csv + albums.csv + lyrics_by_section.json
python analyze/sentiment.py         # phase 2 (~3 min CPU)
python analyze/section_analysis.py  # phase 3 (instant)
python analyze/vocabulary.py        # phase 4 (~30 sec CPU)
python analyze/similarity.py        # phase 5 (~10 sec CPU after model download)
python analyze/vibes.py             # phase 6 (instant, reuses embeddings)
python analyze/dashboard.py         # phase 7 (instant)
```

Each phase will land as a separate commit when ready.

## Legacy

The original 2025-era notebook is preserved at
`notebooks/legacy/Swift_NLP_2025.ipynb` for historical reference. It
uses the archived 199-song dataset (Kaggle-derived, MIT-licensed) and
the older VADER + TextBlob + BERT pipeline (pre-DistilBERT). It is not
part of the active analysis pipeline.

---

Built by [Zaher Alkaei](https://github.com/zaheralkaei).
