# Taylor Swift's Songs — NLP Analysis

A natural-language analysis of Taylor Swift's lyrics across all 13 studio
albums, exploring word usage, sentiment, emotion, and lyrical structure
across her discography. Built on the [Corpus of Taylor Swift (CoTS)
v1.4](https://github.com/sagesolar/Corpus-of-Taylor-Swift).

## What's in this repo

| File / Dir | Description |
|---|---|
| `data/fetch_cots.py` | Downloads the CoTS files from upstream into `data/raw/cots/` (gitignored). |
| `data/build_pipeline.py` | Reads CoTS + `data/album_to_era.json`, produces the analysis-ready CSVs. |
| `data/album_to_era.json` | Maps CoTS album codes to canonical Album names + Era groupings. |
| `data/raw/cots/` | CoTS files (downloaded, gitignored). |
| `data/processed/albums.csv` | 13 albums with structural and linguistic metadata. |
| `data/processed/songs.csv` | 244 songs with structural counts, per-word pre-classification, and section-tagged lyric counts. |
| `data/processed/lyrics_by_section.json` | Section-tagged lyrics (verse / chorus / bridge / refrain / intro-outro). |
| `data/archive/` | Previous dataset (199 songs, MIT-licensed Kaggle-derived, no era metadata). Kept for reference, not used by the active pipeline. |
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

**This project adds**: an `Era` taxonomy grouping the 13 albums into 4
buckets for career-level trend analysis.

### Era taxonomy

Four buckets, derived from
[Wikipedia's classification](https://en.wikipedia.org/wiki/Taylor_Swift)
at the level of stylistic continuity (not per-album genre):

| Era | Albums | Songs | Notes |
|---|---|---|---|
| **Early career** | Taylor Swift (2006), Fearless (2008), Speak Now (2010), Red (2012) | 89 | Country, then crossover to pop/rock. |
| **Mainstream pop** | 1989 (2014), Reputation (2017), Lover (2019), Midnights (2022), TTPD (2024) | 107 | The synth-pop / pop period (with Reputation as trap-pop, Lover as eclectic pop, TTPD as synth-pop — distinguished at the Album level). |
| **Indie/folk** | Folklore (2020), Evermore (2020) | 34 | The pandemic-era indie-folk detour. |
| **Soft rock** | The Life of a Showgirl (2025) | 12 | Newest release; one-album bucket for now. |
| **Other** | Non-album songs | 2 | Standalone releases not tied to a studio album. |

Album-level distinctions (e.g. Red as "eclectic pop/rock", Reputation as
"trap-pop", TTPD as "synth-pop") are preserved at the Album column level,
not collapsed into the Era column. This trades granular genre labels for
analytically useful groupings that show career-level trends without
one-album buckets.

To refine the taxonomy, edit `data/album_to_era.json` and re-run
`data/build_pipeline.py`.

## License

This project is licensed under **MIT** (see `LICENSE`).

It depends on the [Corpus of Taylor Swift](https://github.com/sagesolar/Corpus-of-Taylor-Swift),
which is **GPL-3.0**. CoTS files are downloaded at runtime and not
committed to this repo, so this project's MIT license is preserved.
See `THIRD_PARTY_LICENSES.md` for the full explanation.

## What's coming

The data infrastructure (phase 1) is in place. The planned analysis
phases, in priority order:

- **Phase 2** — Sentiment analysis (VADER, TextBlob, BERT, RoBERTa emotion),
  per-song and per-album, with interactive visualizations.
- **Phase 3** — Section-level analysis. Per-section sentiment scores,
  emotional arc within each song, "which song has the biggest
  verse→chorus sentiment jump?"
- **Phase 4** — Vocabulary complexity over time (CEFR level distribution,
  type-token ratio per album, album uniqueness / Jaccard similarity).
- **Phase 5** — Topic modeling (BERTopic) and song similarity
  (sentence-transformer embeddings), interactive song similarity graph.
- **Phase 6** — LLM pass with a local small model (qwen3:4b via ollama):
  per-song one-line summaries and "vibe" labels.
- **Phase 7** — Visualization report (interactive plotly html) and
  updated findings.

Each phase will land as a separate commit when ready.

## Legacy

The original 2025-era notebook is preserved at
`notebooks/legacy/Swift_NLP_2025.ipynb` for historical reference. It
uses the archived 199-song dataset (Kaggle-derived, MIT-licensed) and
the older VADER + TextBlob + BERT + RoBERTa-emotion pipeline. It is not
part of the active analysis pipeline.

---

Built by [Zaher Alkaei](https://github.com/zaheralkaei).
