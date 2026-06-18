# Archived data (kept for reference, no longer used)

These files were used by the original `Swift_NLP_2025.ipynb` notebook
and the era-mapping infrastructure that preceded the move to the
**Corpus of Taylor Swift** (CoTS, GPL-3.0).

Kept here as a snapshot of the previous approach.

| File | What it was |
|------|-------------|
| `final_lyrics_with_year.csv` | Original dataset (199 songs, Kaggle-derived, MIT-licensed). Re-release year only — no album/era metadata. |
| `final_lyrics_with_era.csv` | Output of `data/build_dataset.py` (commit `dac26c1` and earlier). Same 199 songs + Album + Era columns. |
| `song_to_album.json` | Manual song→album mapping (199 entries). Replaced by CoTS's album codes (TSW, FER, SPN, RED, NEN, REP, LVR, FOL, EVE, MID, TPD, LSG). |

## why archived, not deleted

The data here is MIT-licensed (your own work or the upstream Kaggle
dataset). No reason to delete; just no longer the active pipeline.

To re-create the era-aware dataset from this archive (historical reference only — the active pipeline does not use these files):

```bash
# (from data/archive/)
# recreate build_dataset.py from git history: git show dac26c1:data/build_dataset.py
python build_dataset.py
```

## replacement approach

The active pipeline uses CoTS v1.4 (244+ songs, 13 albums incl.
reputation + The Life of a Showgirl, pre-classified words, section
tags) — see `data/raw/cots/` after running `data/fetch_cots.py`.

CoTS is GPL-3.0 licensed and downloaded at runtime, not committed.
See `THIRD_PARTY_LICENSES.md` at the repo root.
