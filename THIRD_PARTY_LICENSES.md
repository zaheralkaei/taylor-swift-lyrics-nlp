# Third-Party Licenses

This project depends on, links to, or redistributes metadata from
the following third-party works.

## Corpus of Taylor Swift (CoTS) v1.4

- **Upstream**: <https://github.com/sagesolar/Corpus-of-Taylor-Swift>
- **Author**: [sagesolar](https://github.com/sagesolar)
- **License**: GNU General Public License v3.0 (GPL-3.0)
  - Full text: <https://www.gnu.org/licenses/gpl-3.0.txt>
  - SPDX: <https://spdx.org/licenses/GPL-3.0.html>
- **Used for**:
  - Album-level metadata (codes, titles, years, line/word counts)
  - Song-level metadata (structural counts, prevalent words)
  - Per-word linguistic classification (PoS, frequency band, CEFR level, OEC rank)
  - Section-tagged lyrics (verse / chorus / bridge / refrain / intro-outro)
- **How we use it**: CoTS files are **downloaded at runtime** by
  `data/fetch_cots.py` from the upstream repository. They are not
  committed to this repo, and are not redistributed. If you re-run
  the analysis, you agree to CoTS's GPL-3.0 license terms by
  downloading the files yourself from upstream.

## Kaggle `tksmax/taylorswiftlyrics` (only via archived data)

- **Upstream**: <https://www.kaggle.com/datasets/tksmax/taylorswiftlyrics/data>
- **License**: per upstream Kaggle terms
- **Used for**: the **archived** `data/archive/final_lyrics_with_year.csv`
  (199 songs). This file is no longer used by the active pipeline;
  it is preserved as a snapshot of the previous approach.

## Genius (linked, not scraped)

- The CoTS dataset includes `GeniusUrl` links for each song (e.g.
  `https://genius.com/Taylor-swift-tim-mcgraw-lyrics`). We do not
  scrape or redistribute Genius content; the URLs are reference only.

# what this means for this project's license

This project itself is licensed under **MIT** (see `LICENSE` at the
repo root). MIT is compatible with depending on, linking to, and
running against GPL-licensed works — provided the GPL works are not
statically combined into this project's distributed artifacts. We
download CoTS at runtime; we do not redistribute it; therefore this
project's MIT license is preserved.

If you fork this project and choose to commit CoTS files into your
fork, you take on the GPL-3.0 obligation for those specific files
(but not necessarily for the rest of the fork — consult a lawyer
for edge cases).
