# Taylor Swift's Songs — NLP Analysis

A natural-language analysis of Taylor Swift's lyrics, exploring word usage, sentiment, and emotion across her discography. The notebook walks through cleaning, POS tagging, word frequency, and three different sentiment methods (TextBlob, VADER, BERT) plus emotion detection with a RoBERTa model.

## What's in this repo

| File | Description |
|---|---|
| `final_lyrics_with_year.csv` | 199 songs, one per row. Columns: `Song`, `Lyrics`, `Year`. Lyrics are raw — they include section markers like `[Verse 1]`, `[Chorus]`, and embedded newlines. |
| `Swift_NLP_2025.ipynb` | The analysis notebook. Designed for Google Colab but works in any Jupyter environment. |
| `LICENSE` | MIT. |

The dataset is a fork of [this open Kaggle dataset](https://www.kaggle.com/datasets/tksmax/taylorswiftlyrics/data), with one column added: the year of release. For songs that were re-released (Taylor's Version albums), the newer release year is used.

## How to run

The notebook is self-contained. Easiest path:

1. Open `Swift_NLP_2025.ipynb` in [Google Colab](https://colab.research.google.com/) — upload the file or open it from your Drive.
2. Run the cells in order.

The first code cell installs the dependencies, which include `vaderSentiment`, `BERTopic`, `sentence-transformers`, `transformers`, and `torch`. If running locally, you'll want a Python 3.9+ environment with these pre-installed; the first cell will then just be a no-op.

## Data schema

```
Song     string  — the song title
Lyrics   string  — the raw lyrics, including section markers and newlines
Year     integer — release year (Taylor's Version re-releases use the re-release year)
```

A row of cleaned lyrics (after the notebook's `clean_lyrics` function) is in lowercase, with section markers and punctuation stripped, ready for token-level analysis.

## Methodology

The analysis uses four tools:

- **VADER** (Valence Aware Dictionary for sEntiment Reasoning) — a rule-based sentiment tool tuned for short, social-media-style text.
- **TextBlob** — a rule-based tool that returns both *polarity* (-1 to +1) and *subjectivity* (0 to 1).
- **BERT** (distilbert-base-uncased-finetuned-sst-2-english) — a fine-tuned transformer for positive/negative classification with a confidence score.
- **RoBERTa** (j-hartmann/emotion-english-distilroberta-base) — a transformer fine-tuned for emotion detection (joy, sadness, anger, etc.).

POS tagging and stop-word filtering use spaCy.

## Year coverage

The CSV spans 2006–2024 but is sparse in the early years:

| Year | Songs |
|---|---|
| 2006 | 13 |
| 2007 | 1 |
| 2019 | 18 |
| 2020 | 31 |
| 2021 | 52 |
| 2022 | 22 |
| 2023 | 45 |
| 2024 | 17 |

Years 2008–2018 are missing because their songs have not been re-released under "Taylor's Version" yet — the year column uses the most recent release date, so unrevised tracks don't have a 2019+ year attached. The notebook's main analysis filters to **2019 onward (185 songs)** to focus on the most recent eras.

## Highlights

A few findings from the notebook (185 songs, 2019–2024):

- **Most positive song (VADER):** *This Love (Taylor's Version)*
- **Most positive song (TextBlob):** *Bejeweled*
- **Most positive song (BERT):** *The Best Day (Taylor's Version)* — confidence 0.999
- **Most negative song (VADER + TextBlob):** *Shake It Off (Taylor's Version)*
- **Most negative song (BERT):** *illicit affairs* — confidence 1.0
- **Top 5 verbs** (after cleaning): *know*, *think*, *go*, *get*, *come*
- **Top 3 nouns:** *time*, *love*, *baby*

YouTube links for the most-positive and most-negative songs are in the notebook itself.

## What you can do next

Some directions if you want to extend the analysis:

- Add an **era** column (Debut, Fearless, Red, 1989, Reputation, Lover, Folklore/Evermore, Midnights, TTPD) and re-run the sentiment comparisons per era.
- Run **BERTopic** on the cleaned lyrics to discover topic clusters (heartbreak, party, self-reflection, etc.).
- Add **Spotify audio features** (tempo, energy, valence) to cross-reference lyrical sentiment with musical mood.
- Build a **Streamlit / Gradio app** to interactively filter songs by year, era, or sentiment range.

## License

MIT. See [LICENSE](LICENSE).

The lyrics dataset is a derivative of the [Kaggle tksmax/taylorswiftlyrics dataset](https://www.kaggle.com/datasets/tksmax/taylorswiftlyrics/data) — please respect the original source's terms.

---

Built by [Zaher Alkaei](https://github.com/zaheralkaei).
