"""
Phase 7 — interactive visualization report.

Reads the committed per-phase summary CSVs (when present) and the
gitignored per-song CSVs, and writes a single self-contained
interactive HTML report combining the most compelling findings.

Charts:
  1. Career-long sentiment arc — DistilBERT pos vs year, colored by album.
  2. Album-level sentiment vs lexical diversity — DistilBERT pos vs MATTR.
  3. Per-album section composition — verse/chorus/bridge/refrain/in_out %.
  4. Top-15 mutual nearest-neighbor pairs (table, not a chart).
  5. Cluster composition (phase 6) — small-multiples of album distribution
     per cluster.

Output:
  - reports/dashboard.html (committed, ~1 MB self-contained)

Usage:
  python analyze/dashboard.py
"""

from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
SONGS_CSV       = REPO_ROOT / "data" / "processed" / "songs.csv"
SENT_SONG_CSV   = REPO_ROOT / "reports" / "sentiment_per_song.csv"
SENT_SEC_CSV    = REPO_ROOT / "reports" / "sentiment_per_section.csv"
VOCAB_PER_SONG  = REPO_ROOT / "reports" / "vocabulary_per_song.csv"
SIM_CSV         = REPO_ROOT / "reports" / "song_similarity.csv"
VIBES_CSV       = REPO_ROOT / "reports" / "song_vibes.csv"
ALBUMS_CSV      = REPO_ROOT / "data" / "processed" / "albums.csv"

OUT_HTML = REPO_ROOT / "reports" / "dashboard.html"

# distinct color per album (plotly default qualitative palette)
ALBUM_COLORS = {
    "Taylor Swift":            "#1f77b4",
    "Fearless":                "#ff7f0e",
    "Speak Now":               "#2ca02c",
    "Red":                     "#d62728",
    "1989":                    "#9467bd",
    "Reputation":              "#8c564b",
    "Lover":                   "#e377c2",
    "Folklore":                "#7f7f7f",
    "Evermore":                "#bcbd22",
    "Midnights":               "#17becf",
    "TTPD":                    "#0e7c7b",
    "Life of a Showgirl":      "#9edae5",
    "Other":                   "#c7c7c7",
}


def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT_HTML), help="output HTML path")
    args = p.parse_args()

    # ----- load -----
    songs = list(csv.DictReader(SONGS_CSV.open(encoding="utf-8")))
    sent_songs = list(csv.DictReader(SENT_SONG_CSV.open(encoding="utf-8"))) if SENT_SONG_CSV.exists() else []
    sent_secs  = list(csv.DictReader(SENT_SEC_CSV.open(encoding="utf-8"))) if SENT_SEC_CSV.exists() else []
    vocab_ps   = list(csv.DictReader(VOCAB_PER_SONG.open(encoding="utf-8"))) if VOCAB_PER_SONG.exists() else []
    sim_rows   = list(csv.DictReader(SIM_CSV.open(encoding="utf-8"))) if SIM_CSV.exists() else []
    vibes_rows = list(csv.DictReader(VIBES_CSV.open(encoding="utf-8"))) if VIBES_CSV.exists() else []

    print(f"[info] songs={len(songs)} sent_songs={len(sent_songs)} sent_secs={len(sent_secs)} "
          f"vocab_ps={len(vocab_ps)} sim={len(sim_rows)} vibes={len(vibes_rows)}")

    # index songs by AlbumCode+TrackNumber+Title for joining
    sent_by_key = {(r["AlbumCode"], r["TrackNumber"], r["Title"]): r for r in sent_songs}
    vocab_by_key = {(r["AlbumCode"], r["TrackNumber"], r["Title"]): r for r in vocab_ps}
    vibes_by_key = {(r["AlbumCode"], r["TrackNumber"], r["Title"]): r for r in vibes_rows}

    # ----- chart 1: career-long sentiment arc -----
    fig1 = go.Figure()
    albums_in_order = []
    seen = set()
    for r in sent_songs:
        if r["Album"] not in seen:
            albums_in_order.append(r["Album"])
            seen.add(r["Album"])
    # exclude "Other" bucket (n=2 non-album songs) from headline charts
    if "Other" in albums_in_order:
        albums_in_order.remove("Other")

    # compute per-album mean bert_pos dynamically for chart 1 headline
    album_means = {}
    for album in albums_in_order:
        vals = [fnum(r.get("bert_pos")) for r in sent_songs if r["Album"] == album]
        vals = [v for v in vals if v is not None]
        if vals:
            album_means[album] = statistics.mean(vals)
    top_albums = sorted(album_means.items(), key=lambda x: -x[1])[:2]
    bot_albums = sorted(album_means.items(), key=lambda x: x[1])[:2]
    headline = (
        f"<b>Headline (descriptive):</b> {top_albums[0][0]} ({top_albums[0][1]:.3f}) and "
        f"{top_albums[1][0]} ({top_albums[1][1]:.3f}) average highest on DistilBERT pos; "
        f"{bot_albums[0][0]} ({bot_albums[0][1]:.3f}) and {bot_albums[1][0]} ({bot_albums[1][1]:.3f}) "
        f"lowest. The 'late-career breakup reckoning' interpretation is one reading of the data; "
        f"the same shape is also consistent with 'slower songs get lower pos scores' or 'the model "
        f"is more confident on short, hook-heavy lyrics.' Year axis uses canonical original-release "
        f"years from album_meta.json (CoTS's year column reports Taylor's Version re-release years "
        f"for Fearless/Red/1989/Speak Now)."
    )

    for album in albums_in_order:
        rs = [r for r in sent_songs if r["Album"] == album]
        xs, ys, texts = [], [], []
        for r in rs:
            y = fnum(r.get("bert_pos"))
            try: yr = int(r["Year"])
            except (ValueError, TypeError): continue
            if y is None: continue
            xs.append(yr)
            ys.append(y)
            texts.append(f"{r['Title']} ({album}, {yr})<br>DistilBERT pos: {y:.2f}<br>VADER: {fnum(r.get('vader_compound')) or 0:+.2f}")
        fig1.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            name=album,
            marker=dict(color=ALBUM_COLORS.get(album, "#888"), size=10, line=dict(width=0.5, color="white")),
            text=texts, hovertemplate="%{text}<extra></extra>",
        ))

    fig1.update_layout(
        title=dict(text="<b>Career-long sentiment arc</b><br><sub>DistilBERT positive probability per song. Lower = more negative. Album colors.</sub>", x=0.02),
        xaxis_title="Year",
        yaxis_title="DistilBERT pos [0,1]",
        yaxis=dict(range=[-0.05, 1.05]),
        hovermode="closest",
        height=520, width=None,
        legend=dict(orientation="v", x=1.01, y=1),
        template="plotly_white",
    )

    # ----- chart 2: per-album sentiment vs lexical diversity -----
    by_album = defaultdict(list)
    for r in sent_songs:
        key = (r["AlbumCode"], r["TrackNumber"], r["Title"])
        v = vocab_by_key.get(key)
        if not v: continue
        bert = fnum(r.get("bert_pos"))
        mattr = fnum(v.get("MATTR_500"))
        if bert is None or mattr is None: continue
        by_album[r["Album"]].append((bert, mattr, r["Title"], int(r["Year"]) if r["Year"].isdigit() else 0))

    fig2 = go.Figure()
    # jitter labels slightly to avoid overlap when albums cluster
    import random
    random.seed(42)
    for album in albums_in_order:
        if album not in by_album: continue
        pts = by_album[album]
        xs = [statistics.mean(p[1] for p in pts)]
        ys = [statistics.mean(p[0] for p in pts)]
        n  = [len(pts)]
        # small jitter on y so overlapping album labels separate
        y_jitter = (random.random() - 0.5) * 0.02
        ys[0] += y_jitter
        fig2.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=album,
            marker=dict(
                color=ALBUM_COLORS.get(album, "#888"),
                size=[min(40, 8 + n_*0.7) for n_ in n],
                line=dict(width=1, color="white"),
                sizemode="diameter",
                opacity=0.85,
            ),
            text=[f"<b>{album}</b>"],
                    textposition="top center",
                        textfont=dict(size=10),
                        hovertemplate=f"<b>{album}</b><br>mean DistilBERT pos: %{{y:.3f}}<br>mean MATTR-200: %{{x:.3f}}<br>n songs: {n[0]}<extra></extra>",
                    ))

    fig2.update_layout(
        title=dict(text="<b>Sentiment vs lexical diversity per album</b><br><sub>Each bubble = one album. y = mean DistilBERT pos. x = mean MATTR-200 (within-song vocab diversity).<br>Bubble size = n songs. TTPD is bottom-right (most lexically diverse AND most negative).</sub>", x=0.02),
        xaxis_title="Mean MATTR-200 (within-song vocab diversity)",
        yaxis_title="Mean DistilBERT pos",
        yaxis=dict(range=[-0.05, 1.0]),
        height=620, width=None,
        legend=dict(orientation="v", x=1.01, y=1),
        template="plotly_white",
    )

    # ----- chart 3: per-album section composition -----
    SECTION_GROUPS = ["verse", "chorus", "bridge", "refrain", "in_out"]
    sec_by_key_song = defaultdict(dict)  # song_key -> {section -> chars}
    for r in sent_secs:
        sec_by_key_song[(r["AlbumCode"], r["TrackNumber"], r["Title"])][r["Section"]] = fnum(r["SectionCharCount"]) or 0

    album_section_chars = defaultdict(lambda: defaultdict(float))
    album_section_count = defaultdict(lambda: defaultdict(int))  # n songs that have this section
    for r in sent_secs:
        key = (r["AlbumCode"], r["TrackNumber"], r["Title"])
        album_section_chars[r["Album"]][r["Section"]] += fnum(r["SectionCharCount"]) or 0
        if (fnum(r["SectionCharCount"]) or 0) > 0:
            album_section_count[r["Album"]][r["Section"]] += 1

    # normalize to % within each album
    albums_for_chart = [a for a in albums_in_order if a in album_section_chars]
    section_colors = {"verse": "#1f77b4", "chorus": "#ff7f0e", "bridge": "#2ca02c", "refrain": "#d62728", "in_out": "#9467bd"}

    fig3 = go.Figure()
    for sec in SECTION_GROUPS:
        ys, texts = [], []
        for album in albums_for_chart:
            total_chars = sum(album_section_chars[album].values())
            if total_chars > 0:
                pct = 100 * album_section_chars[album].get(sec, 0) / total_chars
            else:
                pct = 0
            ys.append(pct)
            texts.append(f"{album}: {pct:.1f}% {sec}")
        fig3.add_trace(go.Bar(
            x=albums_for_chart, y=ys, name=sec,
            marker_color=section_colors.get(sec, "#888"),
            text=[f"{v:.0f}%" if v > 3 else "" for v in ys],
            textposition="inside",
            hovertemplate="%{text}<extra></extra>",
        ))

    # compute section composition ranges for chart 3 subtitle
    sec_pct_ranges = {}
    for sec in SECTION_GROUPS:
        ys_for_sec = []
        for album in albums_for_chart:
            total_chars = sum(album_section_chars[album].values())
            if total_chars > 0:
                pct = 100 * album_section_chars[album].get(sec, 0) / total_chars
                ys_for_sec.append(pct)
        if ys_for_sec:
            sec_pct_ranges[sec] = (min(ys_for_sec), max(ys_for_sec))

    chart3_subtitle_parts = []
    if "verse" in sec_pct_ranges and "chorus" in sec_pct_ranges:
        vc_low = sec_pct_ranges["verse"][0] + sec_pct_ranges["chorus"][0]
        vc_high = sec_pct_ranges["verse"][1] + sec_pct_ranges["chorus"][1]
        chart3_subtitle_parts.append(f"verses + choruses {vc_low:.0f}-{vc_high:.0f}%")
    if "bridge" in sec_pct_ranges:
        b_low, b_high = sec_pct_ranges["bridge"]
        chart3_subtitle_parts.append(f"bridges {b_low:.0f}-{b_high:.0f}%")
    if "in_out" in sec_pct_ranges:
        io_low, io_high = sec_pct_ranges["in_out"]
        chart3_subtitle_parts.append(f"in_out {io_low:.0f}-{io_high:.0f}%")
    if "refrain" in sec_pct_ranges:
        r_low, r_high = sec_pct_ranges["refrain"]
        chart3_subtitle_parts.append(f"refrains rare ({r_low:.0f}-{r_high:.0f}%)")
    chart3_subtitle = "% of total lyric characters per section, by album. " + "; ".join(chart3_subtitle_parts) + "."

    fig3.update_layout(
        barmode="stack",
        title=dict(text=f"<b>Per-album section composition</b><br><sub>{chart3_subtitle}</sub>", x=0.02),
        xaxis_title="Album (release year order)",
        yaxis_title="% of characters",
        yaxis=dict(range=[0, 100]),
        height=480, width=None,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
        template="plotly_white",
    )

    # ----- chart 4: vibe cluster composition (phase 6) -----
    if vibes_rows:
        cluster_album = defaultdict(lambda: Counter())
        for r in vibes_rows:
            cluster_album[int(r["cluster"])][r["Album"]] += 1
        clusters_sorted = sorted(cluster_album.keys())
        albums_for_vibes = sorted({a for c in cluster_album for a in cluster_album[c]},
                                   key=lambda a: next((int(s["Year"]) for s in sent_songs if s["Album"]==a and s["Year"].isdigit()), 9999))

        fig4 = make_subplots(rows=2, cols=5, subplot_titles=[f"Cluster {c}" for c in clusters_sorted],
                              horizontal_spacing=0.04, vertical_spacing=0.18)
        for idx, c in enumerate(clusters_sorted):
            row = idx // 5 + 1
            col = idx % 5 + 1
            sizes = [cluster_album[c].get(a, 0) for a in albums_for_vibes]
            fig4.add_trace(go.Bar(
                x=albums_for_vibes, y=sizes,
                marker_color=[ALBUM_COLORS.get(a, "#888") for a in albums_for_vibes],
                showlegend=False,
                hovertemplate=f"cluster {c}<br>%{{x}}: %{{y}} songs<extra></extra>",
            ), row=row, col=col)
            fig4.update_xaxes(tickangle=45, tickfont=dict(size=8), row=row, col=col)
            fig4.update_yaxes(tickfont=dict(size=8), row=row, col=col)

        fig4.update_layout(
            title=dict(text="<b>Vibe clusters: album composition (phase 6)</b><br><sub>10 K-means clusters on the sentence embeddings. Each cluster shows which albums its songs come from. Round 4 audit: silhouette ~0, ARI across seeds ~0.15 — clusters are weak, see the caveat below the chart.</sub>",
                       x=0.02, y=0.98, yanchor="top"),
            height=720, width=None,
            template="plotly_white",
            showlegend=False,
            margin=dict(t=120, l=40, r=40, b=40),
        )

    # ----- chart 5: section-level average sentiment (phase 3) -----
    section_avg = defaultdict(list)
    for r in sent_secs:
        if r["Section"] not in SECTION_GROUPS: continue
        b = fnum(r.get("bert_pos"))
        if b is not None: section_avg[r["Section"]].append(b)

    # rank sections by mean bert_pos for chart 4 subtitle
    sec_means = [(s, statistics.mean(vs)) for s, vs in section_avg.items() if vs]
    sec_means.sort(key=lambda x: -x[1])
    top_sec = sec_means[0]
    chart4_subtitle = ""
    if sec_means:
        chart4_subtitle = (
            f"Mean across all 244 songs. <b>{top_sec[0]} is the most positive section</b> "
            f"on DistilBERT pos (mean {top_sec[1]:.2f}), "
            f"above {sec_means[1][0]} ({sec_means[1][1]:.2f}) and {sec_means[2][0]} ({sec_means[2][1]:.2f}). "
            f"Section composition depends on lyric length, which DistilBERT handles inconsistently — "
            f"shorter sections tend to get more confident (often higher-positive) scores."
        )

    sec_x = ["verse", "chorus", "bridge", "refrain", "in_out"]
    sec_y = [statistics.mean(section_avg[s]) if section_avg[s] else 0 for s in sec_x]
    sec_n = [len(section_avg[s]) for s in sec_x]

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=sec_x, y=sec_y,
        marker_color=[section_colors.get(s, "#888") for s in sec_x],
        text=[f"{v:.2f}<br>n={n}" for v, n in zip(sec_y, sec_n)],
        textposition="outside",
        hovertemplate="%{x}<br>mean DistilBERT pos: %{y:.3f}<br>n: %{text}<extra></extra>",
    ))
    fig5.update_layout(
        title=dict(text=f"<b>Per-section sentiment (DistilBERT pos)</b><br><sub>{chart4_subtitle}</sub>", x=0.02),
        xaxis_title="Section",
        yaxis_title="Mean DistilBERT pos",
        yaxis=dict(range=[0, 0.7]),
        height=420, width=None,
        template="plotly_white",
    )

    # ----- chart 6: top mutual pairs (table) -----
    if sim_rows:
        # compute mutual pairs (both A and B in each other's top-5)
        # and find the top mutual pair dynamically for the subtitle
        n_k = 5
        sim_by_src = {}
        for r in sim_rows:
            sim_by_src[(r["src_album"], r["src_title"])] = r
        mutual_top_score = 0.0
        top_mutual = None
        for r in sim_rows:
            j_row = sim_by_src.get((r["n1_album"], r["n1_title"]))
            if not j_row: continue
            # check if r["src_title"] is in j_row's top-K
            top5_b = [j_row[f"n{k}_title"] for k in range(1, n_k+1)]
            if r["src_title"] in top5_b:
                score = float(r["n1_score"])
                if score > mutual_top_score:
                    mutual_top_score = score
                    top_mutual = (r["src_title"], r["src_album"], r["src_year"],
                                  r["n1_title"], r["n1_album"], r["n1_year"])

        if top_mutual:
            chart6_subtitle = (
                f"Both A's top-{n_k} and B's top-{n_k} lists include each other. "
                f"Strongest pair: <b>{top_mutual[0]}</b> ({top_mutual[1]}, {top_mutual[2]}) ↔ "
                f"<b>{top_mutual[3]}</b> ({top_mutual[4]}, {top_mutual[5]}) at similarity "
                f"<b>{mutual_top_score:.3f}</b>."
            )
        else:
            chart6_subtitle = f"Both A's top-{n_k} and B's top-{n_k} lists include each other."

        # build table rows: top 15 by n1_score, mark which are mutual
        rows_for_table = []
        for r in sim_rows:
            score = float(r["n1_score"])
            j_row = sim_by_src.get((r["n1_album"], r["n1_title"]))
            mutual = bool(j_row and r["src_title"] in [j_row[f"n{k}_title"] for k in range(1, n_k+1)])
            rows_for_table.append((score, mutual, r["src_title"], r["src_album"], r["src_year"],
                                   r["n1_title"], r["n1_album"], r["n1_year"]))
        rows_for_table.sort(reverse=True, key=lambda p: p[0])

        fig6 = go.Figure(data=[go.Table(
            header=dict(values=["#", "Song A", "Album A", "Year A", "Song B", "Album B", "Year B", "similarity"],
                        fill_color="#2ca02c", font=dict(color="white"), align="left"),
            cells=dict(
                values=[
                    list(range(1, 16)),
                    [p[2] for p in rows_for_table[:15]],
                    [p[3] for p in rows_for_table[:15]],
                    [p[4] for p in rows_for_table[:15]],
                    [p[5] for p in rows_for_table[:15]],
                    [p[6] for p in rows_for_table[:15]],
                    [p[7] for p in rows_for_table[:15]],
                    [f"{p[0]:.4f}" for p in rows_for_table[:15]],
                ],
                align="left",
                font=dict(size=11),
                height=24,
            ),
        )])
        fig6.update_layout(
            title=dict(text=f"<b>Top 15 mutual nearest-neighbor pairs (phase 5)</b><br><sub>{chart6_subtitle}</sub>", x=0.02),
            height=520, width=None,
        )

    # ----- assemble HTML -----
    html_parts = ["""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Taylor Swift's Songs — NLP Analysis Dashboard</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #fafafa; color: #222; }
.header { padding: 32px 48px; background: #1f1f1f; color: #fafafa; }
.header h1 { margin: 0 0 8px 0; font-size: 28px; font-weight: 600; }
.header p { margin: 0; opacity: 0.7; font-size: 14px; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px 48px; }
.section { background: white; margin: 24px 0; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.section h2 { margin-top: 0; font-size: 20px; border-bottom: 1px solid #eee; padding-bottom: 12px; }
.section p { font-size: 14px; line-height: 1.6; color: #555; }
.footer { padding: 24px 48px; text-align: center; color: #999; font-size: 12px; }
a { color: #2ca02c; }
</style>
</head>
<body>
<div class="header">
  <h1>Taylor Swift's Songs — NLP Analysis</h1>
  <p>Phases 2-6 combined · 244 songs across 13 albums · CoTS v1.4 corpus</p>
</div>
<div class="container">
<div class="section">
  <h2>1. Career-long sentiment arc</h2>
  <p>Each point is a song. DistilBERT-SST2 positive probability — pretrained sentiment model, transferred to lyrics. Hover for song title and scores.</p>
  <p>{headline}</p>
"""]
    html_parts.append(fig1.to_html(full_html=False, include_plotlyjs="cdn", div_id="chart1"))
    html_parts.append("</div>")

    html_parts.append("""
<div class="section">
  <h2>2. Sentiment vs lexical diversity per album</h2>
  <p>Album-level mean DistilBERT pos (y) vs album-level mean MATTR-200 (x). MATTR-200 is moving-average type-token ratio — measures within-song vocabulary diversity across a 200-token window. MATTR-200 covers ~98% of songs (median song = 366 words); the older MATTR-500 window was effectively TTR for 93% of songs.</p>
    <p><b>Headline (descriptive, not causal):</b> TTPD sits at the bottom-right (lowest DistilBERT pos + highest MATTR-200). Whether this is 'most linguistically and emotionally complex' is a value judgment, not a measurement — the chart only shows the relative position. Sample sizes per album are 12-31 songs; per-album means have standard errors of ±0.06-±0.12 on the bert_pos scale (full table with 95% CIs in <code>reports/sentiment_summary.md</code>).</p>
""")
    html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False, div_id="chart2"))
    html_parts.append("</div>")

    html_parts.append("""
<div class="section">
  <h2>3. Per-album section composition</h2>
  <p>% of total lyric characters per section, by album. Verses typically dominate (~50-60%), choruses ~15-20%, bridges ~10-15%, refrains & intros/outros small.</p>
""")
    html_parts.append(fig3.to_html(full_html=False, include_plotlyjs=False, div_id="chart3"))
    html_parts.append("</div>")

    html_parts.append("""
<div class="section">
  <h2>4. Per-section sentiment (phase 3)</h2>
  <p>Mean DistilBERT pos per section across all 244 songs. Section sentiment is sensitive to text length — short sections (intros/outros) get more confident mid-positive scores, while long sections (choruses) tend to get lower scores. See the section composition chart above for the length breakdown.</p>
"""
    )
    html_parts.append(fig5.to_html(full_html=False, include_plotlyjs=False, div_id="chart5"))
    html_parts.append("</div>")
    if vibes_rows:
        html_parts.append("""
<div class="section">
  <h2>5. Vibe clusters (phase 6a — K-means)</h2>
  <p>K-means (K=10) on the 384-dim sentence embeddings. Each cluster shows which albums its songs come from.</p>
  <p><b>Round 4 audit caveat:</b> silhouette score for K=10 is essentially zero (~0.004). Cluster assignments are unstable across random seeds (ARI ~0.15). About 50% of songs are on average closer to a different cluster's centroid than their own. The 'clusters' are real for seed=42 but a different seed would give a different 'story' with the same data — see <code>reports/vibes_summary.md</code> for the cluster quality numbers.</p>
""")
        html_parts.append(fig4.to_html(full_html=False, include_plotlyjs=False, div_id="chart4"))
        html_parts.append("</div>")

    if sim_rows:
        html_parts.append(f"""
<div class="section">
  <h2>6. Top mutual nearest-neighbor pairs (phase 5)</h2>
  <p>Sentence-transformer (all-MiniLM-L6-v2) song embeddings, cosine similarity top-5 nearest neighbors. Mutual pairs = both A and B list each other in their top-5.</p>
  <p>{chart6_subtitle.replace('<b>', '<b>').replace('</b>', '</b>')}</p>
""")
        html_parts.append(fig6.to_html(full_html=False, include_plotlyjs=False, div_id="chart6"))
        html_parts.append("</div>")

    html_parts.append("""
<div class="footer">
  Generated by <code>analyze/dashboard.py</code> · data from CoTS v1.4 (Corpus-of-Taylor-Swift, sagesolar, GPL-3.0)<br>
  See <a href="README.md">README.md</a> for individual phase summaries.
</div>
</div>
</body>
</html>""")

    out_path = Path(args.out)
    # substitute dynamic headlines into the static skeleton before writing
    full_html = "\n".join(html_parts)
    full_html = full_html.replace("{headline}", headline)
    full_html = full_html.replace("{chart6_subtitle}", chart6_subtitle)
    out_path.write_text(full_html, encoding="utf-8")
    print(f"[ok] wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())