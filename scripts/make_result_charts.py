"""Render the v1 evaluation charts for the README, in light and dark versions.

    uv run --with matplotlib python scripts/make_result_charts.py

Writes docs/img/results-*-{light,dark}.png. The numbers are the measured v1
results in docs/eval_results.md: update both together. Colors come from the
validated reference palette (categorical slots 1-3, ink, grid and surface per
mode). Values are direct-labeled on every mark, so no color carries meaning
alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1baf7a",
        "before": "#c3c2b7",
        "good": "#006300",
        "bad": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
        "before": "#52514e",
        "good": "#0ca30c",
        "bad": "#e66767",
    },
}

FONT = ["Segoe UI", "system-ui", "DejaVu Sans", "sans-serif"]

# ---- measured v1 results (docs/eval_results.md) ---------------------------------------

# (label, measured %, target %): every row is "higher is better"
GATES = [
    ("Shown quotes verified by code (G1)", 100.0, 100.0),
    ("Unanswerable questions refused (G5)", 100.0, 80.0),
    ("Adversarial questions handled (G9)", 90.0, 100.0),
    ("Right judgment in top 50 (G3)", 84.1, 90.0),
    ("Answerable questions answered (G5)", 81.4, 90.0),
    ("Right judgment in top 5, reranked (G3)", 72.7, 75.0),
    ("Verified sentences judged supported (G2)", 72.6, 90.0),
]

RAGAS = [
    ("Faithfulness", 0.887, "35 answered"),
    ("Context precision", 0.800, "44 questions"),
    ("Context recall", 0.760, "42 questions"),
    ("Answer relevancy", 0.641, "35 answered"),
]

# (label, before, after): judgment-level recall, 44 dev questions
RETRIEVAL = [
    ("Keyword search, top 50", 0.20, 0.80),
    ("Hybrid search, top 50", 0.77, 0.84),
    ("Hybrid + rerank, top 5", 0.70, 0.73),
]

# median seconds per stage (latency run, 29 requests)
STAGES = [("Retrieve", 0.5), ("Rerank", 3.6), ("Answer LLM", 2.9)]
P50, P95, TARGET = 6.7, 9.4, 15.0


def base(t: dict, w: float, h: float):
    plt.rcParams.update({"font.family": FONT, "font.size": 10})
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def title(fig, t: dict, main: str, sub: str) -> None:
    h = fig.get_figheight()  # inches: keep a fixed gap however tall the figure is
    fig.text(
        0.01, 1 - 0.12 / h, main, ha="left", va="top", fontsize=13, weight="bold", color=t["ink"]
    )
    fig.text(0.01, 1 - 0.42 / h, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"results-{name}-{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def gates_chart(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = base(t, 9, 4.6)
    rows = list(reversed(GATES))
    y = range(len(rows))
    ax.barh(list(y), [m for _, m, _ in rows], height=0.55, color=t["s1"], zorder=2)
    for i, (label, m, target) in enumerate(rows):
        ax.plot([target, target], [i - 0.38, i + 0.38], color=t["ink"], linewidth=2, zorder=3)
        ok = m >= target
        mark = "✓ pass" if ok else "✗ fail"
        ax.text(101.5, i, f"{m:.1f}%", va="center", ha="left", color=t["ink"], fontsize=10)
        ax.text(
            113,
            i,
            mark,
            va="center",
            ha="left",
            fontsize=10,
            color=t["good"] if ok else t["bad"],
            weight="bold",
        )
    ax.set_yticks(list(y), [r[0] for r in rows], color=t["ink2"])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.text(
        100, len(rows) - 0.35, "│ = target", ha="right", va="bottom", color=t["muted"], fontsize=9
    )
    title(
        fig,
        t,
        "Nyaya Search v1: goals measured against targets",
        "104 questions (44 dev · 40 unanswerable · 20 adversarial) · 0 hallucinated sentences "
        "· p95 latency 9.4 s (target 15 s)",
    )
    fig.subplots_adjust(left=0.33, right=0.86, top=0.82, bottom=0.08)
    save(fig, "gates", mode)


def ragas_chart(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = base(t, 9, 3.2)
    rows = list(reversed(RAGAS))
    y = list(range(len(rows)))
    ax.barh(y, [v for _, v, _ in rows], height=0.55, color=t["s1"], zorder=2)
    for i, (_, v, over) in enumerate(rows):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", ha="left", color=t["ink"], fontsize=10)
        ax.text(1.22, i, over, va="center", ha="right", color=t["muted"], fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], color=t["ink2"])
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    title(
        fig,
        t,
        "RAGAS scores (secondary to the exact quote check)",
        "44 dev questions · judge qwen3.8-27b · refused questions excluded from "
        "faithfulness and relevancy",
    )
    fig.subplots_adjust(left=0.2, right=0.97, top=0.74, bottom=0.11)
    save(fig, "ragas", mode)


def retrieval_chart(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = base(t, 9, 3.4)
    rows = list(reversed(RETRIEVAL))
    h = 0.34
    for i, (_, before, after) in enumerate(rows):
        ax.barh(i + h / 2 + 0.02, before, height=h, color=t["before"], zorder=2)
        ax.barh(i - h / 2 - 0.02, after, height=h, color=t["s1"], zorder=2)
        ax.text(
            before + 0.012,
            i + h / 2 + 0.02,
            f"{before:.0%} before",
            va="center",
            color=t["ink2"],
            fontsize=9,
        )
        ax.text(
            after + 0.012,
            i - h / 2 - 0.02,
            f"{after:.0%} after",
            va="center",
            color=t["ink"],
            fontsize=9.5,
            weight="bold",
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=t["ink2"])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    title(
        fig,
        t,
        "Retrieval: share of questions where the right judgment is found",
        "Before vs after replacing Postgres ts_rank_cd keyword search with BM25 · 44 dev questions",
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.76, bottom=0.11)
    save(fig, "retrieval", mode)


def latency_chart(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = base(t, 9, 3.0)
    left = 0.0
    colors = [t["s1"], t["s2"], t["s3"]]
    for (name, sec), c in zip(STAGES, colors, strict=True):
        ax.barh(
            0, sec, left=left, height=0.5, color=c, edgecolor=t["surface"], linewidth=2, zorder=2
        )
        if sec >= 1.5:  # wide enough to hold its label inside
            ax.text(
                left + sec / 2,
                0,
                f"{name}\n{sec:.1f} s",
                ha="center",
                va="center",
                color="#ffffff",
                fontsize=9,
                weight="bold",
            )
        else:  # label below the bar, in ink
            ax.text(
                left + sec / 2,
                -0.3,
                f"{name} {sec:.1f} s",
                ha="left",
                va="top",
                color=t["ink"],
                fontsize=9,
            )
        left += sec
    for x, label, style in ((P95, f"p95 {P95} s", "--"), (TARGET, f"target {TARGET:.0f} s", "-")):
        ax.axvline(x, color=t["ink"], linestyle=style, linewidth=1.4, zorder=3)
        ax.text(x + 0.15, 0.36, label, color=t["ink"], fontsize=9.5, va="bottom")
    ax.set_yticks([])
    ax.set_xlim(0, 16.5)
    ax.set_xticks([0, 3, 6, 9, 12, 15], ["0 s", "3 s", "6 s", "9 s", "12 s", "15 s"])
    ax.set_ylim(-0.6, 0.75)
    title(
        fig,
        t,
        "Latency: where an answer's time goes (G7 pass)",
        f"Median per stage · end-to-end median {P50} s, p95 {P95} s · one request at a time, "
        "warm, RTX 2060",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.12)
    save(fig, "latency", mode)


def main() -> None:
    for mode in THEMES:
        gates_chart(mode)
        ragas_chart(mode)
        retrieval_chart(mode)
        latency_chart(mode)
    print(f"wrote {len(list(OUT.glob('results-*.png')))} charts to {OUT}")


if __name__ == "__main__":
    main()
