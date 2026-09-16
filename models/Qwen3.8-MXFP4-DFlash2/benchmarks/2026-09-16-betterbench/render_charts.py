#!/usr/bin/env python3
"""Render the published, matched BetterBench results. Requires matplotlib==3.10.7.

Run from any directory. No model, GPU, compiler or serving framework is required.
The chart values are calculated from data/*.json. Reported cache token capacity
comes from startup log evidence recorded in provenance.json, not BetterBench.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


PAITON = "#087F8C"
GGZ = "#D98520"
INK = "#162D3B"
MUTED = "#51616E"
GRID = "#DCE4E9"
BACKGROUND = "#FFFFFF"
RUNS = ("paiton-release-repeat", "ggz-current-hipmatched-quick")
LABELS = ("Paiton release", "GGZ14 current · HIP settings matched")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.axisbelow": True,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "svg.hashsalt": "paiton-betterbench-2026-09-16",
    "svg.fonttype": "path",
    "savefig.facecolor": BACKGROUND,
})


def read(path):
    return json.loads(path.read_text())


def weighted_decode(result):
    weights = result["config"]["weights"]
    return sum(weights[category] * median(row["decode_tps"] for row in rows)
               for category, rows in result["single_stream"].items()) / sum(weights.values())


def heading(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=30)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10)


def grouped_bars(ax, groups, values, ylabel, formatter, ylim=None):
    width = 0.34
    positions = list(range(len(groups)))
    for series, color, offset in zip(values, (PAITON, GGZ), (-width / 2, width / 2)):
        bars = ax.bar([x + offset for x in positions], series, width=width,
                      color=color, edgecolor=BACKGROUND, linewidth=0.6)
        ax.bar_label(bars, labels=[formatter(v) for v in series], padding=6,
                     fontsize=10, fontweight="bold", color=INK)
    upper = ylim or max(max(v) for v in values) * 1.2
    ax.set_ylim(0, upper)
    ax.set_xticks(positions, groups)
    ax.tick_params(length=0, pad=7)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def two_horizontal(ax, values, xlabel, formatter):
    bars = ax.barh([1, 0], values, color=(PAITON, GGZ), height=0.45)
    ax.bar_label(bars, labels=[formatter(v) for v in values], padding=8,
                 fontsize=13, fontweight="bold", color=INK)
    ax.set_yticks([1, 0], ["Paiton", "GGZ14"])
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_ylim(-0.55, 1.55)
    ax.tick_params(length=0, pad=7)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def header(fig, title, subtitle, legend_y):
    fig.text(0.07, 0.956, title, fontsize=23, fontweight="bold", va="top")
    fig.text(0.07, 0.915, subtitle, fontsize=11, color=MUTED, va="top")
    fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip((PAITON, GGZ), LABELS)],
               loc="upper left", bbox_to_anchor=(0.065, legend_y), frameon=False,
               ncol=2, handlelength=1.6, columnspacing=2.4)


def footer(fig, y):
    fig.text(0.07, y,
             "1 × Radeon AI PRO R9700 (32 GB) · 5 GiB cache each · identical MXFP4 target + DFlash2 draft",
             fontsize=9.5, color=MUTED)
    fig.text(0.07, y - 0.024,
             "BetterBench 0.6.0 · 128-token generation cap · 52 measured requests per run · 16 September 2026",
             fontsize=9.5, color=MUTED)
    fig.text(0.07, y - 0.048,
             "Short sampled screen; no statistical significance or quality claim. Stronger HIP-matched GGZ14 run shown.",
             fontsize=9.5, color=MUTED)


def overview(results, provenance):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.78, bottom=0.18, hspace=0.7, wspace=0.32)
    header(fig, "Qwen3.8 27B · one Radeon AI PRO R9700",
           "Matched BetterBench throughput and reported cache capacity", 0.882)
    p, g = results
    a = axes[0, 0]
    vals = [weighted_decode(r) for r in results]
    heading(a, "Weighted serial generation", f"Category-weighted medians · Paiton +{(vals[0] / vals[1] - 1) * 100:.1f}%")
    two_horizontal(a, vals, "Output tokens / second · higher is better", lambda x: f"{x:.1f}")

    a = axes[0, 1]
    vals = [[r["aggregate_tps"] for r in run["concurrency"]] for run in results]
    heading(a, "Aggregate generation", f"Eight requests at each level · C8: Paiton +{(vals[0][-1] / vals[1][-1] - 1) * 100:.1f}%")
    grouped_bars(a, [f"C{r['level']}" for r in p["concurrency"]], vals,
                 "Output tokens / second", lambda x: f"{x:.1f}")
    a.set_xlabel("Concurrent requests", labelpad=7)

    a = axes[1, 0]
    vals = [[median(row["pp_tps"]) for row in run["prefill"]] for run in results]
    depths = [f"{median(row['prompt_tokens']):,.0f} tokens" for row in p["prefill"]]
    heading(a, "Prefill", "Median input tokens / TTFT · actual prompt lengths")
    grouped_bars(a, depths, vals, "Input tokens / second", lambda x: f"{x:,.0f}")
    a.set_xlabel("Median input length · two requests per length", labelpad=7)

    a = axes[1, 1]
    capacities = provenance["reported_cache_capacity_tokens"]
    vals = [capacities["paiton"], capacities["ggz"]]
    heading(a, "Reported capacity · fixed 5 GiB cache", f"Startup log values · {vals[0] / vals[1]:.2f}× token capacity")
    two_horizontal(a, vals, "Reported cache tokens · higher is better", lambda x: f"{x:,.0f}")
    a.text(0, -0.33, "Capacity is runtime-reported, not a completed long-context test.",
           transform=a.transAxes, fontsize=9.5, color=MUTED)
    footer(fig, 0.087)
    return fig


def tradeoffs(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.8))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.71, bottom=0.29, wspace=0.3)
    header(fig, "Latency and per-request speed",
           "The same one-GPU runs, including metrics that favor GGZ14", 0.867)
    groups = [f"C{row['level']}" for row in results[0]["concurrency"]]
    vals = [[median(row["ttft_ms"]) for row in run["concurrency"]] for run in results]
    heading(axes[0], "Median time to first token", "Includes server queueing + HTTP · lower is better")
    grouped_bars(axes[0], groups, vals, "Milliseconds", lambda x: f"{x:,.0f}")
    vals = [[median(row["decode_tps"]) for row in run["concurrency"]] for run in results]
    heading(axes[1], "Median per-request generation", "Decode after first token · higher is better")
    grouped_bars(axes[1], groups, vals, "Output tokens / second", lambda x: f"{x:.1f}")
    for ax in axes:
        ax.set_xlabel("Concurrent requests · eight requests per level", labelpad=10)
    fig.text(0.08, 0.202, "GGZ14 has lower TTFT at C1 and C2, and higher per-request decode at C4 and C8.",
             fontsize=11, color=INK, fontweight="bold")
    fig.text(0.08, 0.169, "At C8, Paiton completes the batch faster despite GGZ14's higher per-request decode rate.",
             fontsize=11, color=MUTED)
    footer(fig, 0.096)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--provenance", type=Path, default=Path(__file__).resolve().parent / "provenance.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    args = parser.parse_args()
    results = [read(args.data_dir / f"{name}.json") for name in RUNS]
    assert results[0]["config"] == results[1]["config"], "Benchmark settings differ"
    for run in results:
        assert run["betterbench_version"] == "0.6.0"
        assert [row["level"] for row in run["concurrency"]] == [1, 2, 4, 8]
        assert all(row["ok"] == row["requests"] == 8 for row in run["concurrency"])
    assert [row["prompt_tokens"] for row in results[0]["prefill"]] == [row["prompt_tokens"] for row in results[1]["prefill"]]
    figures = {"throughput-and-cache": overview(results, read(args.provenance)),
               "latency-and-per-request": tradeoffs(results)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(args.output_dir / f"{name}.svg", metadata={"Date": None, "Creator": "Paiton benchmark chart generator"})
        figure.savefig(args.output_dir / f"{name}.png", dpi=180, metadata={"Software": "Paiton benchmark chart generator"})
        svg_path = args.output_dir / f"{name}.svg"
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        plt.close(figure)
        print(f"Rendered {name}.svg and {name}.png")


if __name__ == "__main__":
    main()
