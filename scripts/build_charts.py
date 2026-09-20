#!/usr/bin/env python3
"""Build the benchmark charts committed to the repository."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/system1-transcript-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"
SUMMARY = json.loads((ROOT / "data" / "summary_metrics.json").read_text())
COSTS = json.loads((ROOT / "data" / "cost_assumptions.json").read_text())
NORMALIZED_COST = json.loads((ROOT / "data" / "normalized_cost_scenario.json").read_text())

COLORS = {
    "Open encoder": "#0B678B",
    "JEV": "#8059C3",
    "LLM": "#D47900",
}
INK = "#172033"
MUTED = "#66758A"
GRID = "#D8E0E8"


def setup() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 19,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def metric_chart(metric: str, title: str, filename: str) -> None:
    models = sorted(SUMMARY["models"], key=lambda item: item["accuracy"])
    labels = [item["label"] for item in models]
    values = [100 * item[metric] for item in models]
    colors = [COLORS[item["family"]] for item in models]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    bars = ax.bar(range(len(models)), values, color=colors, width=0.72)
    ax.set_title(title, loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "150 locked test transcripts · 27 binary attributes · models ordered by accuracy",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    ax.set_ylabel(f"{metric.upper()} (%)" if metric == "f1" else "Accuracy (%)")
    ax.set_ylim(80, 101.8)
    ax.set_xticks(range(len(models)), labels)
    ax.tick_params(axis="x", labelrotation=0, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS.values()]
    ax.legend(handles, COLORS.keys(), frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.01,
        0.01,
        f"{metric.upper() if metric == 'f1' else 'Accuracy'} axis begins at 80% to make differences legible.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(CHARTS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metrics_table() -> None:
    models = sorted(SUMMARY["models"], key=lambda item: item["accuracy"])
    columns = ["Approach", "Family", "Accuracy", "Precision", "Recall", "F1", "Exact match"]
    rows = [
        [
            item["label"].replace("\n", " "),
            item["family"],
            f"{100 * item['accuracy']:.2f}%",
            f"{100 * item['precision']:.2f}%",
            f"{100 * item['recall']:.2f}%",
            f"{100 * item['f1']:.2f}%",
            f"{100 * item['exact_match']:.2f}%",
        ]
        for item in models
    ]

    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.axis("off")
    ax.set_title("All transcript-classification quality metrics", loc="left", pad=24)
    ax.text(
        0,
        0.965,
        "Held-out test set · micro metrics across 4,050 decisions; exact match requires all 27 labels correct",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="right",
        colLoc="right",
        bbox=[0, 0.02, 1, 0.88],
        colWidths=[0.31, 0.13, 0.112, 0.112, 0.10, 0.10, 0.136],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#E9EFF5")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#F7F9FB" if row % 2 == 0 else "white")
            if col in (0, 1):
                cell.set_text_props(ha="left")
            if rows[row - 1][1] == "JEV":
                cell.get_text().set_color("#6841A8")
            elif rows[row - 1][1] == "LLM":
                cell.get_text().set_color("#A85E00")
    fig.tight_layout()
    fig.savefig(CHARTS / "all-metrics-table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def cost_chart() -> None:
    by_id = {item["id"]: item for item in SUMMARY["models"]}
    cost = COSTS["estimated_cost_usd_1000_transcripts"]
    ordered_ids = sorted(cost, key=lambda key: (cost[key], by_id[key]["accuracy"]))
    values = [cost[key] for key in ordered_ids]
    labels = [by_id[key]["label"] for key in ordered_ids]
    colors = [COLORS[by_id[key]["family"]] for key in ordered_ids]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    bars = ax.bar(range(len(ordered_ids)), values, color=colors, width=0.72)
    ax.set_title("Estimated marginal inference cost for 1,000 transcripts", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "USD API charges · logarithmic scale above $0.03 · validation/training and local hardware excluded",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    ax.set_ylabel("Estimated USD per 1,000 transcripts")
    ax.set_yscale("symlog", linthresh=0.03, linscale=0.8, base=10)
    ticks = [0, 0.03, 0.1, 0.3, 1, 3, 10]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "$0" if value == 0 else f"${value:g}"))
    ax.set_ylim(0, 12)
    ax.set_xticks(range(len(ordered_ids)), labels)
    ax.tick_params(axis="x", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value, model_id in zip(bars, values, ordered_ids):
        label = "$0 API*" if value == 0 else (f"${value:.3f}" if value < 1 else f"${value:.2f}")
        y = 0.018 if value == 0 else value * 1.18
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS.values()]
    ax.legend(handles, COLORS.keys(), frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.01,
        0.01,
        "* Local encoders have no API fee; electricity, hardware, hosting, and operations are not zero and are not estimated. "
        "Sol/Luna use a standardized compact prompt estimate and exclude hidden reasoning tokens.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(CHARTS / "estimated-cost-1000-transcripts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalized_cost_quality_chart() -> None:
    by_id = {item["id"]: item for item in SUMMARY["models"]}
    costs = NORMALIZED_COST["normalized_results"]
    variants = NORMALIZED_COST["quality_variants"]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    offsets = {
        "modernbert_zero_shot": (8, -18),
        "modernbert_zero_shot_opt_threshold": (8, 8),
        "modernbert_trained": (8, -4),
        "jev_choice": (8, -18),
        "jev_noul_default": (8, -17),
        "jev_noul_calibrated": (8, 8),
    }
    for variant in variants:
        item = by_id[variant["id"]]
        cost = costs[variant["cost_path"]]["cost_usd_per_million_raw_state_tokens"]
        accuracy = 100 * item["accuracy"]
        ax.scatter(
            cost,
            accuracy,
            s=180,
            color=COLORS[item["family"]],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        dx, dy = offsets[variant["id"]]
        ax.annotate(
            f"{item['label'].replace(chr(10), ' ')}\n{accuracy:.2f}% · ${cost:.4f}/M raw",
            (cost, accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.2,
            ha="left",
            va="center",
        )

    ax.set_title("Accuracy versus normalized serving cost", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "170,331 raw state tokens/s · 27 questions · H100 at $5/hour and near-100% utilization · lower cost is better",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.005, 1.0)
    ax.set_ylim(92, 100.7)
    ax.set_xlabel("Estimated USD per million raw transcript tokens (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.03, 0.1, 0.3, 1]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=color)
        for color in (COLORS["Open encoder"], COLORS["JEV"])
    ]
    ax.legend(handles, ["ModernBERT", "TypeSafe.ai JEV"], frameon=False, ncol=2, loc="lower left")
    fig.text(
        0.01,
        0.01,
        "ModernBERT: estimated 170.3k processed tokens/s per H100 at USD 5/hour (proxy, not an exact-checkpoint benchmark). "
        "JEV: USD 0.042/M billed input tokens. JEV capacity and rate limits were not validated.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(CHARTS / "accuracy-vs-normalized-cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalized_cost_state_length_chart() -> None:
    sensitivity = NORMALIZED_COST["state_length_sensitivity"]
    assumptions = sensitivity["assumptions"]
    lower, upper = sensitivity["state_token_range"]
    state_tokens = np.geomspace(lower, upper, 500)
    encoder_rate = assumptions["modernbert_cost_usd_per_million_processed_tokens"]
    questions = assumptions["questions_per_transcript"]
    zero_overhead = assumptions["modernbert_zero_shot_non_state_tokens_per_transcript"]
    trained_overhead = assumptions["modernbert_trained_special_tokens_per_transcript"]
    jev_rate = assumptions["jev_price_usd_per_million_billed_input_tokens"]
    noul_overhead = assumptions["jev_noul_non_state_billed_tokens_per_transcript"]
    choice_overhead = assumptions["jev_choice_non_state_billed_tokens_per_transcript"]

    curves = [
        (
            "ModernBERT trained heads",
            encoder_rate * (state_tokens + trained_overhead) / state_tokens,
            "#0B678B",
            "-",
        ),
        (
            "ModernBERT zero-shot",
            encoder_rate * (questions * state_tokens + zero_overhead) / state_tokens,
            "#2A9D8F",
            "-",
        ),
        (
            "JEV Noul",
            jev_rate * (state_tokens + noul_overhead) / state_tokens,
            "#8059C3",
            "-",
        ),
        (
            "JEV Choice",
            jev_rate * (state_tokens + choice_overhead) / state_tokens,
            "#A78BDB",
            "--",
        ),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    for label, values, color, linestyle in curves:
        ax.plot(state_tokens, values, label=label, color=color, linewidth=3, linestyle=linestyle)

    crossovers = sensitivity["crossovers_with_modernbert_zero_shot"]
    noul_cross = crossovers["jev_noul_state_tokens"]
    choice_cross = crossovers["jev_choice_state_tokens"]
    zero_cost = lambda s: encoder_rate * (questions * s + zero_overhead) / s
    ax.axvspan(noul_cross, upper, color="#8059C3", alpha=0.045, zorder=0)
    ax.axvline(224.46, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(224.46, 1.72, "benchmark mean\n224 tokens", color=MUTED, fontsize=9, ha="center")
    for crossover, label, xytext in [
        (noul_cross, "Noul crossover\n586 tokens", (430, 0.43)),
        (choice_cross, "Choice crossover\n682 tokens", (850, 0.34)),
    ]:
        ax.scatter(crossover, zero_cost(crossover), s=70, color="#8059C3", edgecolor="white", zorder=5)
        ax.annotate(
            label,
            xy=(crossover, zero_cost(crossover)),
            xytext=xytext,
            textcoords="data",
            fontsize=9.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
        )

    ax.set_title("Normalized serving cost versus state length", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "27 questions · nominal question/criteria overhead held fixed · H100 at USD 5/hour · lower cost is better",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(0.006, 3.2)
    ax.set_xlabel("Raw input tokens in each state (log scale)")
    ax.set_ylabel("Estimated USD per million raw state tokens (log scale)")
    ax.xaxis.set_major_locator(FixedLocator([50, 100, 250, 500, 1000, 2000, 4000, 8000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0.008, 0.01, 0.03, 0.1, 0.3, 1, 3]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.grid(color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Normalization is per 1M raw state tokens. Assumes constant 170.3k processed tokens/s per H100 and 1:1 JEV state-token scaling. "
        "Long-context throughput and tokenizer differences are not modeled.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(CHARTS / "normalized-cost-vs-state-tokens.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    metric_chart("f1", "F1 comparison across transcript classifiers", "f1-comparison.png")
    metric_chart("accuracy", "Accuracy comparison across transcript classifiers", "accuracy-comparison.png")
    metrics_table()
    cost_chart()
    normalized_cost_quality_chart()
    normalized_cost_state_length_chart()
    print(f"Wrote charts to {CHARTS}")


if __name__ == "__main__":
    main()
