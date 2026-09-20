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

    fig, ax = plt.subplots(figsize=(max(15.5, 1.62 * len(models)), 8.2))
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
    ax.tick_params(axis="x", labelrotation=0, pad=10, labelsize=9)
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

    fig, ax = plt.subplots(figsize=(max(15.5, 1.62 * len(ordered_ids)), 8.2))
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
    ax.tick_params(axis="x", pad=10, labelsize=9)
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
        "gliclass_modern_large_v3": (8, -20),
        "gliclass_modern_large_v3_opt_threshold": (8, 10),
        "gliclass_large_v3": (8, -18),
        "gliclass_large_v3_opt_threshold": (8, 10),
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
    ax.legend(handles, ["Open encoder", "TypeSafe.ai JEV"], frameon=False, ncol=2, loc="lower left")
    fig.text(
        0.01,
        0.01,
        "Open encoders: simulated H100 serving costs from the stated throughput proxies, not exact-checkpoint H100 benchmarks. "
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
    encoder_rate = assumptions["modernbert_cost_usd_per_million_processed_tokens"]
    questions = assumptions["questions_per_transcript"]
    zero_overhead = assumptions["modernbert_zero_shot_non_state_tokens_per_transcript"]
    trained_overhead = assumptions["modernbert_trained_special_tokens_per_transcript"]
    gliclass_modern_rate = assumptions["gliclass_modern_cost_usd_per_million_processed_tokens"]
    gliclass_modern_overhead = assumptions["gliclass_modern_non_state_tokens_per_pass"]
    gliclass_large_rate = assumptions["gliclass_large_cost_usd_per_million_processed_tokens"]
    gliclass_large_overhead = assumptions["gliclass_large_non_state_tokens_per_chunk_total"]
    gliclass_large_passes = assumptions["gliclass_large_forward_passes_per_chunk"]
    jev_rate = assumptions["jev_price_usd_per_million_billed_input_tokens"]
    noul_overhead = assumptions["jev_noul_non_state_billed_tokens_per_transcript"]
    choice_overhead = assumptions["jev_choice_non_state_billed_tokens_per_transcript"]
    overlap = assumptions["chunk_overlap_tokens"]
    modernbert_payload = assumptions["modernbert_nominal_state_payload_tokens_per_chunk"]
    gliclass_modern_payload = assumptions["gliclass_modern_nominal_state_payload_tokens_per_chunk"]
    gliclass_large_payload = assumptions["gliclass_large_nominal_state_payload_tokens_per_chunk"]
    jev_payload = assumptions["jev_nominal_state_payload_tokens_per_request"]

    def chunk_breaks(payload: float) -> list[float]:
        boundaries = []
        covered = payload
        while covered < upper:
            boundaries.extend([covered * (1 - 1e-6), covered * (1 + 1e-6)])
            covered += payload - overlap
        return boundaries

    state_tokens = np.array(sorted(set(
        np.geomspace(lower, upper, 900).tolist()
        + chunk_breaks(modernbert_payload)
        + chunk_breaks(gliclass_modern_payload)
        + chunk_breaks(gliclass_large_payload)
        + chunk_breaks(jev_payload)
    )))

    def chunk_count(values: np.ndarray, payload: float) -> np.ndarray:
        return np.maximum(1, np.ceil((values - overlap) / (payload - overlap))).astype(int)

    modernbert_chunks = chunk_count(state_tokens, modernbert_payload)
    gliclass_modern_chunks = chunk_count(state_tokens, gliclass_modern_payload)
    gliclass_large_chunks = chunk_count(state_tokens, gliclass_large_payload)
    jev_requests = chunk_count(state_tokens, jev_payload)
    modernbert_encoded_state = state_tokens + overlap * (modernbert_chunks - 1)
    gliclass_modern_encoded_state = state_tokens + overlap * (gliclass_modern_chunks - 1)
    gliclass_large_encoded_state = state_tokens + overlap * (gliclass_large_chunks - 1)
    jev_billed_state = state_tokens + overlap * (jev_requests - 1)

    curves = [
        (
            "ModernBERT trained heads",
            encoder_rate * (modernbert_encoded_state + trained_overhead * modernbert_chunks) / 1_000,
            "#0B678B",
            "-",
            None,
        ),
        (
            "GLiClass Modern Large (sim.)",
            gliclass_modern_rate
            * (gliclass_modern_encoded_state + gliclass_modern_overhead * gliclass_modern_chunks)
            / 1_000,
            "#D47900",
            "-.",
            "o",
        ),
        (
            "GLiClass Large, 2 groups (sim.)",
            gliclass_large_rate
            * (
                gliclass_large_passes * gliclass_large_encoded_state
                + gliclass_large_overhead * gliclass_large_chunks
            )
            / 1_000,
            "#C44E52",
            "-.",
            "s",
        ),
        (
            "ModernBERT zero-shot",
            encoder_rate * (questions * modernbert_encoded_state + zero_overhead * modernbert_chunks) / 1_000,
            "#2A9D8F",
            "-",
            None,
        ),
        (
            "JEV Noul",
            jev_rate * (jev_billed_state + noul_overhead * jev_requests) / 1_000,
            "#8059C3",
            "-",
            None,
        ),
        (
            "JEV Choice",
            jev_rate * (jev_billed_state + choice_overhead * jev_requests) / 1_000,
            "#A78BDB",
            "--",
            None,
        ),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    marker_positions = np.unique(np.linspace(0, len(state_tokens) - 1, 14, dtype=int))
    for label, values, color, linestyle, marker in curves:
        ax.plot(
            state_tokens,
            values,
            label=label,
            color=color,
            linewidth=3.2 if marker else 2.8,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_positions if marker else None,
            markersize=6.5,
            markeredgecolor="white" if marker else None,
            markeredgewidth=0.9 if marker else None,
            zorder=4 if marker else 3,
        )

    crossovers = sensitivity["crossovers_with_modernbert_zero_shot"]
    noul_cross = crossovers["jev_noul_state_tokens"]
    choice_cross = crossovers["jev_choice_state_tokens"]
    zero_cost = lambda s: encoder_rate * (questions * s + zero_overhead) / 1_000
    ax.axvspan(noul_cross, upper, color="#8059C3", alpha=0.045, zorder=0)
    ax.axvline(224.46, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(224.46, 0.89, "benchmark mean\n224 tokens", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, ha="center")
    for boundary, label, color, y_position in [
        (gliclass_large_payload, "GLiClass Large payload\n357 state tokens", "#C44E52", 0.47),
        (modernbert_payload, "ModernBERT / GLiClass Modern\nchunking starts near 8k", "#2A9D8F", 0.72),
        (jev_payload, "JEV request chunking\nstarts near 31.9k", "#8059C3", 0.72),
    ]:
        ax.axvline(boundary, color=color, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.text(boundary, y_position, label, transform=ax.get_xaxis_transform(), color=color, fontsize=9, ha="center")
    for crossover, label, offset in [
        (noul_cross, "Noul crossover\n586 tokens", (-35, 52)),
        (choice_cross, "Choice crossover\n682 tokens", (55, 30)),
    ]:
        ax.scatter(crossover, zero_cost(crossover), s=70, color="#8059C3", edgecolor="white", zorder=5)
        ax.annotate(
            label,
            xy=(crossover, zero_cost(crossover)),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
        )

    ax.set_title("Estimated cost per 1,000 transcripts by state length", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "27 questions / labels · 256-token chunk overlap · H100 at USD 5/hour · lower cost is better",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(2e-4, 2e1)
    ax.set_xlabel("State length (tokens, log scale)")
    ax.set_ylabel("Estimated USD per 1,000 transcripts (log scale)")
    x_ticks = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    ax.xaxis.set_major_locator(FixedLocator([tick for tick in x_ticks if lower <= tick <= upper]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.grid(color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Each curve is the estimated cost of 1,000 transcripts. GLiClass Modern uses the 170.3k token/s H100 proxy; GLiClass Large uses 109.6k token/s, "
        "scaled by the official 32-label A6000 throughput ratio. These are simulations, not measured H100 results. Long-context throughput changes are not modeled.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
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
