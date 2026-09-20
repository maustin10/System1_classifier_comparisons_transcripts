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
    jev_rate = assumptions["jev_price_usd_per_million_billed_input_tokens"]
    noul_overhead = assumptions["jev_noul_non_state_billed_tokens_per_transcript"]
    choice_overhead = assumptions["jev_choice_non_state_billed_tokens_per_transcript"]
    llm_input_overhead = assumptions["llm_non_state_input_tokens_per_transcript_proxy"]
    llm_output_tokens = assumptions["llm_standardized_mean_output_tokens_per_transcript"]
    sol_input_rate = assumptions["gpt_5_6_sol_input_usd_per_million_tokens"]
    sol_output_rate = assumptions["gpt_5_6_sol_output_usd_per_million_tokens"]
    luna_input_rate = assumptions["gpt_5_6_luna_input_usd_per_million_tokens"]
    luna_output_rate = assumptions["gpt_5_6_luna_output_usd_per_million_tokens"]
    overlap = assumptions["chunk_overlap_tokens"]
    modernbert_payload = assumptions["modernbert_nominal_state_payload_tokens_per_chunk"]
    gliclass_modern_payload = assumptions["gliclass_modern_nominal_state_payload_tokens_per_chunk"]
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
        + chunk_breaks(jev_payload)
    )))

    def chunk_count(values: np.ndarray, payload: float) -> np.ndarray:
        return np.maximum(1, np.ceil((values - overlap) / (payload - overlap))).astype(int)

    modernbert_chunks = chunk_count(state_tokens, modernbert_payload)
    gliclass_modern_chunks = chunk_count(state_tokens, gliclass_modern_payload)
    jev_requests = chunk_count(state_tokens, jev_payload)
    modernbert_encoded_state = state_tokens + overlap * (modernbert_chunks - 1)
    gliclass_modern_encoded_state = state_tokens + overlap * (gliclass_modern_chunks - 1)
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
        (
            "GPT-5.6 Luna",
            (
                luna_input_rate * (state_tokens + llm_input_overhead)
                + luna_output_rate * llm_output_tokens
            )
            / 1_000,
            "#F4A261",
            ":",
            "D",
        ),
        (
            "GPT-5.6 Sol",
            (
                sol_input_rate * (state_tokens + llm_input_overhead)
                + sol_output_rate * llm_output_tokens
            )
            / 1_000,
            "#303846",
            ":",
            "^",
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
    ax.set_ylim(2e-4, 2e2)
    ax.set_xlabel("State length (tokens, log scale)")
    ax.set_ylabel("Estimated USD per 1,000 transcripts (log scale)")
    x_ticks = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    ax.xaxis.set_major_locator(FixedLocator([tick for tick in x_ticks if lower <= tick <= upper]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.grid(color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9.5)
    fig.text(
        0.01,
        0.01,
        "Each curve is the estimated cost of 1,000 transcripts. Sol/Luna include standardized input plus 212 output tokens; hidden reasoning tokens are excluded. "
        "GLiClass Modern uses an H100 throughput proxy, not a measured H100 result. Long-context throughput changes are not modeled.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(CHARTS / "normalized-cost-vs-state-tokens.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


GENERIC_APPROACHES = [
    ("modernbert_trained", "Fixed trained encoder", "#0B678B", "-", None),
    ("gliclass_modern_large_v3", "Shared-state zero-shot encoder", "#D47900", "-.", "o"),
    ("modernbert_zero_shot", "Pairwise zero-shot encoder", "#2A9D8F", "-", None),
    ("jev_noul", "Hosted shared-state billing", "#8059C3", "-", None),
    ("gpt_5_6_luna", "Single-call LLM: Luna", "#F4A261", ":", "D"),
    ("gpt_5_6_sol", "Single-call LLM: Sol", "#303846", ":", "^"),
]


def generic_transcript_costs(state_tokens: np.ndarray | float, questions: np.ndarray | float) -> dict[str, np.ndarray]:
    """Return cost per 1,000 transcripts for generic state-length/taxonomy scenarios."""
    base = NORMALIZED_COST["state_length_sensitivity"]["assumptions"]
    generic = NORMALIZED_COST["generic_transcript_sensitivity"]
    proxy = generic["question_scaling_proxies"]
    state, question_count = np.broadcast_arrays(
        np.asarray(state_tokens, dtype=float),
        np.asarray(questions, dtype=float),
    )
    overlap = base["chunk_overlap_tokens"]

    def chunks(payload: np.ndarray | float) -> np.ndarray:
        return np.maximum(1, np.ceil((state - overlap) / (payload - overlap))).astype(int)

    modernbert_payload = base["modernbert_nominal_state_payload_tokens_per_chunk"]
    modernbert_chunks = chunks(modernbert_payload)
    modernbert_state = state + overlap * (modernbert_chunks - 1)

    label_tokens = proxy["gliclass_modern_non_state_tokens_per_label"] * question_count
    gliclass_payload = base["gliclass_modern_max_sequence_tokens"] - label_tokens
    gliclass_chunks = chunks(gliclass_payload)
    gliclass_state = state + overlap * (gliclass_chunks - 1)

    jev_payload = base["jev_nominal_state_payload_tokens_per_request"]
    jev_requests = chunks(jev_payload)
    jev_state = state + overlap * (jev_requests - 1)

    nli_question_tokens = proxy["modernbert_zero_shot_non_state_tokens_per_question"] * question_count
    jev_question_tokens = proxy["jev_noul_non_state_billed_tokens_per_question"] * question_count
    llm_input_tokens = proxy["llm_non_state_input_tokens_per_label"] * question_count
    llm_output_tokens = proxy["llm_output_tokens_per_label"] * question_count

    return {
        "modernbert_trained": base["modernbert_cost_usd_per_million_processed_tokens"]
        * (
            modernbert_state
            + base["modernbert_trained_special_tokens_per_transcript"] * modernbert_chunks
        )
        / 1_000,
        "gliclass_modern_large_v3": base["gliclass_modern_cost_usd_per_million_processed_tokens"]
        * (gliclass_state + label_tokens * gliclass_chunks)
        / 1_000,
        "modernbert_zero_shot": base["modernbert_cost_usd_per_million_processed_tokens"]
        * (question_count * modernbert_state + nli_question_tokens * modernbert_chunks)
        / 1_000,
        "jev_noul": base["jev_price_usd_per_million_billed_input_tokens"]
        * (jev_state + jev_question_tokens * jev_requests)
        / 1_000,
        "gpt_5_6_luna": (
            base["gpt_5_6_luna_input_usd_per_million_tokens"] * (state + llm_input_tokens)
            + base["gpt_5_6_luna_output_usd_per_million_tokens"] * llm_output_tokens
        )
        / 1_000,
        "gpt_5_6_sol": (
            base["gpt_5_6_sol_input_usd_per_million_tokens"] * (state + llm_input_tokens)
            + base["gpt_5_6_sol_output_usd_per_million_tokens"] * llm_output_tokens
        )
        / 1_000,
    }


def style_generic_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 2e2)
    ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.grid(color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def generic_duration_chart() -> None:
    generic = NORMALIZED_COST["generic_transcript_sensitivity"]
    tokens_per_minute = generic["state_tokens_per_minute"]
    duration_ticks = generic["duration_minutes"]
    durations = np.geomspace(min(duration_ticks), max(duration_ticks), 420)
    state_tokens = durations * tokens_per_minute

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.7), sharex=True, sharey=True)
    for ax, questions in zip(axes, generic["question_facets"]):
        costs = generic_transcript_costs(state_tokens, questions)
        marker_positions = np.unique(np.linspace(0, len(durations) - 1, 10, dtype=int))
        for key, label, color, linestyle, marker in GENERIC_APPROACHES:
            ax.plot(
                durations,
                costs[key],
                label=label,
                color=color,
                linewidth=2.6,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_positions if marker else None,
                markersize=5.5,
                markeredgecolor="white" if marker else None,
                markeredgewidth=0.8 if marker else None,
            )
        style_generic_axis(ax)
        ax.set_xscale("log")
        ax.set_xlim(min(duration_ticks), max(duration_ticks))
        ax.xaxis.set_major_locator(FixedLocator(duration_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_title(f"{questions} questions", fontsize=14, pad=30)
        ax.set_xlabel("Transcript duration (minutes)")
        top = ax.secondary_xaxis(
            "top",
            functions=(lambda minutes: minutes * tokens_per_minute, lambda tokens: tokens / tokens_per_minute),
        )
        top.set_xticks([value * tokens_per_minute for value in duration_ticks])
        top.set_xticklabels([f"{value * tokens_per_minute / 1000:g}k" for value in duration_ticks])
        top.tick_params(labelsize=8, colors=MUTED, pad=3)
    axes[0].set_ylabel("Estimated USD per 1,000 transcripts (log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.905), fontsize=10)
    fig.suptitle("Transcript classification cost by duration and taxonomy size", x=0.055, ha="left", fontsize=21, fontweight="bold")
    fig.text(
        0.055,
        0.925,
        "Top axes show state tokens · 200 state tokens/minute scenario · fixed encoder heads versus runtime questions",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.01,
        0.012,
        "Question overhead is scaled proportionally from the measured 27-question workload. Fixed trained heads require prior training; JEV shows billed tokens, not verified internal compute. Hidden LLM reasoning tokens are excluded.",
        color=MUTED,
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.995, 0.82), w_pad=2.0)
    fig.savefig(CHARTS / "generic-cost-vs-transcript-duration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generic_question_count_chart() -> None:
    generic = NORMALIZED_COST["generic_transcript_sensitivity"]
    lower, upper = generic["question_count_range"]
    questions = np.linspace(lower, upper, 400)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), sharex=True, sharey=True)
    for ax, scenario in zip(axes, generic["transcript_scenarios"]):
        costs = generic_transcript_costs(scenario["state_tokens"], questions)
        marker_positions = np.unique(np.linspace(0, len(questions) - 1, 10, dtype=int))
        for key, label, color, linestyle, marker in GENERIC_APPROACHES:
            ax.plot(
                questions,
                costs[key],
                label=label,
                color=color,
                linewidth=2.6,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_positions if marker else None,
                markersize=5.5,
                markeredgecolor="white" if marker else None,
                markeredgewidth=0.8 if marker else None,
            )
        style_generic_axis(ax)
        ax.set_xlim(lower, upper)
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 25, 50, 75, 100]))
        ax.set_title(
            f"{scenario['label']}: {scenario['minutes']} min / {scenario['state_tokens'] / 1000:g}k tokens",
            fontsize=13,
            pad=14,
        )
        ax.set_xlabel("Classification questions / labels")
    axes[0].set_ylabel("Estimated USD per 1,000 transcripts (log scale)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.895), fontsize=10)
    fig.suptitle("Transcript classification cost as the taxonomy grows", x=0.055, ha="left", fontsize=21, fontweight="bold")
    fig.text(
        0.055,
        0.925,
        "Short, medium, and long transcript scenarios · pairwise NLI repeats the state for every runtime question",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.01,
        0.012,
        "Fixed trained-head cost is nearly flat because head compute is treated as negligible. Runtime-label systems scale label, billing, or output overhead without repeating the full state in the external cost model.",
        color=MUTED,
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.995, 0.81), w_pad=2.0)
    fig.savefig(CHARTS / "generic-cost-vs-question-count.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


EXECUTIVE_MODELS = [
    ("modernbert_trained", "Fixed trained encoder", "ModernBERT-large + trained heads", "modernbert_trained", 0.20, "#0B678B"),
    ("gliclass_modern_large_v3", "Shared-state zero-shot", "GLiClass Modern Large v3", "gliclass_modern_large_v3_opt_threshold", 0.74, "#D47900"),
    ("jev_noul", "Hosted shared-state API", "TypeSafe.ai JEV Noul", "jev_noul_calibrated", 0.86, "#8059C3"),
    ("modernbert_zero_shot", "Pairwise zero-shot", "ModernBERT-large NLI", "modernbert_zero_shot_opt_threshold", 0.67, "#2A9D8F"),
    ("gpt_5_6_luna", "Single-call LLM", "Luna", "luna", 0.94, "#F4A261"),
    ("gpt_5_6_sol", "Single-call LLM", "Sol", "sol", 0.94, "#303846"),
]


def executive_decision_matrix() -> None:
    """Build an executive 2x2 using one generic state/question workload."""
    generic = NORMALIZED_COST["generic_transcript_sensitivity"]
    reference = generic["executive_matrix_reference"]
    costs = generic_transcript_costs(reference["state_tokens"], reference["questions"])
    quality = {item["id"]: 100 * item["accuracy"] for item in SUMMARY["models"]}
    boundary = reference["economical_cost_boundary_usd_per_1000"]

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(50, 0.025)
    ax.set_yscale("log")
    ax.axvline(0.5, color=GRID, linewidth=1.4)
    ax.axhline(boundary, color=GRID, linewidth=1.4)
    ax.axvspan(0.5, 1.08, ymin=0.5, ymax=1, color="#EAF5EF", alpha=0.9, zorder=0)

    ax.text(0.04, 0.96, "ECONOMICAL / FIXED", transform=ax.transAxes, color=MUTED, fontsize=10, fontweight="bold", va="top")
    ax.text(0.53, 0.96, "ECONOMICAL / FLEXIBLE", transform=ax.transAxes, color="#27805A", fontsize=10, fontweight="bold", va="top")
    ax.text(0.04, 0.06, "PREMIUM / FIXED", transform=ax.transAxes, color=MUTED, fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.53, 0.06, "PREMIUM / FLEXIBLE", transform=ax.transAxes, color=MUTED, fontsize=10, fontweight="bold", va="bottom")

    offsets = {
        "modernbert_trained": (12, -22),
        "gliclass_modern_large_v3": (12, 18),
        "jev_noul": (12, 12),
        "modernbert_zero_shot": (-142, -30),
        "gpt_5_6_luna": (-155, -5),
        "gpt_5_6_sol": (-125, -5),
    }
    for cost_key, _, product, quality_key, x, color in EXECUTIVE_MODELS:
        cost = float(costs[cost_key])
        accuracy = quality[quality_key]
        ax.scatter(x, cost, s=260, color=color, edgecolor="white", linewidth=1.8, zorder=4)
        dx, dy = offsets[cost_key]
        ax.annotate(
            f"{product}\n${cost:.3f} / 1k · {accuracy:.2f}% accuracy",
            (x, cost),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.6,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
        )

    ax.set_xticks([0.20, 0.80], ["Fixed taxonomy", "Questions can change at runtime"])
    ax.tick_params(axis="x", labelsize=11, pad=10)
    ax.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3, 10, 30]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    ax.set_ylabel("Estimated USD per 1,000 states (lower is better)")
    ax.grid(axis="y", color=GRID, linewidth=0.7, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Executive decision map: economics versus runtime flexibility", loc="left", pad=22)
    ax.text(
        0,
        1.015,
        f"Reference workload: {reference['state_tokens']:,} state tokens × {reference['questions']} questions · self-hosted cost at 100% H100 utilization",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.01,
        0.012,
        "The $1/1,000 boundary is an executive guide, not a universal purchasing threshold. Flexible systems accept new questions without retraining; trained heads do not.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.07, 0.99, 0.95))
    fig.savefig(CHARTS / "executive-decision-matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def executive_gpu_utilization_bars() -> None:
    """Show the effect of paid H100 utilization on self-hosted economics."""
    generic = NORMALIZED_COST["generic_transcript_sensitivity"]
    reference = generic["executive_matrix_reference"]
    utilizations = generic["executive_gpu_utilization_percent"]
    base_costs = generic_transcript_costs(reference["state_tokens"], reference["questions"])
    models = [
        ("modernbert_trained", "Fixed trained encoder\nModernBERT-large + heads", "#0B678B", True),
        ("gliclass_modern_large_v3", "Shared-state zero-shot\nGLiClass Modern Large v3", "#D47900", True),
        ("jev_noul", "Hosted shared-state API\nTypeSafe.ai JEV Noul", "#8059C3", False),
        ("modernbert_zero_shot", "Pairwise zero-shot\nModernBERT-large NLI", "#2A9D8F", True),
        ("gpt_5_6_luna", "Single-call LLM\nGPT-5.6 Luna", "#F4A261", False),
        ("gpt_5_6_sol", "Single-call LLM\nGPT-5.6 Sol", "#303846", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), sharex=True)
    for ax, utilization_pct in zip(axes.flat, utilizations):
        utilization = utilization_pct / 100
        costs = {
            key: float(base_costs[key]) / utilization if self_hosted else float(base_costs[key])
            for key, _, _, self_hosted in models
        }
        ordered = sorted(models, key=lambda item: costs[item[0]], reverse=True)
        values = [costs[key] for key, _, _, _ in ordered]
        labels = [label for _, label, _, _ in ordered]
        colors = [color for _, _, color, _ in ordered]
        y = np.arange(len(ordered))
        bars = ax.barh(
            y,
            values,
            color=colors,
            height=0.62,
            edgecolor=["#4B2A79" if key == "jev_noul" else "white" for key, _, _, _ in ordered],
            linewidth=[2.4 if key == "jev_noul" else 0.8 for key, _, _, _ in ordered],
        )
        ax.set_xscale("log")
        ax.set_xlim(0.025, 400)
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", labelsize=9.2)
        ax.grid(axis="x", color=GRID, linewidth=0.8, which="major")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(f"{utilization_pct}% paid H100 utilization", fontsize=14, pad=12)
        for bar, value in zip(bars, values):
            ax.text(
                min(value * 1.14, 330),
                bar.get_y() + bar.get_height() / 2,
                f"${value:.3f}" if value < 10 else f"${value:.1f}",
                va="center",
                fontsize=9.2,
                fontweight="bold",
            )
        ax.text(
            0.98,
            0.97,
            "Purple JEV stays at $0.352",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#6841A8",
            fontsize=9,
            fontweight="bold",
        )

    for ax in axes[1, :]:
        ax.set_xlabel("Estimated USD per 1,000 states (log scale)")

    trained_break_even = 100 * float(base_costs["modernbert_trained"]) / float(base_costs["jev_noul"])
    gliclass_break_even = 100 * float(base_costs["gliclass_modern_large_v3"]) / float(base_costs["jev_noul"])
    fig.suptitle("GPU utilization determines self-hosted encoder economics", x=0.055, ha="left", fontsize=21, fontweight="bold")
    fig.text(
        0.055,
        0.94,
        f"Fixed workload: {reference['state_tokens'] / 1000:g}k state tokens × {reference['questions']} questions · NVIDIA H100 at USD 5/hour · lower is better",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.01,
        0.012,
        f"Self-hosted encoder cost is divided by paid GPU utilization; hosted JEV/Luna/Sol prices remain usage-based. In this scenario, JEV crosses trained ModernBERT near {trained_break_even:.1f}% utilization and GLiClass Modern near {gliclass_break_even:.1f}%. Hidden LLM reasoning tokens are excluded.",
        color=MUTED,
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.03, 0.055, 0.995, 0.90), w_pad=4.0, h_pad=2.1)
    fig.savefig(CHARTS / "executive-cost-vs-gpu-utilization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    metric_chart("f1", "F1 comparison across transcript classifiers", "f1-comparison.png")
    metric_chart("accuracy", "Accuracy comparison across transcript classifiers", "accuracy-comparison.png")
    metrics_table()
    cost_chart()
    normalized_cost_quality_chart()
    normalized_cost_state_length_chart()
    generic_duration_chart()
    generic_question_count_chart()
    executive_decision_matrix()
    executive_gpu_utilization_bars()
    print(f"Wrote charts to {CHARTS}")


if __name__ == "__main__":
    main()
