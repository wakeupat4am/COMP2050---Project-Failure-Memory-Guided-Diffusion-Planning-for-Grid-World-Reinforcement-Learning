from __future__ import annotations

import os

cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures", ".cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHM_ORDER = [
    "Random Policy",
    "Value Iteration",
    "Policy Iteration",
    "Q-learning",
    "MCTS",
    "Standard Diffusion",
    "Failure-Memory Diffusion",
    "Improved Failure-Memory Diffusion",
]

ALGORITHM_COLORS = {
    "Random Policy": "#7f8c8d",
    "Value Iteration": "#1f77b4",
    "Policy Iteration": "#17becf",
    "Q-learning": "#ff7f0e",
    "MCTS": "#2ca02c",
    "Standard Diffusion": "#d62728",
    "Failure-Memory Diffusion": "#9467bd",
    "Improved Failure-Memory Diffusion": "#8c564b",
}

AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 14
LEGEND_TITLE_FONT_SIZE = 14


def _ordered_algorithms(df: pd.DataFrame) -> list[str]:
    # Keep a stable plotting order so figures remain comparable across different experiment subsets.
    present = set(df["Algorithm"].tolist())
    ordered = [name for name in ALGORITHM_ORDER if name in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def _plot_grouped_bars(
    df: pd.DataFrame,
    value_col: str,
    error_col: str | None,
    ylabel: str,
    title: str,
    output_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    # Shared bar-chart template for the headline benchmark metrics.
    maps = list(df["Map"].drop_duplicates())
    algorithms = _ordered_algorithms(df)
    x = np.arange(len(maps))
    width = 0.11 if len(algorithms) >= 6 else 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, algorithm in enumerate(algorithms):
        subset = (
            df[df["Algorithm"] == algorithm]
            .set_index("Map")
            .reindex(maps)
        )
        offsets = x + (idx - (len(algorithms) - 1) / 2.0) * width
        errors = subset[error_col].to_numpy() if error_col is not None and error_col in subset.columns else None
        ax.bar(
            offsets,
            subset[value_col].to_numpy(),
            width=width,
            label=algorithm,
            yerr=errors,
            capsize=3 if errors is not None else 0,
            color=ALGORITHM_COLORS.get(algorithm, None),
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in maps], fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        columnspacing=1.4,
        handletextpad=0.7,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_success_rate_table(df: pd.DataFrame, output_dir: str) -> None:
    _plot_grouped_bars(
        df=df,
        value_col="Success Rate Mean",
        error_col="Success Rate CI Halfwidth",
        ylabel="Success Rate",
        title="Success Rate by Algorithm and Map",
        output_path=os.path.join(output_dir, "success_rate_comparison.png"),
        ylim=(0.0, 1.05),
    )


def plot_average_return_table(df: pd.DataFrame, output_dir: str) -> None:
    _plot_grouped_bars(
        df=df,
        value_col="Average Return Mean",
        error_col="Average Return CI Halfwidth",
        ylabel="Average Return",
        title="Average Return by Algorithm and Map",
        output_path=os.path.join(output_dir, "average_return_comparison.png"),
    )


def plot_collision_rate_table(df: pd.DataFrame, output_dir: str) -> None:
    _plot_grouped_bars(
        df=df,
        value_col="Collision Rate Mean",
        error_col="Collision Rate CI Halfwidth",
        ylabel="Collision Rate",
        title="Collision Rate by Algorithm and Map",
        output_path=os.path.join(output_dir, "collision_rate_comparison.png"),
        ylim=(0.0, 1.05),
    )


def plot_inference_time_table(df: pd.DataFrame, output_dir: str) -> None:
    _plot_grouped_bars(
        df=df,
        value_col="Inference Time Mean",
        error_col="Inference Time CI Halfwidth",
        ylabel="Seconds per Action",
        title="Inference Time per Action by Algorithm and Map",
        output_path=os.path.join(output_dir, "inference_time_comparison.png"),
    )


def plot_metric_heatmap(df: pd.DataFrame, output_dir: str, metric: str, filename: str, title: str) -> None:
    # Heatmaps provide a compact cross-map summary when exact mean values still matter.
    pivot = df.pivot(index="Algorithm", columns="Map", values=metric)
    pivot = pivot.reindex(_ordered_algorithms(df))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=TICK_LABEL_FONT_SIZE)

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iloc[row_idx, col_idx]
            label = "NaN" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diffusion_focus(df: pd.DataFrame, output_dir: str) -> None:
    # Highlight only diffusion-family planners when comparing their internal tradeoffs.
    focus_algorithms = [
        name
        for name in [
            "Standard Diffusion",
            "Failure-Memory Diffusion",
            "Improved Failure-Memory Diffusion",
        ]
        if name in set(df["Algorithm"])
    ]
    focus = df[df["Algorithm"].isin(focus_algorithms)].copy()
    maps = list(df["Map"].drop_duplicates())
    metrics = [
        ("Success Rate Mean", "Success Rate"),
        ("Average Return Mean", "Average Return"),
        ("Collision Rate Mean", "Collision Rate"),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric_col, title) in zip(axes, metrics):
        error_col = metric_col.replace("Mean", "CI Halfwidth")
        for algorithm in focus_algorithms:
            subset = focus[focus["Algorithm"] == algorithm].set_index("Map").reindex(maps)
            ax.errorbar(
                maps,
                subset[metric_col].to_numpy(),
                yerr=subset[error_col].to_numpy() if error_col in subset.columns else None,
                marker="o",
                linewidth=2.2,
                label=algorithm,
                color=ALGORITHM_COLORS[algorithm],
                capsize=3,
            )
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=20, labelsize=TICK_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)

    axes[0].set_ylim(0.0, 1.05)
    axes[2].set_ylim(0.0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.4,
        handletextpad=0.7,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(os.path.join(output_dir, "diffusion_focus_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_all_plots(df: pd.DataFrame, output_dir: str) -> None:
    # Main benchmark plotting bundle used by the comparison scripts.
    os.makedirs(output_dir, exist_ok=True)
    plot_success_rate_table(df, output_dir)
    plot_average_return_table(df, output_dir)
    plot_collision_rate_table(df, output_dir)
    plot_inference_time_table(df, output_dir)
    plot_metric_heatmap(
        df,
        output_dir,
        metric="Success Rate Mean",
        filename="success_rate_heatmap.png",
        title="Success Rate Heatmap",
    )
    plot_metric_heatmap(
        df,
        output_dir,
        metric="Average Return Mean",
        filename="average_return_heatmap.png",
        title="Average Return Heatmap",
    )
    plot_diffusion_focus(df, output_dir)


def plot_lambda_failure_ablation(df: pd.DataFrame, output_dir: str) -> None:
    # Show how the failure-memory penalty weight changes performance and stability.
    os.makedirs(output_dir, exist_ok=True)
    x = df["lambda_F"].to_numpy()
    metrics = [
        ("Success Rate Mean", "Success Rate", "#2ca02c"),
        ("Collision Rate Mean", "Collision Rate", "#d62728"),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate", "#9467bd"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (column, title, color) in zip(axes, metrics):
        error_col = column.replace("Mean", "CI Halfwidth")
        ax.errorbar(x, df[column].to_numpy(), yerr=df[error_col].to_numpy(), marker="o", linewidth=2.2, color=color, capsize=3)
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel(r"$\lambda_F$", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if "Rate" in title:
            ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_failure_ablation.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_k_ablation(df: pd.DataFrame, output_dir: str) -> None:
    # Plot sensitivity to the number of sampled candidate trajectories.
    os.makedirs(output_dir, exist_ok=True)
    x = df["K"].to_numpy()
    metrics = [
        ("Success Rate Mean", "Success Rate", "#2ca02c"),
        ("Collision Rate Mean", "Collision Rate", "#d62728"),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate", "#9467bd"),
        ("Average Return Mean", "Average Return", "#1f77b4"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (column, title, color) in zip(axes, metrics):
        error_col = column.replace("Mean", "CI Halfwidth")
        ax.errorbar(x, df[column].to_numpy(), yerr=df[error_col].to_numpy(), marker="o", linewidth=2.2, color=color, capsize=3)
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel("K", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if "Rate" in title:
            ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[2].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_ablation_failure_memory.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_standard_lambda_ablation(df: pd.DataFrame, output_dir: str) -> None:
    # Mirror the lambda sweep for the standard diffusion planner's distance penalty.
    os.makedirs(output_dir, exist_ok=True)
    x = df["lambda_D"].to_numpy()
    metrics = [
        ("Success Rate Mean", "Success Rate", "#2ca02c"),
        ("Collision Rate Mean", "Collision Rate", "#d62728"),
        ("Average Return Mean", "Average Return", "#1f77b4"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (column, title, color) in zip(axes, metrics):
        error_col = column.replace("Mean", "CI Halfwidth")
        ax.errorbar(x, df[column].to_numpy(), yerr=df[error_col].to_numpy(), marker="o", linewidth=2.2, color=color, capsize=3)
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel(r"$\lambda_D$", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if "Rate" in title:
            ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_distance_ablation_standard_diffusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_standard_k_ablation(df: pd.DataFrame, output_dir: str) -> None:
    # Mirror the candidate-count sweep for the standard diffusion planner.
    os.makedirs(output_dir, exist_ok=True)
    x = df["K"].to_numpy()
    metrics = [
        ("Success Rate Mean", "Success Rate", "#2ca02c"),
        ("Collision Rate Mean", "Collision Rate", "#d62728"),
        ("Average Return Mean", "Average Return", "#1f77b4"),
        ("Inference Time Mean", "Inference Time", "#ff7f0e"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (column, title, color) in zip(axes, metrics):
        error_col = column.replace("Mean", "CI Halfwidth")
        yerr = df[error_col].to_numpy() if error_col in df.columns else None
        ax.errorbar(x, df[column].to_numpy(), yerr=yerr, marker="o", linewidth=2.2, color=color, capsize=3)
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel("K", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if "Rate" in title:
            ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[2].set_ylabel("Metric Value", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_ablation_standard_diffusion.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_best_diffusion_variants(df: pd.DataFrame, output_dir: str) -> None:
    # Final side-by-side summary for the tuned standard and failure-memory diffusion variants.
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["Success Rate", "Collision Rate", "Repeated Failure Rate", "Average Return"]
    colors = ["#d62728", "#9467bd"]
    planners = df["Planner"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    x = np.arange(len(planners))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, df[metric].to_numpy(), color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(planners, rotation=10, fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_title(metric, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if "Rate" in metric:
            ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_diffusion_variant_comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_exploration_plots(df: pd.DataFrame, output_dir: str) -> None:
    # Plot the exploration-variant benchmark with the same grouped-bar style across metrics.
    os.makedirs(output_dir, exist_ok=True)
    metric_specs = [
        ("Success Rate Mean", "Success Rate", "exploration_success_rate.png", (0.0, 1.05)),
        ("Average Return Mean", "Average Return", "exploration_average_return.png", None),
        ("Collision Rate Mean", "Collision Rate", "exploration_collision_rate.png", (0.0, 1.05)),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate", "exploration_repeated_failure_rate.png", (0.0, 1.05)),
        ("Inference Time Mean", "Inference Time per Action", "exploration_inference_time.png", None),
        ("Optimality Gap Mean", "Optimality Gap", "exploration_optimality_gap.png", None),
    ]

    maps = list(df["Map"].drop_duplicates())
    algorithms = list(df["Algorithm"].drop_duplicates())
    x = np.arange(len(maps))
    width = 0.14

    for value_col, ylabel, filename, ylim in metric_specs:
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, algorithm in enumerate(algorithms):
            subset = df[df["Algorithm"] == algorithm].set_index("Map").reindex(maps)
            offsets = x + (idx - (len(algorithms) - 1) / 2.0) * width
            error_col = value_col.replace("Mean", "CI Halfwidth")
            ax.bar(
                offsets,
                subset[value_col].to_numpy(),
                width=width,
                label=algorithm,
                yerr=subset[error_col].to_numpy() if error_col in subset.columns else None,
                capsize=3 if error_col in subset.columns else 0,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in maps], fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_FONT_SIZE,
            columnspacing=1.4,
            handletextpad=0.7,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")
        plt.close(fig)


def create_component_ablation_plots(df: pd.DataFrame, output_dir: str) -> None:
    # Plot remove-one ablations to show which exploration components matter most.
    os.makedirs(output_dir, exist_ok=True)
    metric_specs = [
        ("Success Rate Mean", "Success Rate", "component_ablation_success_rate.png", (0.0, 1.05)),
        ("Average Return Mean", "Average Return", "component_ablation_average_return.png", None),
        ("Collision Rate Mean", "Collision Rate", "component_ablation_collision_rate.png", (0.0, 1.05)),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate", "component_ablation_repeated_failure_rate.png", (0.0, 1.05)),
    ]

    maps = list(df["Map"].drop_duplicates())
    variants = list(df["Algorithm"].drop_duplicates())
    x = np.arange(len(maps))
    width = 0.15 if len(variants) <= 5 else 0.12

    for value_col, ylabel, filename, ylim in metric_specs:
        fig, ax = plt.subplots(figsize=(13, 6.5))
        for idx, variant in enumerate(variants):
            subset = df[df["Algorithm"] == variant].set_index("Map").reindex(maps)
            offsets = x + (idx - (len(variants) - 1) / 2.0) * width
            error_col = value_col.replace("Mean", "CI Halfwidth")
            ax.bar(
                offsets,
                subset[value_col].to_numpy(),
                width=width,
                label=variant,
                yerr=subset[error_col].to_numpy() if error_col in subset.columns else None,
                capsize=3 if error_col in subset.columns else 0,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in maps], fontsize=TICK_LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_FONT_SIZE,
            columnspacing=1.4,
            handletextpad=0.7,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")
        plt.close(fig)
