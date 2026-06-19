from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "final_image" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


OUTPUT_DIR = PROJECT_ROOT / "final_image"
LEGEND_FONT_SIZE = 18


ALGO_ORDER = [
    "Random Policy",
    "Value Iteration",
    "Policy Iteration",
    "Q-learning",
    "MCTS",
    "Standard Diffusion",
    "Failure-Memory Diffusion",
    "Improved Failure-Memory Diffusion",
]

ALGO_COLORS = {
    "Random Policy": "#7f8c8d",
    "Value Iteration": "#1f77b4",
    "Policy Iteration": "#17becf",
    "Q-learning": "#ff7f0e",
    "MCTS": "#2ca02c",
    "Standard Diffusion": "#d62728",
    "Failure-Memory Diffusion": "#9467bd",
    "Improved Failure-Memory Diffusion": "#8c564b",
}

EXPLORATION_ORDER = [
    "Failure-Memory Baseline",
    "Dead-End Memory",
    "Diverse Candidates",
    "Adaptive Failure",
    "Combined Exploration",
]

EXPLORATION_COLORS = {
    "Failure-Memory Baseline": "#9467bd",
    "Dead-End Memory": "#7f8c8d",
    "Diverse Candidates": "#d62728",
    "Adaptive Failure": "#2ca02c",
    "Combined Exploration": "#8c564b",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def arrow_between(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#1f2933"),
    )


def _wrap_text(text: str, width: int) -> str:
    return textwrap.fill(text, width=max(1, width), break_long_words=False, break_on_hyphens=False)


def _fit_text(ax, x, y, text, *, max_fontsize, min_fontsize, width_chars, **kwargs):
    wrapped = _wrap_text(text, width_chars)
    txt = ax.text(x, y, wrapped, fontsize=max_fontsize, **kwargs)
    renderer = ax.figure.canvas.get_renderer()
    while txt.get_window_extent(renderer=renderer).width > ax.bbox.width * 0.9 and txt.get_fontsize() > min_fontsize:
        txt.set_fontsize(txt.get_fontsize() - 0.5)
    return txt


def box(ax, xy, w, h, title, lines, fc, ec):
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(rect)
    x, y = xy
    title_chars = max(10, int(w * 95))
    line_chars = max(12, int(w * 110))
    _fit_text(
        ax,
        x + w / 2,
        y + h * 0.73,
        title,
        max_fontsize=16,
        min_fontsize=12,
        width_chars=title_chars,
        ha="center",
        va="center",
        fontweight="bold",
        color="#1f2933",
        linespacing=1.0,
    )
    for i, line in enumerate(lines):
        _fit_text(
            ax,
            x + w / 2,
            y + h * (0.45 - i * 0.17),
            line,
            max_fontsize=13,
            min_fontsize=10,
            width_chars=line_chars,
            ha="center",
            va="center",
            color="#25313c",
            linespacing=1.0,
        )


def load_benchmark_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "Exploration" / "benchmark_results" / "tables" / "improved_benchmark_comparison.csv")


def load_exploration_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "Exploration" / "results" / "tables" / "exploration_comparison.csv")


def load_lambda_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "results" / "tables" / "failure_memory_lambda_ablation.csv")


def load_failure_k_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "results" / "tables" / "failure_memory_k_ablation.csv")


def load_standard_k_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "results" / "tables" / "standard_diffusion_k_ablation.csv")


def plot_final_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")

    ax.text(0.03, 0.94, "Final Method Overview", fontsize=24, fontweight="bold", color="#1f2933")
    ax.text(0.03, 0.89, "Current state and goal -> candidate generation -> scoring -> memory guidance -> action -> feedback", fontsize=13, color="#44515c")

    box(ax, (0.04, 0.58), 0.14, 0.2, "State + Goal", ["current row/col", "goal row/col"], "#dceefb", "#2f6b8a")
    box(ax, (0.23, 0.58), 0.15, 0.2, "Diffusion Model", ["sample K x 3 raw", "action sequences"], "#ffe3c2", "#b96a20")
    box(ax, (0.43, 0.58), 0.15, 0.2, "Simulation", ["simulate each", "candidate path"], "#eadbf7", "#7a4da1")
    box(ax, (0.63, 0.58), 0.15, 0.2, "Failure Guidance", ["dead-end memory", "adaptive weights"], "#d8f0d2", "#3a7a2d")
    box(ax, (0.83, 0.58), 0.13, 0.2, "Select Action", ["best score", "take first action"], "#f8dede", "#9c3939")
    box(ax, (0.36, 0.18), 0.28, 0.18, "Environment Feedback", ["success / collision / timeout", "update memory using failed tail"], "#ffffff", "#1f2933")

    arrow_between(ax, 0.18, 0.68, 0.23, 0.68)
    arrow_between(ax, 0.38, 0.68, 0.43, 0.68)
    arrow_between(ax, 0.58, 0.68, 0.63, 0.68)
    arrow_between(ax, 0.78, 0.68, 0.83, 0.68)
    arrow_between(ax, 0.895, 0.58, 0.55, 0.36)

    ax.text(0.5, 0.08, r"Score($\tau$) = R($\tau$) - $\lambda_{distance}$ D(final, goal) - $\lambda_{failure}$ F($\tau$) - $\lambda_{loop}$ LoopPenalty($\tau$)", ha="center", fontsize=14, color="#1f2933")
    save_fig(fig, "final_method_overview.png")


def draw_grid(ax, grid, start, goal, title):
    rows, cols = grid.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            color = "#ffffff" if val == 0 else "#283845"
            rect = patches.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#8b8b8b", linewidth=1.0)
            ax.add_patch(rect)
    sr, sc = start
    gr, gc = goal
    ax.text(sc + 0.5, sr + 0.62, "S", ha="center", va="center", fontsize=20, fontweight="bold", color="#0b7a3a")
    ax.text(gc + 0.5, gr + 0.62, "G", ha="center", va="center", fontsize=20, fontweight="bold", color="#c0392b")
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                ax.text(c + 0.5, r + 0.62, "X", ha="center", va="center", fontsize=16, fontweight="bold", color="#ffffff")
            elif (r, c) not in [start, goal]:
                ax.text(c + 0.5, r + 0.62, ".", ha="center", va="center", fontsize=14, color="#9aa5b1")
    ax.set_title(title, fontsize=14, fontweight="bold")


def plot_map_scenes() -> None:
    from utils.map_generator import create_deceptive_map, create_easy_map, create_obstacle_map, generate_random_valid_maps

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_grid(axes[0, 0], *create_easy_map(), "Easy")
    draw_grid(axes[0, 1], *create_obstacle_map(), "Obstacle")
    draw_grid(axes[1, 0], *create_deceptive_map(), "Deceptive")
    random_map = generate_random_valid_maps(1, (5, 5), 0.2, 123)[0]
    draw_grid(axes[1, 1], *random_map, "Random-Small")
    fig.suptitle("Evaluation Map Scenes", fontsize=22, fontweight="bold")
    save_fig(fig, "map_scenes.png")


def plot_diffusion_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")
    ax.text(0.03, 0.94, "Conditional Diffusion Architecture", fontsize=24, fontweight="bold", color="#1f2933")

    box(ax, (0.04, 0.55), 0.16, 0.22, r"Noisy sequence $x_t$", ["flattened one-hot", "action trajectory"], "#dceefb", "#2f6b8a")
    box(ax, (0.26, 0.55), 0.16, 0.22, "Timestep Embedding", [r"sin/cos embed", r"for diffusion step $t$"], "#ffe3c2", "#b96a20")
    box(ax, (0.48, 0.55), 0.16, 0.22, "State-Goal Condition", ["[r, c, goal_r, goal_c]", "normalized by grid size"], "#eadbf7", "#7a4da1")
    box(ax, (0.70, 0.55), 0.16, 0.22, "Denoising MLP", ["concat inputs", "predict noise epsilon"], "#d8f0d2", "#3a7a2d")
    box(ax, (0.47, 0.18), 0.22, 0.18, "Reverse Diffusion", ["iteratively denoise", "convert to action sequence"], "#ffffff", "#1f2933")

    arrow_between(ax, 0.20, 0.66, 0.26, 0.66)
    arrow_between(ax, 0.42, 0.66, 0.48, 0.66)
    arrow_between(ax, 0.64, 0.66, 0.70, 0.66)
    arrow_between(ax, 0.78, 0.55, 0.60, 0.36)
    ax.text(0.80, 0.40, r"Predicted noise $\hat{\epsilon}$", fontsize=13, color="#1f2933")
    ax.text(0.50, 0.08, r"Training target: minimize MSE($\hat{\epsilon}, \epsilon$)   |   Sampling: start from Gaussian noise and denoise backward", ha="center", fontsize=14, color="#1f2933")
    save_fig(fig, "diffusion_architecture.png")


def plot_planner_comparison() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    titles = ["Standard Diffusion", "Failure-Memory Diffusion", "Improved Failure-Memory Diffusion"]
    colors = ["#d62728", "#9467bd", "#8c564b"]
    extras = [
        [r"Score = R - $\lambda_D D$", "no memory term"],
        [r"Score = R - $\lambda_D D - \lambda_F F$", "whole failed path memory"],
        [r"Score = R - $\lambda_D D - \lambda_F F_{dead} - \lambda_L Loop$", "dead-end + diversity + adaptive weights"],
    ]
    for ax, title, color, lines in zip(axes, titles, colors, extras):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", color=color)
        box(ax, (0.10, 0.68), 0.80, 0.16, "Generate Candidates", ["diffusion action sequences"], "#f7f7f7", color)
        box(ax, (0.10, 0.42), 0.80, 0.16, "Simulate and Score", lines, "#f7f7f7", color)
        box(ax, (0.10, 0.16), 0.80, 0.16, "Execute", ["take first action of", "best trajectory"], "#f7f7f7", color)
        arrow_between(ax, 0.50, 0.68, 0.50, 0.58)
        arrow_between(ax, 0.50, 0.42, 0.50, 0.32)
    fig.suptitle("Diffusion Planner Comparison", fontsize=22, fontweight="bold")
    save_fig(fig, "planner_comparison.png")


def grouped_bar(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, ylim=None) -> None:
    maps = list(df["Map"].drop_duplicates())
    algos = [a for a in ALGO_ORDER if a in set(df["Algorithm"])]
    x = np.arange(len(maps))
    width = 0.09
    fig, ax = plt.subplots(figsize=(17, 9))
    for i, algo in enumerate(algos):
        subset = df[df["Algorithm"] == algo].set_index("Map").reindex(maps)
        offsets = x + (i - (len(algos) - 1) / 2) * width
        error_col = value_col.replace("Mean", "CI Halfwidth")
        ax.bar(
            offsets,
            subset[value_col].to_numpy(),
            width=width,
            label=algo,
            yerr=subset[error_col].to_numpy() if error_col in subset.columns else None,
            capsize=3 if error_col in subset.columns else 0,
            color=ALGO_COLORS.get(algo),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in maps], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_title(f"{ylabel} by Algorithm and Map", fontsize=20, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.4,
        handletextpad=0.7,
    )
    save_fig(fig, filename)


def plot_success_average_collision_inference() -> None:
    df = load_benchmark_df()
    grouped_bar(df, "Success Rate Mean", "Success Rate", "success_rate_comparison.png", ylim=(0, 1.05))
    grouped_bar(df, "Average Return Mean", "Average Return", "average_return_comparison.png")
    grouped_bar(df, "Collision Rate Mean", "Collision Rate", "collision_rate_comparison.png", ylim=(0, 1.05))
    grouped_bar(df, "Inference Time Mean", "Inference Time (s/action)", "inference_time_comparison.png")


def plot_exploration_success_rate() -> None:
    df = load_exploration_df()
    maps = list(df["Map"].drop_duplicates())
    algos = [a for a in EXPLORATION_ORDER if a in set(df["Algorithm"])]
    x = np.arange(len(maps))
    width = 0.15
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    for i, algo in enumerate(algos):
        subset = df[df["Algorithm"] == algo].set_index("Map").reindex(maps)
        offsets = x + (i - (len(algos) - 1) / 2) * width
        ax.bar(
            offsets,
            subset["Success Rate Mean"].to_numpy(),
            width=width,
            label=algo,
            yerr=subset["Success Rate CI Halfwidth"].to_numpy() if "Success Rate CI Halfwidth" in subset.columns else None,
            capsize=3 if "Success Rate CI Halfwidth" in subset.columns else 0,
            color=EXPLORATION_COLORS[algo],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in maps], fontsize=13)
    ax.set_ylabel("Success Rate", fontsize=17)
    ax.set_ylim(0, 1.05)
    ax.set_title("Exploration Method Success Rate", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.4,
        handletextpad=0.7,
    )
    save_fig(fig, "exploration_success_rate.png")


def plot_lambda_failure_ablation() -> None:
    df = load_lambda_df()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("Success Rate Mean", "Success Rate", "#2ca02c"), ("Collision Rate Mean", "Collision Rate", "#d62728"), ("Repeated Failure Rate Mean", "Repeated Failure Rate", "#9467bd")]
    x = df["lambda_F"].to_numpy()
    for ax, (col, title, color) in zip(axes, metrics):
        error_col = col.replace("Mean", "CI Halfwidth")
        ax.errorbar(x, df[col].to_numpy(), yerr=df[error_col].to_numpy(), marker="o", linewidth=2.5, color=color, capsize=3)
        ax.set_xlabel(r"$\lambda_F$", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, 1.05)
    fig.suptitle(r"Failure-Memory Strength Ablation on $\lambda_F$", fontsize=20, fontweight="bold")
    save_fig(fig, "lambda_failure_ablation.png")


def plot_k_candidate_ablation() -> None:
    fm_df = load_failure_k_df()
    std_df = load_standard_k_df()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = [
        ("Success Rate Mean", "Success Rate", (0, 1.05)),
        ("Average Return Mean", "Average Return", None),
        ("Repeated Failure Rate Mean", "Repeated Failure Rate", (0, 1.05)),
    ]
    for ax, (metric, title, ylim) in zip(axes, panels):
        std_error_col = metric.replace("Mean", "CI Halfwidth")
        ax.errorbar(
            std_df["K"],
            std_df[metric],
            yerr=std_df[std_error_col].to_numpy() if std_error_col in std_df.columns else None,
            marker="o",
            linewidth=2.5,
            color="#d62728",
            label="Standard Diffusion",
            capsize=3,
        )
        if metric in fm_df.columns:
            fm_error_col = metric.replace("Mean", "CI Halfwidth")
            ax.errorbar(
                fm_df["K"],
                fm_df[metric],
                yerr=fm_df[fm_error_col].to_numpy() if fm_error_col in fm_df.columns else None,
                marker="o",
                linewidth=2.5,
                color="#9467bd",
                label="Failure-Memory Diffusion",
                capsize=3,
            )
        ax.set_xlabel("K", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.suptitle("Candidate-Budget Ablation", fontsize=20, fontweight="bold")
    save_fig(fig, "k_candidate_ablation.png")


def main():
    ensure_output_dir()
    plot_final_method_overview()
    plot_map_scenes()
    plot_diffusion_architecture()
    plot_planner_comparison()
    plot_success_average_collision_inference()
    plot_exploration_success_rate()
    plot_lambda_failure_ablation()
    plot_k_candidate_ablation()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
