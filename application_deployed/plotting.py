from __future__ import annotations

import os
from pathlib import Path

cache_dir = Path(__file__).resolve().parent / "results" / "figures" / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deployment_maps import HoldoutMap


DISPLAY_NAMES = {
    "Random Policy": "Random Policy",
    "Value Iteration": "Value Iteration",
    "Policy Iteration": "Policy Iteration",
    "Q-learning": "Q-learning",
    "MCTS": "MCTS",
    "Map-Conditioned Standard Diffusion": "MC Standard Diffusion",
    "Map-Conditioned Failure-Memory Diffusion": "MC Failure-Memory Diffusion",
    "Map-Conditioned Improved Failure-Memory Diffusion": "MC Improved FM Diffusion",
}

ALGORITHM_COLORS = {
    "Random Policy": "#7f8c8d",
    "Value Iteration": "#1f77b4",
    "Policy Iteration": "#17becf",
    "Q-learning": "#ff7f0e",
    "MCTS": "#2ca02c",
    "Map-Conditioned Standard Diffusion": "#d62728",
    "Map-Conditioned Failure-Memory Diffusion": "#9467bd",
    "Map-Conditioned Improved Failure-Memory Diffusion": "#8c564b",
}

AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11


def _ordered_algorithms(summary_df: pd.DataFrame) -> list[str]:
    preferred = [
        "Random Policy",
        "Value Iteration",
        "Policy Iteration",
        "Q-learning",
        "MCTS",
        "Map-Conditioned Standard Diffusion",
        "Map-Conditioned Failure-Memory Diffusion",
        "Map-Conditioned Improved Failure-Memory Diffusion",
    ]
    present = set(summary_df["Algorithm"].tolist())
    ordered = [name for name in preferred if name in present]
    return ordered + sorted(present - set(ordered))


def plot_holdout_map(holdout_map: HoldoutMap, output_path: Path) -> None:
    grid = holdout_map.grid
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)

    rows, cols = grid.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for idx, task in enumerate(holdout_map.tasks, start=1):
        sr, sc = task.start
        gr, gc = task.goal
        ax.scatter(sc, sr, marker="o", s=100, c="#1f77b4", edgecolors="white", linewidths=1.0)
        ax.scatter(gc, gr, marker="*", s=180, c="#ff7f0e", edgecolors="white", linewidths=0.8)
        ax.text(sc, sr - 0.22, str(idx), ha="center", va="center", color="white", fontsize=8, weight="bold")
        ax.text(gc, gr + 0.22, str(idx), ha="center", va="center", color="#2d2d2d", fontsize=8, weight="bold")

    ax.set_xlabel("Blue circles: start states | Orange stars: goal states", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_task_difficulty(holdout_map: HoldoutMap, output_path: Path) -> None:
    labels = [task.task_id.replace("task_", "T") for task in holdout_map.tasks]
    values = [task.shortest_path_length for task in holdout_map.tasks]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#5b8c5a", edgecolor="black", linewidth=0.4)
    ax.set_ylabel("BFS Shortest Path Length", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Held-Out Tasks", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(loss_df: pd.DataFrame, output_path: Path) -> None:
    summary = loss_df.groupby("epoch")["loss"].agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "loss_mean", "std": "loss_std"})

    epochs = summary["epoch"].to_numpy()
    loss_mean = summary["loss_mean"].to_numpy()
    loss_std = summary["loss_std"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(epochs, loss_mean, color="#7a4da1", linewidth=2.4, marker="o")
    ax.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color="#cdb7e8", alpha=0.35)
    ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Training Loss (MSE)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_dashboard(summary_df: pd.DataFrame, output_path: Path) -> None:
    algorithms = [
        name
        for name in [
            "Random Policy",
            "Value Iteration",
            "Policy Iteration",
            "Q-learning",
            "MCTS",
            "Map-Conditioned Improved Failure-Memory Diffusion",
        ]
        if name in set(summary_df["Algorithm"].tolist())
    ]
    subset = summary_df.set_index("Algorithm").loc[algorithms]
    x = np.arange(len(algorithms))
    labels = [DISPLAY_NAMES.get(name, name) for name in algorithms]
    colors = [ALGORITHM_COLORS.get(name, "#7f8c8d") for name in algorithms]

    metric_specs = [
        ("Success Rate Mean", "Success Rate CI Halfwidth", "Success Rate", (0.0, 1.05)),
        ("Collision Rate Mean", "Collision Rate CI Halfwidth", "Collision Rate", (0.0, 1.05)),
        ("Average Return Mean", "Average Return CI Halfwidth", "Average Return", None),
        ("Inference Time Mean", "Inference Time CI Halfwidth", "Seconds per Action", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, (value_col, error_col, title, ylim) in zip(axes, metric_specs):
        ax.bar(
            x,
            subset[value_col].to_numpy(),
            yerr=subset[error_col].to_numpy(),
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            capsize=4,
        )
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
