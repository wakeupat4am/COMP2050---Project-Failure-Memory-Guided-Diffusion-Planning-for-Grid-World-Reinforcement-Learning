from __future__ import annotations

import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import GridWorldEnv
from planners import DiffusionActionModel, StandardDiffusionPlanner
from planners.diffusion_planner import build_diffusion_training_data
from utils import DEFAULT_SEEDS, aggregate_raw_to_seed_summary, evaluate_planner, set_global_seed, shortest_path_length
from utils.plotting import compare_best_diffusion_variants, plot_standard_k_ablation

from diffusion_planner.run_lambda_ablation import (
    build_ablation_maps,
    choose_best_lambda_distance,
    run_lambda_ablation,
)
from failure_diffusion_planner.run_k_ablation import run_k_ablation as run_failure_k_ablation


def _aggregate_k_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    _, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["K", "lambda_D"])
    return summary_df.sort_values("K").reset_index(drop=True)


def run_k_ablation(
    k_values=None,
    lambda_value: float | None = None,
    seeds=None,
    num_episodes_eval: int = 10,
    diffusion_epochs: int = 12,
    diffusion_horizon: int = 12,
):
    if k_values is None:
        k_values = [5, 10, 20, 30, 40]
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    tables_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if lambda_value is None:
        lambda_csv = os.path.join(tables_dir, "standard_diffusion_lambda_ablation.csv")
        if os.path.exists(lambda_csv):
            lambda_df = pd.read_csv(lambda_csv)
        else:
            lambda_df, _, _ = run_lambda_ablation(
                seeds=seeds,
                num_episodes_eval=num_episodes_eval,
                diffusion_epochs=diffusion_epochs,
                diffusion_horizon=diffusion_horizon,
            )
        lambda_value = choose_best_lambda_distance(lambda_df)

    episode_records = []
    maps = build_ablation_maps()

    for seed in seeds:
        set_global_seed(seed)
        for map_name, (grid, start, goal) in maps.items():
            env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
            bfs_length = shortest_path_length(grid, start, goal)
            action_vectors, conditions = build_diffusion_training_data(env, horizon=diffusion_horizon)
            diffusion_model = DiffusionActionModel(horizon=diffusion_horizon, diffusion_steps=25)
            diffusion_model.fit(action_vectors, conditions, epochs=diffusion_epochs, batch_size=32)

            for candidate_k in k_values:
                planner = StandardDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=int(candidate_k),
                    lambda_distance=float(lambda_value),
                    seed=seed,
                )
                eval_env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
                metrics_df = evaluate_planner(
                    planner,
                    eval_env,
                    num_episodes=num_episodes_eval,
                    seed=seed,
                    bfs_length=bfs_length,
                )
                metrics_df["K"] = int(candidate_k)
                metrics_df["lambda_D"] = float(lambda_value)
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    per_seed_df, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["K", "lambda_D"])

    raw_csv_path = os.path.join(tables_dir, "standard_diffusion_k_ablation_raw.csv")
    per_seed_csv_path = os.path.join(tables_dir, "standard_diffusion_k_ablation_per_seed.csv")
    summary_csv_path = os.path.join(tables_dir, "standard_diffusion_k_ablation.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    per_seed_df.to_csv(per_seed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    plot_standard_k_ablation(summary_df, figures_dir)
    return summary_df, summary_csv_path, raw_csv_path, float(lambda_value)


def run_best_variant_comparison():
    tables_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures")

    standard_df = pd.read_csv(os.path.join(tables_dir, "standard_diffusion_k_ablation.csv"))
    failure_df = pd.read_csv(os.path.join(tables_dir, "failure_memory_k_ablation.csv"))

    standard_best = standard_df.sort_values(
        by=["Success Rate Mean", "Collision Rate Mean", "Average Return Mean", "Inference Time Mean"],
        ascending=[False, True, False, True],
    ).iloc[0]
    failure_best = failure_df.sort_values(
        by=["Success Rate Mean", "Collision Rate Mean", "Average Return Mean", "Repeated Failure Rate Mean", "Inference Time Mean"],
        ascending=[False, True, False, True, True],
    ).iloc[0]

    comparison_df = pd.DataFrame(
        [
            {
                "Planner": "Standard Diffusion",
                "Best Lambda": float(standard_best["lambda_D"]),
                "Best K": int(standard_best["K"]),
                "Success Rate": float(standard_best["Success Rate Mean"]),
                "Collision Rate": float(standard_best["Collision Rate Mean"]),
                "Repeated Failure Rate": float(standard_best["Repeated Failure Rate Mean"]),
                "Average Return": float(standard_best["Average Return Mean"]),
                "Optimality Gap": float(standard_best["Optimality Gap Mean"]),
                "Inference Time Mean": float(standard_best["Inference Time Mean"]),
            },
            {
                "Planner": "Failure-Memory Diffusion",
                "Best Lambda": float(failure_best["lambda_F"]),
                "Best K": int(failure_best["K"]),
                "Success Rate": float(failure_best["Success Rate Mean"]),
                "Collision Rate": float(failure_best["Collision Rate Mean"]),
                "Repeated Failure Rate": float(failure_best["Repeated Failure Rate Mean"]),
                "Average Return": float(failure_best["Average Return Mean"]),
                "Optimality Gap": float(failure_best["Optimality Gap Mean"]),
                "Inference Time Mean": float(failure_best["Inference Time Mean"]),
            },
        ]
    )

    comparison_csv_path = os.path.join(tables_dir, "best_diffusion_variant_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    compare_best_diffusion_variants(comparison_df, figures_dir)
    return comparison_df, comparison_csv_path


def main():
    print("Running standard diffusion K ablation...")
    df, summary_csv_path, raw_csv_path, best_lambda = run_k_ablation()
    print(f"Using best lambda_D = {best_lambda}")
    print(f"Saved standard diffusion K ablation summary to {summary_csv_path}")
    print(f"Saved standard diffusion K ablation raw episodes to {raw_csv_path}")

    if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "tables", "failure_memory_k_ablation.csv")):
        run_failure_k_ablation()

    comparison_df, comparison_csv_path = run_best_variant_comparison()
    print(f"Saved best diffusion variant comparison to {comparison_csv_path}")
    print("\nStandard Diffusion K Ablation:")
    print(df)
    print("\nBest Standard vs Failure-Memory Diffusion:")
    print(comparison_df)


if __name__ == "__main__":
    main()
