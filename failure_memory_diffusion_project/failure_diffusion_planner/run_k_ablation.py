from __future__ import annotations

import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import GridWorldEnv
from planners import DiffusionActionModel, FailureMemoryDiffusionPlanner
from planners.diffusion_planner import build_diffusion_training_data
from utils import DEFAULT_SEEDS, aggregate_raw_to_seed_summary, evaluate_planner, set_global_seed, shortest_path_length
from utils.plotting import plot_k_ablation

from failure_diffusion_planner.run_lambda_ablation import (
    build_ablation_maps,
    choose_best_lambda,
    run_lambda_ablation,
)


def _aggregate_k_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    _, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["K", "lambda_F"])
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

    base_dir = PROJECT_ROOT
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if lambda_value is None:
        lambda_csv = os.path.join(tables_dir, "failure_memory_lambda_ablation.csv")
        if os.path.exists(lambda_csv):
            lambda_df = pd.read_csv(lambda_csv)
        else:
            lambda_df, _, _ = run_lambda_ablation(
                seeds=seeds,
                num_episodes_eval=num_episodes_eval,
                diffusion_epochs=diffusion_epochs,
                diffusion_horizon=diffusion_horizon,
            )
        lambda_value = choose_best_lambda(lambda_df)

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
                planner = FailureMemoryDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=int(candidate_k),
                    lambda_failure=float(lambda_value),
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
                metrics_df["lambda_F"] = float(lambda_value)
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    per_seed_df, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["K", "lambda_F"])

    raw_csv_path = os.path.join(tables_dir, "failure_memory_k_ablation_raw.csv")
    per_seed_csv_path = os.path.join(tables_dir, "failure_memory_k_ablation_per_seed.csv")
    summary_csv_path = os.path.join(tables_dir, "failure_memory_k_ablation.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    per_seed_df.to_csv(per_seed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    plot_k_ablation(summary_df, figures_dir)
    return summary_df, summary_csv_path, raw_csv_path, float(lambda_value)


def main():
    print("Running failure-memory K ablation...")
    df, summary_csv_path, raw_csv_path, best_lambda = run_k_ablation()
    print(f"Using best lambda_F = {best_lambda}")
    print(f"Saved K ablation summary to {summary_csv_path}")
    print(f"Saved K ablation raw episodes to {raw_csv_path}")
    print("\nFailure-Memory K Ablation:")
    print(df)


if __name__ == "__main__":
    main()
