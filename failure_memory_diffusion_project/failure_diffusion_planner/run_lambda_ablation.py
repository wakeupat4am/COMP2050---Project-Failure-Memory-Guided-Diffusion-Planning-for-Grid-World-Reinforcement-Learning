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
from utils import (
    create_deceptive_map,
    create_easy_map,
    create_obstacle_map,
    evaluate_planner,
    generate_random_valid_maps,
    set_global_seed,
    shortest_path_length,
)
from utils.plotting import plot_lambda_failure_ablation


def build_ablation_maps():
    maps = {
        "easy": create_easy_map(),
        "obstacle": create_obstacle_map(),
        "deceptive": create_deceptive_map(),
    }
    maps["random_small"] = generate_random_valid_maps(
        num_maps=1,
        size=(5, 5),
        obstacle_density=0.2,
        seed=123,
    )[0]
    return maps


def _aggregate_lambda_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = raw_df.groupby("lambda_F")
    for lambda_value, group in grouped:
        rows.append(
            {
                "lambda_F": float(lambda_value),
                "Success Rate": group["success"].mean(),
                "Collision Rate": group["collision"].mean(),
                "Repeated Failure Rate": group["repeated_failure_rate"].mean(),
                "Average Return": group["total_return"].mean(),
                "Path Length": group["path_length"].mean(),
                "Optimality Gap": group["optimality_gap"].mean(skipna=True),
                "Inference Time Mean": group["inference_time_per_action"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("lambda_F").reset_index(drop=True)


def choose_best_lambda(summary_df: pd.DataFrame) -> float:
    ranked = summary_df.sort_values(
        by=[
            "Success Rate",
            "Collision Rate",
            "Average Return",
            "Repeated Failure Rate",
            "Inference Time Mean",
        ],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    return float(ranked.loc[0, "lambda_F"])


def run_lambda_ablation(
    lambda_values=None,
    seeds=None,
    num_episodes_eval: int = 10,
    diffusion_epochs: int = 12,
    diffusion_horizon: int = 12,
    diffusion_candidates: int = 6,
):
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    if seeds is None:
        seeds = [0, 1, 2]

    base_dir = PROJECT_ROOT
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

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

            for lambda_value in lambda_values:
                planner = FailureMemoryDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=diffusion_candidates,
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
                metrics_df["lambda_F"] = float(lambda_value)
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    summary_df = _aggregate_lambda_results(raw_df)

    raw_csv_path = os.path.join(tables_dir, "failure_memory_lambda_ablation_raw.csv")
    summary_csv_path = os.path.join(tables_dir, "failure_memory_lambda_ablation.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    plot_lambda_failure_ablation(summary_df, figures_dir)
    return summary_df, summary_csv_path, raw_csv_path


def main():
    print("Running failure-memory lambda ablation...")
    df, summary_csv_path, raw_csv_path = run_lambda_ablation()
    best_lambda = choose_best_lambda(df)
    print(f"Saved lambda ablation summary to {summary_csv_path}")
    print(f"Saved lambda ablation raw episodes to {raw_csv_path}")
    print(f"Chosen best lambda_F = {best_lambda}")
    print("\nFailure-Memory Lambda Ablation:")
    print(df)


if __name__ == "__main__":
    main()
