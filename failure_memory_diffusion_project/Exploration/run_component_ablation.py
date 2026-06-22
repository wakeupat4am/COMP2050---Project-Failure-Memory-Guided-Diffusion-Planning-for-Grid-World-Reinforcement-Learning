# Ablation-study design and component-removal setup developed by the author.
# Implementation drafting/editing assisted by AI; no copied external code.
from __future__ import annotations

import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import GridWorldEnv
from planners import DiffusionActionModel
from planners.diffusion_planner import build_diffusion_training_data
from utils import DEFAULT_SEEDS, aggregate_raw_to_seed_summary, evaluate_planner, set_global_seed, shortest_path_length
from utils.plotting import create_component_ablation_plots

from Exploration import (
    CombinedExplorationFailureMemoryPlanner,
    NoAdaptiveWeightsFailureMemoryPlanner,
    NoDiversityFailureMemoryPlanner,
    NoLoopPenaltyFailureMemoryPlanner,
    NoTailMemoryFailureMemoryPlanner,
)
from experiments.run_comparison import _build_maps


def run_component_ablation(
    seeds=None,
    num_episodes_eval: int = 10,
    diffusion_epochs: int = 12,
    diffusion_horizon: int = 12,
    tuned_lambda_failure: float = 0.5,
    tuned_k: int = 40,
):
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    output_dir = os.path.join(PROJECT_ROOT, "Exploration", "component_ablation_results")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    episode_records = []
    maps = _build_maps()
    for seed in seeds:
        set_global_seed(seed)
        for map_name, (grid, start, goal) in maps.items():
            env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
            bfs_length = shortest_path_length(grid, start, goal)
            action_vectors, conditions = build_diffusion_training_data(env, horizon=diffusion_horizon)
            diffusion_model = DiffusionActionModel(horizon=diffusion_horizon, diffusion_steps=25)
            diffusion_model.fit(action_vectors, conditions, epochs=diffusion_epochs, batch_size=32)

            # Remove one mechanism at a time to show which part of the full method drives the gains.
            planners = {
                "Full Method": CombinedExplorationFailureMemoryPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=tuned_k,
                    lambda_failure_base=tuned_lambda_failure,
                    lambda_distance_base=0.1,
                    seed=seed,
                ),
                "Without Tail Memory": NoTailMemoryFailureMemoryPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=tuned_k,
                    lambda_failure_base=tuned_lambda_failure,
                    lambda_distance_base=0.1,
                    seed=seed,
                ),
                "Without Diversity": NoDiversityFailureMemoryPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=tuned_k,
                    lambda_failure_base=tuned_lambda_failure,
                    lambda_distance_base=0.1,
                    seed=seed,
                ),
                "Without Adaptive Weights": NoAdaptiveWeightsFailureMemoryPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=tuned_k,
                    lambda_failure_base=tuned_lambda_failure,
                    lambda_distance_base=0.1,
                    seed=seed,
                ),
                "Without Loop Penalty": NoLoopPenaltyFailureMemoryPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=tuned_k,
                    lambda_failure_base=tuned_lambda_failure,
                    lambda_distance_base=0.1,
                    seed=seed,
                ),
            }

            for variant_name, planner in planners.items():
                # Evaluate every ablation variant under the same seed and map before aggregation.
                eval_env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
                metrics_df = evaluate_planner(
                    planner,
                    eval_env,
                    num_episodes=num_episodes_eval,
                    seed=seed,
                    bfs_length=bfs_length,
                )
                metrics_df["Algorithm"] = variant_name
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    # Persist all aggregation levels so the report can inspect both overall trends and per-seed variance.
    per_seed_df, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["Algorithm", "Map"])

    raw_csv_path = os.path.join(tables_dir, "component_ablation_raw.csv")
    per_seed_csv_path = os.path.join(tables_dir, "component_ablation_per_seed.csv")
    summary_csv_path = os.path.join(tables_dir, "component_ablation.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    per_seed_df.to_csv(per_seed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    create_component_ablation_plots(summary_df, figures_dir)
    return summary_df, summary_csv_path, per_seed_csv_path, raw_csv_path


def main():
    print("Running remove-one component ablation...")
    df, summary_csv_path, per_seed_csv_path, raw_csv_path = run_component_ablation()
    print(f"Saved component ablation summary to {summary_csv_path}")
    print(f"Saved component ablation per-seed table to {per_seed_csv_path}")
    print(f"Saved component ablation raw episodes to {raw_csv_path}")
    print("\nRemove-One Component Ablation:")
    print(df)


if __name__ == "__main__":
    main()
