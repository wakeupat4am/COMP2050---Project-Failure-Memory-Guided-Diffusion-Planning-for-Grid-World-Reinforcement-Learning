from __future__ import annotations

import os
import sys

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import GridWorldEnv
from planners import (
    DiffusionActionModel,
    FailureMemoryDiffusionPlanner,
    MonteCarloTreeSearchPlanner,
    PolicyIterationPlanner,
    RandomPolicyPlanner,
    StandardDiffusionPlanner,
    TabularQLearningPlanner,
    ValueIterationPlanner,
)
from planners.diffusion_planner import build_diffusion_training_data
from utils import aggregate_raw_to_seed_summary, evaluate_planner, set_global_seed, shortest_path_length
from utils.plotting import compare_best_diffusion_variants, create_all_plots

from experiments.run_comparison import _aggregate_results, _build_maps


def run_after_fix_comparison(
    seeds=None,
    num_episodes_eval: int = 10,
    q_learning_episodes: int = 400,
    diffusion_epochs: int = 12,
    diffusion_horizon: int = 12,
    mcts_simulations: int = 25,
    standard_lambda_distance: float = 0.1,
    standard_k: int = 40,
    failure_lambda: float = 0.5,
    failure_k: int = 40,
):
    if seeds is None:
        seeds = [0, 1, 2]

    output_dir = os.path.join(PROJECT_ROOT, "After-fix results")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    maps = _build_maps()
    episode_records = []

    for seed in seeds:
        set_global_seed(seed)
        for map_name, (grid, start, goal) in maps.items():
            env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
            bfs_length = shortest_path_length(grid, start, goal)

            vi_planner = ValueIterationPlanner(env)
            vi_planner.train()

            pi_planner = PolicyIterationPlanner(env)
            pi_planner.train()

            q_planner = TabularQLearningPlanner(env, episodes=q_learning_episodes, seed=seed)
            q_planner.train()

            action_vectors, conditions = build_diffusion_training_data(env, horizon=diffusion_horizon)
            diffusion_model = DiffusionActionModel(horizon=diffusion_horizon, diffusion_steps=25)
            diffusion_model.fit(action_vectors, conditions, epochs=diffusion_epochs, batch_size=32)

            planners = {
                "Random Policy": RandomPolicyPlanner(seed=seed),
                "Value Iteration": vi_planner,
                "Policy Iteration": pi_planner,
                "Q-learning": q_planner,
                "MCTS": MonteCarloTreeSearchPlanner(
                    env,
                    simulations=mcts_simulations,
                    heuristic_rollout=True,
                    seed=seed,
                ),
                "Standard Diffusion": StandardDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=standard_k,
                    lambda_distance=standard_lambda_distance,
                    seed=seed,
                ),
                "Failure-Memory Diffusion": FailureMemoryDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=failure_k,
                    lambda_failure=failure_lambda,
                    seed=seed,
                ),
            }

            for algorithm_name, planner in planners.items():
                eval_env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
                metrics_df = evaluate_planner(
                    planner,
                    eval_env,
                    num_episodes=num_episodes_eval,
                    seed=seed,
                    bfs_length=bfs_length,
                )
                metrics_df["Algorithm"] = algorithm_name
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    per_seed_df, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["Algorithm", "Map"])

    summary_csv_path = os.path.join(tables_dir, "after_fix_efficiency_comparison.csv")
    per_seed_csv_path = os.path.join(tables_dir, "after_fix_efficiency_comparison_per_seed.csv")
    raw_csv_path = os.path.join(tables_dir, "after_fix_efficiency_comparison_raw.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    per_seed_df.to_csv(per_seed_csv_path, index=False)
    raw_df.to_csv(raw_csv_path, index=False)

    create_all_plots(summary_df, figures_dir)

    diffusion_comparison_df = summary_df[
        summary_df["Algorithm"].isin(["Standard Diffusion", "Failure-Memory Diffusion"])
    ][
        [
            "Algorithm",
            "Success Rate Mean",
            "Collision Rate Mean",
            "Repeated Failure Rate Mean",
            "Average Return Mean",
            "Optimality Gap Mean",
            "Inference Time Mean",
        ]
    ].copy()
    diffusion_comparison_df = diffusion_comparison_df.groupby("Algorithm", as_index=False).mean(numeric_only=True)
    diffusion_comparison_df = diffusion_comparison_df.rename(
        columns={
            "Algorithm": "Planner",
            "Success Rate Mean": "Success Rate",
            "Collision Rate Mean": "Collision Rate",
            "Repeated Failure Rate Mean": "Repeated Failure Rate",
            "Average Return Mean": "Average Return",
            "Optimality Gap Mean": "Optimality Gap",
        }
    )
    lambda_by_planner = {
        "Standard Diffusion": float(standard_lambda_distance),
        "Failure-Memory Diffusion": float(failure_lambda),
    }
    k_by_planner = {
        "Standard Diffusion": int(standard_k),
        "Failure-Memory Diffusion": int(failure_k),
    }
    diffusion_comparison_df.insert(1, "Best Lambda", diffusion_comparison_df["Planner"].map(lambda_by_planner))
    diffusion_comparison_df.insert(2, "Best K", diffusion_comparison_df["Planner"].map(k_by_planner))

    diffusion_compare_csv_path = os.path.join(tables_dir, "after_fix_best_diffusion_comparison.csv")
    diffusion_comparison_df.to_csv(diffusion_compare_csv_path, index=False)
    compare_best_diffusion_variants(diffusion_comparison_df, figures_dir)

    return {
        "summary_df": summary_df,
        "raw_df": raw_df,
        "summary_csv_path": summary_csv_path,
        "raw_csv_path": raw_csv_path,
        "diffusion_compare_csv_path": diffusion_compare_csv_path,
        "output_dir": output_dir,
    }


def main():
    print("Running after-fix comparison with tuned diffusion settings...")
    outputs = run_after_fix_comparison()
    print(f"Saved after-fix summary to {outputs['summary_csv_path']}")
    print(f"Saved after-fix raw episodes to {outputs['raw_csv_path']}")
    print(f"Saved tuned diffusion comparison to {outputs['diffusion_compare_csv_path']}")
    print("\nAfter-fix Efficiency Comparison:")
    print(outputs["summary_df"])


if __name__ == "__main__":
    main()
