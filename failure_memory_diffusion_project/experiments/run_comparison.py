from __future__ import annotations

import os

import numpy as np
import pandas as pd

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
from utils import (
    create_deceptive_map,
    create_easy_map,
    create_obstacle_map,
    evaluate_planner,
    generate_random_valid_maps,
    set_global_seed,
    shortest_path_length,
)
from utils.plotting import create_all_plots


def _build_maps():
    maps = {
        "easy": create_easy_map(),
        "obstacle": create_obstacle_map(),
        "deceptive": create_deceptive_map(),
    }
    random_map = generate_random_valid_maps(num_maps=1, size=(5, 5), obstacle_density=0.2, seed=123)[0]
    maps["random_small"] = random_map
    return maps


def _aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = raw_df.groupby(["Algorithm", "Map"])
    for (algorithm, map_name), group in grouped:
        rows.append(
            {
                "Algorithm": algorithm,
                "Map": map_name,
                "Success Rate Mean": group["success"].mean(),
                "Success Rate Std": group["success"].std(ddof=0),
                "Average Return Mean": group["total_return"].mean(),
                "Average Return Std": group["total_return"].std(ddof=0),
                "Path Length Mean": group["path_length"].mean(),
                "Path Length Std": group["path_length"].std(ddof=0),
                "Collision Rate Mean": group["collision"].mean(),
                "Optimality Gap Mean": group["optimality_gap"].mean(skipna=True),
                "Repeated Failure Rate Mean": group["repeated_failure_rate"].mean(),
                "Inference Time Mean": group["inference_time_per_action"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["Map", "Algorithm"]).reset_index(drop=True)


def run_comparison(
    seeds=None,
    num_episodes_eval: int = 10,
    q_learning_episodes: int = 400,
    diffusion_epochs: int = 12,
    diffusion_horizon: int = 12,
    diffusion_candidates: int = 6,
    mcts_simulations: int = 25,
):
    if seeds is None:
        seeds = [0, 1, 2]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
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
                    num_candidates=diffusion_candidates,
                    seed=seed,
                ),
                "Failure-Memory Diffusion": FailureMemoryDiffusionPlanner(
                    env,
                    diffusion_model=diffusion_model,
                    horizon=diffusion_horizon,
                    num_candidates=diffusion_candidates,
                    seed=seed,
                ),
            }

            for algorithm_name, planner in planners.items():
                eval_env = GridWorldEnv(grid=grid, start=start, goal=goal, max_steps=50)
                metrics_df = evaluate_planner(planner, eval_env, num_episodes=num_episodes_eval, seed=seed, bfs_length=bfs_length)
                metrics_df["Algorithm"] = algorithm_name
                metrics_df["Map"] = map_name
                metrics_df["Seed"] = seed
                episode_records.append(metrics_df)

    raw_df = pd.concat(episode_records, ignore_index=True)
    summary_df = _aggregate_results(raw_df)

    csv_path = os.path.join(tables_dir, "efficiency_comparison.csv")
    summary_df.to_csv(csv_path, index=False)

    if not summary_df.empty:
        create_all_plots(summary_df, figures_dir)

    return summary_df, csv_path
