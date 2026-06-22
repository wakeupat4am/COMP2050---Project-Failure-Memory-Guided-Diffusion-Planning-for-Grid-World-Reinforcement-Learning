from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from envs import GridWorldEnv
from planners import (
    MonteCarloTreeSearchPlanner,
    PolicyIterationPlanner,
    RandomPolicyPlanner,
    TabularQLearningPlanner,
    ValueIterationPlanner,
)
from utils import DEFAULT_SEEDS, aggregate_raw_to_seed_summary, evaluate_planner, set_global_seed

from deployment_maps import (
    generate_holdout_map,
    generate_training_maps,
    task_metadata_rows,
    training_goal_metadata_rows,
    training_map_metadata_rows,
)
from deployment_maps import HoldoutMap
from deployment_planners import (
    MapConditionedFailureMemoryDiffusionPlanner,
    MapConditionedImprovedFailureMemoryPlanner,
    MapConditionedStandardDiffusionPlanner,
)
from map_conditioned_diffusion import build_map_conditioned_training_data, create_map_conditioned_model
from plotting import plot_efficiency_dashboard, plot_holdout_map, plot_task_difficulty, plot_training_loss


APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-map deployment experiment for the improved failure-memory diffusion planner.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--num-training-maps", type=int, default=18)
    parser.add_argument("--goals-per-map", type=int, default=3)
    parser.add_argument("--num-holdout-tasks", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--diffusion-epochs", type=int, default=18)
    parser.add_argument("--diffusion-horizon", type=int, default=16)
    parser.add_argument("--diffusion-candidates", type=int, default=12)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=128)
    parser.add_argument("--mcts-simulations", type=int, default=35)
    parser.add_argument("--q-learning-episodes", type=int, default=400)
    parser.add_argument("--map-seed", type=int, default=31415)
    parser.add_argument("--holdout-seed", type=int, default=27182)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS[:3])
    return parser.parse_args()


def _prepare_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_metadata(training_maps, holdout: HoldoutMap, config: dict[str, object]) -> None:
    pd.DataFrame(training_map_metadata_rows(training_maps)).to_csv(TABLES_DIR / "training_maps.csv", index=False)
    pd.DataFrame(training_goal_metadata_rows(training_maps)).to_csv(TABLES_DIR / "training_goals.csv", index=False)
    pd.DataFrame(task_metadata_rows(holdout)).to_csv(TABLES_DIR / "holdout_tasks.csv", index=False)
    with (TABLES_DIR / "deployment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _build_planners(
    env: GridWorldEnv,
    diffusion_model,
    diffusion_horizon: int,
    diffusion_candidates: int,
    mcts_simulations: int,
    q_learning_episodes: int,
    seed: int,
) -> dict[str, object]:
    vi_planner = ValueIterationPlanner(env)
    vi_planner.train()
    pi_planner = PolicyIterationPlanner(env)
    pi_planner.train()
    q_planner = TabularQLearningPlanner(env, episodes=q_learning_episodes, seed=seed)
    q_planner.train()

    return {
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
        "Map-Conditioned Standard Diffusion": MapConditionedStandardDiffusionPlanner(
            env,
            diffusion_model=diffusion_model,
            horizon=diffusion_horizon,
            num_candidates=diffusion_candidates,
            seed=seed,
        ),
        "Map-Conditioned Failure-Memory Diffusion": MapConditionedFailureMemoryDiffusionPlanner(
            env,
            diffusion_model=diffusion_model,
            horizon=diffusion_horizon,
            num_candidates=diffusion_candidates,
            seed=seed,
        ),
        "Map-Conditioned Improved Failure-Memory Diffusion": MapConditionedImprovedFailureMemoryPlanner(
            env,
            diffusion_model=diffusion_model,
            horizon=diffusion_horizon,
            num_candidates=diffusion_candidates,
            seed=seed,
        ),
    }


def _write_results_summary(
    summary_df: pd.DataFrame,
    training_map_count: int,
    training_sample_count: int,
    model_seed_count: int,
    holdout: HoldoutMap,
    output_path: Path,
) -> None:
    lines = [
        "# Deployment Experiment Summary",
        "",
        f"- Training maps: `{training_map_count}`",
        f"- Training samples: `{training_sample_count}`",
        f"- Model seeds evaluated: `{model_seed_count}`",
        f"- Held-out map free cells: `{holdout.free_cell_count}`",
        f"- Held-out tasks: `{len(holdout.tasks)}`",
        "",
        "## Held-Out Efficiency",
        "",
        "| Algorithm | Success Rate | Collision Rate | Average Return | Inference Time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Algorithm"],
                    f"{row['Success Rate Mean']:.3f} +- {row['Success Rate Std']:.3f}",
                    f"{row['Collision Rate Mean']:.3f} +- {row['Collision Rate Std']:.3f}",
                    f"{row['Average Return Mean']:.3f} +- {row['Average Return Std']:.3f}",
                    f"{row['Inference Time Mean'] * 1000:.3f} ms +- {row['Inference Time Std'] * 1000:.3f} ms",
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _prepare_dirs()

    config = {
        "grid_size": [args.rows, args.cols],
        "num_training_maps": args.num_training_maps,
        "goals_per_map": args.goals_per_map,
        "num_holdout_tasks": args.num_holdout_tasks,
        "eval_episodes_per_task": args.eval_episodes,
        "diffusion_epochs": args.diffusion_epochs,
        "diffusion_horizon": args.diffusion_horizon,
        "diffusion_candidates": args.diffusion_candidates,
        "diffusion_hidden_dim": args.diffusion_hidden_dim,
        "mcts_simulations": args.mcts_simulations,
        "q_learning_episodes": args.q_learning_episodes,
        "map_seed": args.map_seed,
        "holdout_seed": args.holdout_seed,
        "model_seeds": args.seeds,
        "train_obstacle_density_range": [0.15, 0.32],
        "holdout_obstacle_density_range": [0.15, 0.32],
    }

    training_maps = generate_training_maps(
        num_maps=args.num_training_maps,
        size=(args.rows, args.cols),
        obstacle_density_range=(0.15, 0.32),
        goals_per_map=args.goals_per_map,
        seed=args.map_seed,
    )
    holdout = generate_holdout_map(
        size=(args.rows, args.cols),
        obstacle_density_range=(0.15, 0.32),
        num_tasks=args.num_holdout_tasks,
        seed=args.holdout_seed,
    )
    _save_metadata(training_maps, holdout, config)

    action_vectors, conditions, training_sample_df = build_map_conditioned_training_data(
        training_maps=training_maps,
        horizon=args.diffusion_horizon,
    )
    training_sample_df.to_csv(TABLES_DIR / "training_samples.csv", index=False)

    raw_records: list[pd.DataFrame] = []
    loss_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        set_global_seed(seed)
        diffusion_model = create_map_conditioned_model(
            grid_shape=(args.rows, args.cols),
            horizon=args.diffusion_horizon,
            hidden_dim=args.diffusion_hidden_dim,
            diffusion_steps=25,
            learning_rate=1e-3,
        )
        loss_history = diffusion_model.fit(
            action_vectors=action_vectors,
            conditions=conditions,
            epochs=args.diffusion_epochs,
            batch_size=32,
        )
        for epoch_idx, loss_value in enumerate(loss_history, start=1):
            loss_rows.append({"seed": seed, "epoch": epoch_idx, "loss": loss_value})

        for task_idx, task in enumerate(holdout.tasks):
            base_env = GridWorldEnv(grid=holdout.grid, start=task.start, goal=task.goal, max_steps=50)
            planners = _build_planners(
                env=base_env,
                diffusion_model=diffusion_model,
                diffusion_horizon=args.diffusion_horizon,
                diffusion_candidates=args.diffusion_candidates,
                mcts_simulations=args.mcts_simulations,
                q_learning_episodes=args.q_learning_episodes,
                seed=seed * 100 + task_idx,
            )

            for algorithm_name, planner in planners.items():
                eval_env = GridWorldEnv(grid=holdout.grid, start=task.start, goal=task.goal, max_steps=50)
                metrics_df = evaluate_planner(
                    planner=planner,
                    env=eval_env,
                    num_episodes=args.eval_episodes,
                    seed=seed * 1000 + task_idx,
                    bfs_length=task.shortest_path_length,
                )
                metrics_df["Algorithm"] = algorithm_name
                metrics_df["Seed"] = seed
                metrics_df["Task"] = task.task_id
                raw_records.append(metrics_df)

    raw_df = pd.concat(raw_records, ignore_index=True)
    per_seed_df, summary_df = aggregate_raw_to_seed_summary(raw_df=raw_df, group_cols=["Algorithm"])
    summary_df = summary_df.sort_values("Algorithm").reset_index(drop=True)

    raw_df.to_csv(TABLES_DIR / "deployment_raw.csv", index=False)
    per_seed_df.to_csv(TABLES_DIR / "deployment_per_seed.csv", index=False)
    summary_df.to_csv(TABLES_DIR / "deployment_summary.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(TABLES_DIR / "training_loss.csv", index=False)

    plot_holdout_map(holdout, FIGURES_DIR / "heldout_map_overview.png")
    plot_task_difficulty(holdout, FIGURES_DIR / "heldout_task_difficulty.png")
    plot_training_loss(pd.DataFrame(loss_rows), FIGURES_DIR / "training_loss_curve.png")
    plot_efficiency_dashboard(summary_df, FIGURES_DIR / "deployment_efficiency_dashboard.png")

    _write_results_summary(
        summary_df=summary_df,
        training_map_count=len(training_maps),
        training_sample_count=len(training_sample_df),
        model_seed_count=len(args.seeds),
        holdout=holdout,
        output_path=RESULTS_DIR / "RESULTS.md",
    )

    print(f"Saved deployment results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
