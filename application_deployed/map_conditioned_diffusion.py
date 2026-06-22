from __future__ import annotations

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from planners.diffusion_model import DiffusionActionModel
from utils import bfs_shortest_path, path_to_actions

from deployment_maps import TrainingMap, free_positions


def encode_map_condition(grid: np.ndarray, state: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    rows, cols = grid.shape
    grid_features = grid.astype(np.float32).reshape(-1)
    position_features = np.array(
        [
            state[0] / max(rows - 1, 1),
            state[1] / max(cols - 1, 1),
            goal[0] / max(rows - 1, 1),
            goal[1] / max(cols - 1, 1),
        ],
        dtype=np.float32,
    )
    return np.concatenate([grid_features, position_features], axis=0)


def pad_actions(actions: list[int], horizon: int) -> list[int]:
    if not actions:
        actions = [3]
    if len(actions) >= horizon:
        return actions[:horizon]
    return actions + [actions[-1]] * (horizon - len(actions))


def action_sequence_to_one_hot(actions: list[int], horizon: int, action_dim: int = 4) -> np.ndarray:
    padded = pad_actions(actions, horizon)
    one_hot = np.zeros((horizon, action_dim), dtype=np.float32)
    for idx, action in enumerate(padded):
        one_hot[idx, int(action)] = 1.0
    return one_hot.reshape(-1)


def build_map_conditioned_training_data(
    training_maps: list[TrainingMap],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    action_vectors: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    for train_map in training_maps:
        for goal_idx, goal in enumerate(train_map.training_goals):
            for start_state in free_positions(train_map.grid):
                if start_state == goal:
                    continue
                path = bfs_shortest_path(train_map.grid, start_state, goal)
                if path is None:
                    continue
                actions = path_to_actions(path)
                action_vectors.append(action_sequence_to_one_hot(actions, horizon))
                conditions.append(encode_map_condition(train_map.grid, start_state, goal))
                metadata_rows.append(
                    {
                        "map_id": train_map.map_id,
                        "goal_id": f"{train_map.map_id}_goal_{goal_idx:02d}",
                        "start_row": start_state[0],
                        "start_col": start_state[1],
                        "goal_row": goal[0],
                        "goal_col": goal[1],
                        "path_length": len(path) - 1,
                    }
                )

    return (
        np.array(action_vectors, dtype=np.float32),
        np.array(conditions, dtype=np.float32),
        pd.DataFrame(metadata_rows),
    )


def create_map_conditioned_model(
    grid_shape: tuple[int, int],
    horizon: int,
    hidden_dim: int = 128,
    diffusion_steps: int = 25,
    learning_rate: float = 1e-3,
) -> DiffusionActionModel:
    condition_dim = int(np.prod(grid_shape) + 4)
    return DiffusionActionModel(
        horizon=horizon,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        diffusion_steps=diffusion_steps,
        learning_rate=learning_rate,
    )
