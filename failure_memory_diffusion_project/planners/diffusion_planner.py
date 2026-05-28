from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from utils.bfs import bfs_shortest_path, path_to_actions


def normalize_condition(state, goal, grid_shape):
    rows, cols = grid_shape
    return np.array(
        [
            state[0] / max(rows - 1, 1),
            state[1] / max(cols - 1, 1),
            goal[0] / max(rows - 1, 1),
            goal[1] / max(cols - 1, 1),
        ],
        dtype=np.float32,
    )


def pad_actions(actions: Sequence[int], horizon: int) -> List[int]:
    actions = list(actions)
    if not actions:
        actions = [3]
    if len(actions) >= horizon:
        return actions[:horizon]
    return actions + [actions[-1]] * (horizon - len(actions))


def action_sequence_to_one_hot(actions: Sequence[int], horizon: int, action_dim: int = 4) -> np.ndarray:
    padded = pad_actions(actions, horizon)
    one_hot = np.zeros((horizon, action_dim), dtype=np.float32)
    for idx, action in enumerate(padded):
        one_hot[idx, int(action)] = 1.0
    return one_hot.reshape(-1)


def build_diffusion_training_data(env, horizon: int):
    free_states = env.get_all_states()
    action_vectors = []
    conditions = []

    for start_state in free_states:
        if start_state == env.goal:
            continue
        path = bfs_shortest_path(env.grid, start_state, env.goal)
        if path is None:
            continue
        actions = path_to_actions(path)
        action_vectors.append(action_sequence_to_one_hot(actions, horizon))
        conditions.append(normalize_condition(start_state, env.goal, env.grid.shape))

    return np.array(action_vectors, dtype=np.float32), np.array(conditions, dtype=np.float32)


class StandardDiffusionPlanner:
    def __init__(
        self,
        env,
        diffusion_model,
        horizon: int = 16,
        num_candidates: int = 10,
        lambda_distance: float = 0.1,
        seed: int | None = None,
    ):
        self.env = env.copy()
        self.diffusion_model = diffusion_model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.lambda_distance = lambda_distance
        self.rng = np.random.default_rng(seed)
        self.last_episode_repeated_failure_rate = 0.0

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _condition(self, state):
        return normalize_condition(state, self.env.goal, self.env.grid.shape)

    def simulate_trajectory(self, state, action_sequence):
        return self.env.simulate_action_sequence(state, action_sequence)

    def score_trajectory(self, state, action_sequence):
        rollout = self.simulate_trajectory(state, action_sequence)
        final_state = rollout["final_state"]
        distance = abs(final_state[0] - self.env.goal[0]) + abs(final_state[1] - self.env.goal[1])
        # Standard diffusion scoring: reward minus distance-to-goal penalty.
        score = rollout["total_reward"] - self.lambda_distance * distance
        return score, rollout

    def act(self, state):
        condition = self._condition(state)
        candidates = self.diffusion_model.sample(condition, num_samples=self.num_candidates, horizon=self.horizon)

        best_action = 0
        best_score = -float("inf")
        for candidate in candidates:
            actions = [int(a) for a in candidate.tolist()]
            score, _ = self.score_trajectory(state, actions)
            if score > best_score:
                best_score = score
                best_action = actions[0]
        return best_action
