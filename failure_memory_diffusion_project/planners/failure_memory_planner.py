from __future__ import annotations

import numpy as np

from .diffusion_planner import StandardDiffusionPlanner


class FailureMemoryDiffusionPlanner(StandardDiffusionPlanner):
    def __init__(
        self,
        env,
        diffusion_model,
        horizon: int = 16,
        num_candidates: int = 10,
        lambda_distance: float = 0.1,
        lambda_failure: float = 0.5,
        seed: int | None = None,
    ):
        super().__init__(
            env=env,
            diffusion_model=diffusion_model,
            horizon=horizon,
            num_candidates=num_candidates,
            lambda_distance=lambda_distance,
            seed=seed,
        )
        self.lambda_failure = lambda_failure
        # Failure memory stores where unsuccessful trajectories have already concentrated.
        self.failure_memory = np.zeros_like(self.env.grid, dtype=np.float32)
        self.last_episode_repeated_failure_rate = 0.0

    def reset_failure_memory(self):
        self.failure_memory.fill(0.0)

    def update_failure_memory(self, path):
        for state in path:
            self.failure_memory[state] += 1.0

    def get_repeated_failure_rate(self, path):
        if not path:
            return 0.0
        repeated = sum(1 for state in path if self.failure_memory[state] > 0)
        return repeated / len(path)

    def start_episode(self):
        self.last_episode_repeated_failure_rate = 0.0

    def score_trajectory(self, state, action_sequence):
        rollout = self.simulate_trajectory(state, action_sequence)
        final_state = rollout["final_state"]
        distance = abs(final_state[0] - self.env.goal[0]) + abs(final_state[1] - self.env.goal[1])

        # Normalize the memory map before scoring so the penalty stays comparable across episodes.
        if np.max(self.failure_memory) > 0:
            normalized_memory = self.failure_memory / (np.max(self.failure_memory) + 1e-8)
        else:
            normalized_memory = self.failure_memory

        failure_penalty = float(np.mean([normalized_memory[pos] for pos in rollout["path"]]))

        # Failure-memory scoring: reward minus distance-to-goal and failure-memory penalty.
        score = rollout["total_reward"] - self.lambda_distance * distance - self.lambda_failure * failure_penalty
        return score, rollout

    def end_episode(self, path, success: bool, collision: bool, truncated: bool):
        # Only failed or unfinished episodes reinforce the memory map for future avoidance.
        self.last_episode_repeated_failure_rate = self.get_repeated_failure_rate(path)
        if collision or truncated or (not success):
            self.update_failure_memory(path)
