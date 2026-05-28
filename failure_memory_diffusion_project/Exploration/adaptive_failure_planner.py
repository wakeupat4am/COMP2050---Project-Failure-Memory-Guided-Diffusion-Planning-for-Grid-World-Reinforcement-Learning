from __future__ import annotations

import numpy as np

from planners.failure_memory_planner import FailureMemoryDiffusionPlanner


class AdaptiveFailureMemoryDiffusionPlanner(FailureMemoryDiffusionPlanner):
    def __init__(
        self,
        *args,
        lambda_distance_base: float = 0.1,
        lambda_failure_base: float = 0.5,
        alpha_fail: float = 0.5,
        beta_fail: float = 0.5,
        failure_window: int = 10,
        **kwargs,
    ):
        kwargs["lambda_distance"] = lambda_distance_base
        kwargs["lambda_failure"] = lambda_failure_base
        super().__init__(*args, **kwargs)
        self.lambda_distance_base = lambda_distance_base
        self.lambda_failure_base = lambda_failure_base
        self.alpha_fail = alpha_fail
        self.beta_fail = beta_fail
        self.failure_window = failure_window
        self.recent_failure_history: list[int] = []

    def _current_lambdas(self):
        recent_failures = sum(self.recent_failure_history)
        lambda_distance_current = self.lambda_distance_base / (1.0 + self.alpha_fail * recent_failures)
        lambda_failure_current = self.lambda_failure_base * (1.0 + self.beta_fail * recent_failures)
        return lambda_distance_current, lambda_failure_current

    def score_trajectory(self, state, action_sequence):
        rollout = self.simulate_trajectory(state, action_sequence)
        final_state = rollout["final_state"]
        distance = abs(final_state[0] - self.env.goal[0]) + abs(final_state[1] - self.env.goal[1])
        if np.max(self.failure_memory) > 0:
            normalized_memory = self.failure_memory / (np.max(self.failure_memory) + 1e-8)
        else:
            normalized_memory = self.failure_memory
        failure_penalty = float(np.mean([normalized_memory[pos] for pos in rollout["path"]]))
        lambda_distance_current, lambda_failure_current = self._current_lambdas()
        score = rollout["total_reward"] - lambda_distance_current * distance - lambda_failure_current * failure_penalty
        return score, rollout

    def notify_episode_result(self, success: bool, collision: bool, timeout: bool, path):
        failed = int(collision or timeout or (not success))
        self.recent_failure_history.append(failed)
        self.recent_failure_history = self.recent_failure_history[-self.failure_window :]
        self.last_episode_repeated_failure_rate = self.get_repeated_failure_rate(path)
        if failed:
            self.update_failure_memory(path)

    def end_episode(self, path, success: bool, collision: bool, truncated: bool):
        if not self.recent_failure_history:
            self.notify_episode_result(success=success, collision=collision, timeout=truncated, path=path)
