from __future__ import annotations

from planners.failure_memory_planner import FailureMemoryDiffusionPlanner

from .combined_exploration_planner import CombinedExplorationFailureMemoryPlanner


class NoTailMemoryFailureMemoryPlanner(CombinedExplorationFailureMemoryPlanner):
    def update_failure_memory(self, path):
        FailureMemoryDiffusionPlanner.update_failure_memory(self, path)


class NoAdaptiveWeightsFailureMemoryPlanner(CombinedExplorationFailureMemoryPlanner):
    def _current_lambdas(self):
        return self.lambda_distance_base, self.lambda_failure_base


class NoDiversityFailureMemoryPlanner(CombinedExplorationFailureMemoryPlanner):
    def act(self, state):
        raw_candidates = self.diffusion_model.sample(
            self._condition(state),
            num_samples=self.num_candidates * self.raw_multiplier,
            horizon=self.horizon,
        )

        best_action = 0
        best_score = -float("inf")
        for candidate in raw_candidates:
            actions = [int(a) for a in candidate.tolist()]
            if not actions or not self._is_valid_first_action(state, actions[0]):
                continue
            score, _ = self.score_trajectory(state, actions)
            if score > best_score:
                best_score = score
                best_action = actions[0]
        return int(best_action)


class NoLoopPenaltyFailureMemoryPlanner(CombinedExplorationFailureMemoryPlanner):
    def __init__(self, *args, **kwargs):
        kwargs["lambda_loop"] = 0.0
        super().__init__(*args, **kwargs)
