from __future__ import annotations

from collections import defaultdict

import numpy as np

from .adaptive_failure_planner import AdaptiveFailureMemoryDiffusionPlanner


class CombinedExplorationFailureMemoryPlanner(AdaptiveFailureMemoryDiffusionPlanner):
    def __init__(
        self,
        *args,
        tail_k: int = 5,
        raw_multiplier: int = 3,
        lambda_loop: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tail_k = tail_k
        self.raw_multiplier = raw_multiplier
        self.lambda_loop = lambda_loop

    def update_failure_memory(self, path):
        # Tail-only updates emphasize the terminal part of failed trajectories where traps usually manifest.
        tail = path[-self.tail_k :] if path else []
        for state in tail:
            self.failure_memory[state] += 1.0

    def _is_valid_first_action(self, state, action: int) -> bool:
        _, _, _, _, info = self.env.get_transition(state, action)
        return not bool(info["collision"])

    def _loop_penalty(self, path):
        repeated_states = len(path) - len(set(path))
        return repeated_states / max(len(path), 1)

    def score_trajectory(self, state, action_sequence):
        # Combine reward, goal distance, failure memory, and loop aversion into one rollout score.
        rollout = self.simulate_trajectory(state, action_sequence)
        final_state = rollout["final_state"]
        distance = abs(final_state[0] - self.env.goal[0]) + abs(final_state[1] - self.env.goal[1])
        if np.max(self.failure_memory) > 0:
            normalized_memory = self.failure_memory / (np.max(self.failure_memory) + 1e-8)
        else:
            normalized_memory = self.failure_memory
        failure_penalty = float(np.mean([normalized_memory[pos] for pos in rollout["path"]]))
        lambda_distance_current, lambda_failure_current = self._current_lambdas()
        loop_penalty = self._loop_penalty(rollout["path"])
        score = (
            rollout["total_reward"]
            - lambda_distance_current * distance
            - lambda_failure_current * failure_penalty
            - self.lambda_loop * loop_penalty
        )
        return score, rollout

    def act(self, state):
        # Oversample, score, and then keep a diverse shortlist across first actions before choosing.
        raw_candidates = self.diffusion_model.sample(
            self._condition(state),
            num_samples=self.num_candidates * self.raw_multiplier,
            horizon=self.horizon,
        )

        scored = []
        grouped = defaultdict(list)
        for candidate in raw_candidates:
            actions = [int(a) for a in candidate.tolist()]
            if not actions or not self._is_valid_first_action(state, actions[0]):
                continue
            score, rollout = self.score_trajectory(state, actions)
            item = {"actions": actions, "score": score, "rollout": rollout}
            scored.append(item)
            grouped[actions[0]].append(item)

        if not scored:
            return 0

        selected = []
        used_ids = set()
        # Reserve one high-scoring candidate per first action to avoid collapsing onto a single mode.
        for action, items in grouped.items():
            best = max(items, key=lambda entry: entry["score"])
            selected.append(best)
            used_ids.add(id(best))

        remaining = sorted(
            [item for item in scored if id(item) not in used_ids],
            key=lambda entry: entry["score"],
            reverse=True,
        )
        while len(selected) < self.num_candidates and remaining:
            selected.append(remaining.pop(0))

        best_item = max(selected, key=lambda entry: entry["score"])
        return int(best_item["actions"][0])
