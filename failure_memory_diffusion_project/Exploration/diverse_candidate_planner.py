from __future__ import annotations

from collections import defaultdict

from planners.failure_memory_planner import FailureMemoryDiffusionPlanner


class DiverseCandidateFailureMemoryPlanner(FailureMemoryDiffusionPlanner):
    def __init__(self, *args, raw_multiplier: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_multiplier = raw_multiplier

    def _is_valid_first_action(self, state, action: int) -> bool:
        _, _, _, _, info = self.env.get_transition(state, action)
        return not bool(info["collision"])

    def act(self, state):
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
