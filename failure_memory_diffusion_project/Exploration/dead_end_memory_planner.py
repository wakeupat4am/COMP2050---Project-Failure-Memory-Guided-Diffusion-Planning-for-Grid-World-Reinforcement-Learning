from __future__ import annotations

from planners.failure_memory_planner import FailureMemoryDiffusionPlanner


class DeadEndMemoryDiffusionPlanner(FailureMemoryDiffusionPlanner):
    def __init__(self, *args, tail_k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_k = tail_k

    def update_failure_memory(self, path):
        tail = path[-self.tail_k :] if path else []
        for state in tail:
            self.failure_memory[state] += 1.0
