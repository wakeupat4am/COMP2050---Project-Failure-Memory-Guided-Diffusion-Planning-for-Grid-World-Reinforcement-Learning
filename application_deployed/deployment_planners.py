from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401
from Exploration.combined_exploration_planner import CombinedExplorationFailureMemoryPlanner
from planners.diffusion_planner import StandardDiffusionPlanner
from planners.failure_memory_planner import FailureMemoryDiffusionPlanner

from map_conditioned_diffusion import encode_map_condition


class MapConditionedPlannerMixin:
    def _condition(self, state):
        return encode_map_condition(self.env.grid, state, self.env.goal)


class MapConditionedStandardDiffusionPlanner(MapConditionedPlannerMixin, StandardDiffusionPlanner):
    pass


class MapConditionedFailureMemoryDiffusionPlanner(MapConditionedPlannerMixin, FailureMemoryDiffusionPlanner):
    pass


class MapConditionedImprovedFailureMemoryPlanner(MapConditionedPlannerMixin, CombinedExplorationFailureMemoryPlanner):
    pass
