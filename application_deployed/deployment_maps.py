from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from utils import bfs_shortest_path, shortest_path_length


Position = tuple[int, int]


@dataclass
class TrainingMap:
    map_id: str
    grid: np.ndarray
    training_goals: list[Position]
    obstacle_density: float
    free_cell_count: int
    goal_reachable_counts: list[int]


@dataclass
class HoldoutTask:
    task_id: str
    start: Position
    goal: Position
    shortest_path_length: int


@dataclass
class HoldoutMap:
    map_id: str
    grid: np.ndarray
    obstacle_density: float
    free_cell_count: int
    tasks: list[HoldoutTask]


def free_positions(grid: np.ndarray) -> list[Position]:
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]


def _neighbors(grid: np.ndarray, state: Position) -> Iterable[Position]:
    rows, cols = grid.shape
    r, c = state
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            yield (nr, nc)


def reachable_states(grid: np.ndarray, source: Position) -> list[Position]:
    frontier = [source]
    visited = {source}
    while frontier:
        current = frontier.pop()
        for nxt in _neighbors(grid, current):
            if nxt not in visited:
                visited.add(nxt)
                frontier.append(nxt)
    return sorted(visited)


def _generate_random_grid(
    rows: int,
    cols: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = (rng.random((rows, cols)) < obstacle_density).astype(np.int64)
    if np.all(grid == 1):
        grid[rng.integers(rows), rng.integers(cols)] = 0
    return grid


def _find_valid_training_goals(
    grid: np.ndarray,
    goals_per_map: int,
    min_reachable_states: int,
    rng: np.random.Generator,
) -> tuple[list[Position], list[int]]:
    candidates = free_positions(grid)
    rng.shuffle(candidates)
    goals: list[Position] = []
    reachable_counts: list[int] = []
    for goal in candidates:
        reachable = reachable_states(grid, goal)
        if len(reachable) < min_reachable_states:
            continue
        goals.append(goal)
        reachable_counts.append(len(reachable))
        if len(goals) == goals_per_map:
            break
    return goals, reachable_counts


def generate_training_maps(
    num_maps: int,
    size: tuple[int, int],
    obstacle_density_range: tuple[float, float],
    goals_per_map: int,
    seed: int,
    min_free_cells: int = 14,
    min_reachable_states: int = 10,
) -> list[TrainingMap]:
    rng = np.random.default_rng(seed)
    rows, cols = size
    training_maps: list[TrainingMap] = []

    while len(training_maps) < num_maps:
        obstacle_density = float(rng.uniform(*obstacle_density_range))
        grid = _generate_random_grid(rows, cols, obstacle_density, rng)
        free_cell_count = int(np.sum(grid == 0))
        if free_cell_count < min_free_cells:
            continue

        goals, reachable_counts = _find_valid_training_goals(
            grid=grid,
            goals_per_map=goals_per_map,
            min_reachable_states=min_reachable_states,
            rng=rng,
        )
        if len(goals) < goals_per_map:
            continue

        training_maps.append(
            TrainingMap(
                map_id=f"train_map_{len(training_maps):02d}",
                grid=grid,
                training_goals=goals,
                obstacle_density=obstacle_density,
                free_cell_count=free_cell_count,
                goal_reachable_counts=reachable_counts,
            )
        )
    return training_maps


def _sample_holdout_tasks(
    grid: np.ndarray,
    num_tasks: int,
    rng: np.random.Generator,
    min_path_length: int,
) -> list[HoldoutTask]:
    free_cells = free_positions(grid)
    tasks: list[HoldoutTask] = []
    seen_pairs: set[tuple[Position, Position]] = set()

    for _ in range(num_tasks * 200):
        start = free_cells[int(rng.integers(len(free_cells)))]
        goal = free_cells[int(rng.integers(len(free_cells)))]
        if start == goal or (start, goal) in seen_pairs:
            continue
        path_length = shortest_path_length(grid, start, goal)
        if not np.isfinite(path_length) or path_length < min_path_length:
            continue
        seen_pairs.add((start, goal))
        tasks.append(
            HoldoutTask(
                task_id=f"task_{len(tasks):02d}",
                start=start,
                goal=goal,
                shortest_path_length=int(path_length),
            )
        )
        if len(tasks) == num_tasks:
            break
    return tasks


def generate_holdout_map(
    size: tuple[int, int],
    obstacle_density_range: tuple[float, float],
    num_tasks: int,
    seed: int,
    min_free_cells: int = 14,
    min_path_length: int = 5,
) -> HoldoutMap:
    rng = np.random.default_rng(seed)
    rows, cols = size

    while True:
        obstacle_density = float(rng.uniform(*obstacle_density_range))
        grid = _generate_random_grid(rows, cols, obstacle_density, rng)
        free_cell_count = int(np.sum(grid == 0))
        if free_cell_count < min_free_cells:
            continue

        tasks = _sample_holdout_tasks(
            grid=grid,
            num_tasks=num_tasks,
            rng=rng,
            min_path_length=min_path_length,
        )
        if len(tasks) < num_tasks:
            continue

        return HoldoutMap(
            map_id="holdout_unseen_map",
            grid=grid,
            obstacle_density=obstacle_density,
            free_cell_count=free_cell_count,
            tasks=tasks,
        )


def task_metadata_rows(holdout_map: HoldoutMap) -> list[dict[str, object]]:
    rows = []
    for task in holdout_map.tasks:
        rows.append(
            {
                "task_id": task.task_id,
                "start_row": task.start[0],
                "start_col": task.start[1],
                "goal_row": task.goal[0],
                "goal_col": task.goal[1],
                "shortest_path_length": task.shortest_path_length,
            }
        )
    return rows


def training_map_metadata_rows(training_maps: list[TrainingMap]) -> list[dict[str, object]]:
    rows = []
    for train_map in training_maps:
        rows.append(
            {
                "map_id": train_map.map_id,
                "obstacle_density": train_map.obstacle_density,
                "free_cell_count": train_map.free_cell_count,
                "goal_count": len(train_map.training_goals),
            }
        )
    return rows


def training_goal_metadata_rows(training_maps: list[TrainingMap]) -> list[dict[str, object]]:
    rows = []
    for train_map in training_maps:
        for idx, (goal, reachable_count) in enumerate(zip(train_map.training_goals, train_map.goal_reachable_counts)):
            rows.append(
                {
                    "map_id": train_map.map_id,
                    "goal_id": f"{train_map.map_id}_goal_{idx:02d}",
                    "goal_row": goal[0],
                    "goal_col": goal[1],
                    "reachable_state_count": reachable_count,
                }
            )
    return rows
