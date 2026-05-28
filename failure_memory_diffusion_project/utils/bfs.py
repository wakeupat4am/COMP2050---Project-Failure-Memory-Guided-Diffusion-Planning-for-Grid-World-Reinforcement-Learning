from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


Position = Tuple[int, int]
ACTION_DELTAS = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
}


def bfs_shortest_path(grid: np.ndarray, start: Position, goal: Position) -> Optional[List[Position]]:
    if start == goal:
        return [start]

    rows, cols = grid.shape
    queue = deque([start])
    parents = {start: None}

    while queue:
        state = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = state[0] + dr, state[1] + dc
            next_state = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1 or next_state in parents:
                continue
            parents[next_state] = state
            if next_state == goal:
                path = [goal]
                cursor = goal
                while parents[cursor] is not None:
                    cursor = parents[cursor]
                    path.append(cursor)
                return list(reversed(path))
            queue.append(next_state)
    return None


def path_to_actions(path: List[Position]) -> List[int]:
    if path is None or len(path) < 2:
        return []
    actions = []
    for current, nxt in zip(path[:-1], path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        actions.append(ACTION_DELTAS[delta])
    return actions


def shortest_path_length(grid: np.ndarray, start: Position, goal: Position) -> float:
    path = bfs_shortest_path(grid, start, goal)
    if path is None:
        return np.inf
    return max(0, len(path) - 1)
