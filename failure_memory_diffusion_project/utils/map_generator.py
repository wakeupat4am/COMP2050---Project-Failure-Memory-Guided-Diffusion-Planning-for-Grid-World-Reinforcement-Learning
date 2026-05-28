from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .bfs import bfs_shortest_path


def create_easy_map():
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    start = (0, 0)
    goal = (0, 5)
    return grid, start, goal


def create_obstacle_map():
    grid = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    start = (0, 0)
    goal = (0, 5)
    return grid, start, goal


def create_deceptive_map():
    grid = np.array(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    start = (0, 0)
    goal = (0, 5)
    return grid, start, goal


def generate_random_valid_maps(
    num_maps: int,
    size: Tuple[int, int],
    obstacle_density: float,
    seed: int,
) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    maps = []
    rows, cols = size

    while len(maps) < num_maps:
        grid = (rng.random((rows, cols)) < obstacle_density).astype(np.int64)
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        grid[start] = 0
        grid[goal] = 0
        if bfs_shortest_path(grid, start, goal) is not None:
            maps.append((grid, start, goal))
    return maps
