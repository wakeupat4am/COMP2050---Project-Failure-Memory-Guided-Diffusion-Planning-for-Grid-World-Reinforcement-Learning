from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .seed import set_global_seed


def evaluate_planner(planner, env, num_episodes: int, seed: int, bfs_length: Optional[float] = None) -> pd.DataFrame:
    set_global_seed(seed)
    records: List[Dict[str, float]] = []

    for episode_idx in range(num_episodes):
        episode_seed = seed * 1000 + episode_idx
        set_global_seed(episode_seed)
        if hasattr(planner, "set_seed"):
            planner.set_seed(episode_seed)
        if hasattr(planner, "start_episode"):
            planner.start_episode()

        state = env.reset()
        done = False
        path = [state]
        total_return = 0.0
        collisions = 0
        step_count = 0
        action_times = []

        while not done:
            t0 = time.perf_counter()
            action = planner.act(state)
            action_times.append(time.perf_counter() - t0)

            next_state, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            path.append(next_state)
            step_count += 1

            if hasattr(planner, "observe_transition"):
                planner.observe_transition(state, action, next_state, reward, terminated, truncated, info)

            if info.get("collision", False):
                collisions = 1

            state = next_state
            done = terminated or truncated

        if hasattr(planner, "notify_episode_result"):
            planner.notify_episode_result(
                success=bool(info["success"]),
                collision=bool(info["collision"]),
                timeout=bool(truncated),
                path=path,
            )

        if hasattr(planner, "end_episode"):
            planner.end_episode(path=path, success=bool(info["success"]), collision=bool(info["collision"]), truncated=truncated)

        success = 1 if info["success"] else 0
        path_length = max(0, len(path) - 1)
        if success and bfs_length not in (None, np.inf):
            optimality_gap = path_length - bfs_length
        else:
            optimality_gap = np.nan

        repeated_failure_rate = getattr(planner, "last_episode_repeated_failure_rate", 0.0)

        records.append(
            {
                "success": success,
                "total_return": total_return,
                "path_length": path_length,
                "collision": collisions,
                "repeated_failure_rate": repeated_failure_rate,
                "optimality_gap": optimality_gap,
                "inference_time_per_action": float(np.mean(action_times)) if action_times else 0.0,
            }
        )

    return pd.DataFrame(records)
