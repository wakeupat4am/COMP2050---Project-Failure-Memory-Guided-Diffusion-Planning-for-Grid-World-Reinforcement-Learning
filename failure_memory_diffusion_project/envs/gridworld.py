from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Position = Tuple[int, int]


@dataclass
class TransitionResult:
    next_state: Position
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class GridWorldEnv:
    # Shared action encoding used by every planner and by the diffusion dataset builder.
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(self, grid: np.ndarray, start: Position, goal: Position, max_steps: int = 50):
        self.grid = np.array(grid, dtype=np.int64)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.max_steps = max_steps
        self.num_actions = 4
        self.height, self.width = self.grid.shape
        self.state = self.start
        self.steps_taken = 0

    def reset(self) -> Position:
        self.state = self.start
        self.steps_taken = 0
        return self.state

    def copy(self) -> "GridWorldEnv":
        return GridWorldEnv(self.grid.copy(), self.start, self.goal, self.max_steps)

    def is_valid_position(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0

    def is_terminal_state(self, pos: Position) -> bool:
        return tuple(pos) == self.goal

    def get_all_states(self) -> List[Position]:
        states = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    states.append((r, c))
        return states

    def _compute_transition(self, state: Position, action: int, steps_taken: int) -> TransitionResult:
        # Central transition model for both online interaction and planner-side simulation.
        if self.is_terminal_state(state):
            return TransitionResult(
                next_state=state,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"success": True, "collision": False, "position": state},
            )

        dr, dc = self.ACTIONS[action]
        candidate = (state[0] + dr, state[1] + dc)
        next_steps = steps_taken + 1

        if not self.is_valid_position(candidate):
            return TransitionResult(
                next_state=state,
                reward=-1.0,
                terminated=True,
                truncated=False,
                info={"success": False, "collision": True, "position": state},
            )

        if self.is_terminal_state(candidate):
            return TransitionResult(
                next_state=candidate,
                reward=1.0,
                terminated=True,
                truncated=False,
                info={"success": True, "collision": False, "position": candidate},
            )

        if next_steps >= self.max_steps:
            return TransitionResult(
                next_state=candidate,
                reward=-0.5,
                terminated=False,
                truncated=True,
                info={"success": False, "collision": False, "position": candidate},
            )

        return TransitionResult(
            next_state=candidate,
            reward=-0.01,
            terminated=False,
            truncated=False,
            info={"success": False, "collision": False, "position": candidate},
        )

    def get_transition(self, state: Position, action: int):
        result = self._compute_transition(tuple(state), int(action), 0)
        return result.next_state, result.reward, result.terminated, result.truncated, result.info

    def step(self, action: int):
        result = self._compute_transition(self.state, int(action), self.steps_taken)
        self.state = result.next_state
        self.steps_taken += 1
        return result.next_state, result.reward, result.terminated, result.truncated, result.info

    def simulate_action_sequence(self, start_state: Position, action_sequence: List[int]):
        # Roll out an open-loop candidate plan and return enough detail for trajectory scoring.
        state = tuple(start_state)
        path = [state]
        total_reward = 0.0
        terminated = False
        truncated = False
        collision = False
        success = False
        steps = 0

        for action in action_sequence:
            result = self._compute_transition(state, int(action), steps)
            state = result.next_state
            total_reward += result.reward
            path.append(state)
            steps += 1
            success = bool(result.info["success"])
            collision = bool(result.info["collision"])
            terminated = result.terminated
            truncated = result.truncated
            if terminated or truncated:
                break

        return {
            "final_state": state,
            "total_reward": total_reward,
            "path": path,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "collision": collision,
        }

    def render(self) -> None:
        grid_view = np.full(self.grid.shape, ".", dtype=object)
        grid_view[self.grid == 1] = "X"
        sr, sc = self.start
        gr, gc = self.goal
        cr, cc = self.state
        grid_view[sr, sc] = "S"
        grid_view[gr, gc] = "G"
        if (cr, cc) not in [self.start, self.goal]:
            grid_view[cr, cc] = "A"
        print("\n".join(" ".join(row) for row in grid_view))
