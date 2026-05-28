from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MCTSNode:
    state: tuple
    parent: "MCTSNode | None" = None
    action_from_parent: int | None = None
    reward_from_parent: float = 0.0
    terminated: bool = False
    truncated: bool = False
    depth: int = 0
    untried_actions: list = field(default_factory=lambda: [0, 1, 2, 3])
    children: dict = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class MonteCarloTreeSearchPlanner:
    def __init__(
        self,
        env,
        simulations: int = 100,
        max_rollout_depth: int = 30,
        exploration_c: float = 1.41,
        heuristic_rollout: bool = True,
        seed: int | None = None,
    ):
        self.env = env.copy()
        self.simulations = simulations
        self.max_rollout_depth = max_rollout_depth
        self.exploration_c = exploration_c
        self.heuristic_rollout = heuristic_rollout
        self.rng = np.random.default_rng(seed)
        self.last_episode_repeated_failure_rate = 0.0

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _manhattan(self, state):
        return abs(state[0] - self.env.goal[0]) + abs(state[1] - self.env.goal[1])

    def _transition(self, state, action, depth):
        # Use the environment's deterministic simulator with the current search depth
        # so MCTS sees the same rewards and truncation logic as real interaction.
        result = self.env._compute_transition(tuple(state), int(action), depth)
        return result.next_state, result.reward, result.terminated, result.truncated, result.info

    def _rollout_action(self, state):
        valid_candidates = []
        for action in range(self.env.num_actions):
            next_state, _, _, _, info = self._transition(state, action, depth=0)
            if not info["collision"]:
                valid_candidates.append((action, next_state))

        if not valid_candidates:
            return int(self.rng.integers(self.env.num_actions))

        if self.heuristic_rollout and self.rng.random() < 0.8:
            distances = [self._manhattan(next_state) for _, next_state in valid_candidates]
            return valid_candidates[int(np.argmin(distances))][0]
        return valid_candidates[int(self.rng.integers(len(valid_candidates)))][0]

    def selection(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.untried_actions and current.children and not (current.terminated or current.truncated):
            best_score = -float("inf")
            best_child = None
            for child in current.children.values():
                # MCTS UCB selection.
                ucb = child.q_value + self.exploration_c * math.sqrt(
                    math.log(current.visits + 1) / (child.visits + 1e-8)
                )
                if ucb > best_score:
                    best_score = ucb
                    best_child = child
            current = best_child
        return current

    def expansion(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions or node.terminated or node.truncated:
            return node
        action = node.untried_actions.pop(int(self.rng.integers(len(node.untried_actions))))
        next_state, reward, terminated, truncated, _ = self._transition(node.state, action, node.depth)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action_from_parent=action,
            reward_from_parent=reward,
            terminated=terminated,
            truncated=truncated,
            depth=node.depth + 1,
            untried_actions=[] if (terminated or truncated) else [0, 1, 2, 3],
        )
        node.children[action] = child
        return child

    def rollout(self, state, depth):
        current_state = state
        total_reward = 0.0
        current_depth = depth
        for _ in range(self.max_rollout_depth):
            if current_state == self.env.goal:
                break
            action = self._rollout_action(current_state)
            next_state, reward, terminated, truncated, _ = self._transition(current_state, action, current_depth)
            total_reward += reward
            current_state = next_state
            current_depth += 1
            if terminated or truncated:
                break
        return total_reward

    def backpropagation(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def search(self, root_state):
        root = MCTSNode(state=tuple(root_state))

        for _ in range(self.simulations):
            leaf = self.selection(root)
            child = self.expansion(leaf)
            reward = child.reward_from_parent
            if not (child.terminated or child.truncated):
                reward += self.rollout(child.state, child.depth)
            self.backpropagation(child, reward)

        if not root.children:
            return 0
        best_action = max(
            root.children.items(),
            key=lambda item: (item[1].q_value, item[1].visits),
        )[0]
        return int(best_action)

    def act(self, state):
        return self.search(state)
