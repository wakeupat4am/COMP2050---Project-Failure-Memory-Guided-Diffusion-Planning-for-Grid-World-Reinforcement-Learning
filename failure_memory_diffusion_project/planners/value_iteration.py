from __future__ import annotations

import numpy as np


class ValueIterationPlanner:
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6):
        self.env = env.copy()
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0.0 for state in self.env.get_all_states()}
        self.policy = {state: 0 for state in self.env.get_all_states()}
        self.last_episode_repeated_failure_rate = 0.0

    def train(self):
        # Solve the tabular MDP exactly by repeatedly applying the Bellman optimality backup.
        states = self.env.get_all_states()
        while True:
            delta = 0.0
            for state in states:
                if self.env.is_terminal_state(state):
                    self.V[state] = 0.0
                    continue

                old_value = self.V[state]
                action_values = []
                for action in range(self.env.num_actions):
                    next_state, reward, terminated, truncated, _ = self.env.get_transition(state, action)
                    next_value = 0.0 if (terminated or truncated) else self.V[next_state]
                    # Value Iteration Bellman optimality update.
                    action_values.append(reward + self.gamma * next_value)

                self.V[state] = max(action_values)
                self.policy[state] = int(np.argmax(action_values))
                delta = max(delta, abs(old_value - self.V[state]))
            if delta < self.theta:
                break

    def act(self, state):
        return self.policy.get(tuple(state), 0)
