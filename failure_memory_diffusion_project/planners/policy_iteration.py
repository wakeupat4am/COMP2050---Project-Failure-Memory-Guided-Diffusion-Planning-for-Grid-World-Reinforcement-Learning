# Baseline selection and project integration developed by the author.
# Implementation drafting/editing assisted by AI; based on the standard policy-iteration formulation, no copied external code.
from __future__ import annotations

import numpy as np


class PolicyIterationPlanner:
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6):
        self.env = env.copy()
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0.0 for state in self.env.get_all_states()}
        self.policy = {state: 0 for state in self.env.get_all_states()}
        self.last_episode_repeated_failure_rate = 0.0

    def policy_evaluation(self):
        # Evaluate the current policy until the tabular value function converges.
        states = self.env.get_all_states()
        while True:
            delta = 0.0
            for state in states:
                if self.env.is_terminal_state(state):
                    self.V[state] = 0.0
                    continue
                action = self.policy[state]
                next_state, reward, terminated, truncated, _ = self.env.get_transition(state, action)
                next_value = 0.0 if (terminated or truncated) else self.V[next_state]
                # Policy Iteration policy evaluation step.
                new_value = reward + self.gamma * next_value
                delta = max(delta, abs(self.V[state] - new_value))
                self.V[state] = new_value
            if delta < self.theta:
                break

    def policy_improvement(self) -> bool:
        # Greedify the policy against the evaluated value function and report whether it changed.
        stable = True
        for state in self.env.get_all_states():
            if self.env.is_terminal_state(state):
                continue

            old_action = self.policy[state]
            action_values = []
            for action in range(self.env.num_actions):
                next_state, reward, terminated, truncated, _ = self.env.get_transition(state, action)
                next_value = 0.0 if (terminated or truncated) else self.V[next_state]
                # Policy Iteration policy improvement step.
                action_values.append(reward + self.gamma * next_value)

            best_action = int(np.argmax(action_values))
            self.policy[state] = best_action
            if best_action != old_action:
                stable = False
        return stable

    def train(self):
        # Alternate evaluation and improvement until the policy becomes stable.
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def act(self, state):
        return self.policy.get(tuple(state), 0)
