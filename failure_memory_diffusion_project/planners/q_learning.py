# Baseline selection and project integration developed by the author.
# Implementation drafting/editing assisted by AI; based on standard tabular Q-learning, no copied external code.
from __future__ import annotations

import numpy as np


class TabularQLearningPlanner:
    def __init__(
        self,
        env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        episodes: int = 1000,
        seed: int | None = None,
    ):
        self.env = env.copy()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.rng = np.random.default_rng(seed)
        self.states = self.env.get_all_states()
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.Q = np.zeros((len(self.states), self.env.num_actions), dtype=np.float32)
        self.training_returns = []
        self.last_episode_repeated_failure_rate = 0.0

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _choose_action(self, state):
        state_idx = self.state_to_idx[state]
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.num_actions))
        return int(np.argmax(self.Q[state_idx]))

    def train(self):
        # Learn state-action values from sampled interaction rather than a full model of the gridworld.
        for _ in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state_idx = self.state_to_idx[state]
                next_idx = self.state_to_idx[next_state]

                td_target = reward
                if not (terminated or truncated):
                    td_target += self.gamma * np.max(self.Q[next_idx])

                # Q-learning update.
                self.Q[state_idx, action] += self.alpha * (td_target - self.Q[state_idx, action])

                total_reward += reward
                state = next_state
                done = terminated or truncated

            self.training_returns.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):
        state_idx = self.state_to_idx[tuple(state)]
        return int(np.argmax(self.Q[state_idx]))
