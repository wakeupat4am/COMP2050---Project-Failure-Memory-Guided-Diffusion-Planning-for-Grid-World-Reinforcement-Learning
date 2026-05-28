import numpy as np


class RandomPolicyPlanner:
    def __init__(self, action_space_size: int = 4, seed: int | None = None):
        self.action_space_size = action_space_size
        self.rng = np.random.default_rng(seed)
        self.last_episode_repeated_failure_rate = 0.0

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        return int(self.rng.integers(self.action_space_size))
