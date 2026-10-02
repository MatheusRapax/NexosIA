import numpy as np

class RandomAgent:
    def select_action(self, obs: np.ndarray) -> int:
        return np.random.randint(0, 5)
