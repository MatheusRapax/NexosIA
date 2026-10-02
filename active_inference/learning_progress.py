import numpy as np

class LearningProgressTracker:
    def __init__(self, n_cells: int, n_actions: int,
                 alpha_fast: float = 0.3, alpha_slow: float = 0.05, lp_init: float = 1.0):
        self.fast_ema = np.zeros((n_cells, n_actions))
        self.slow_ema = np.zeros((n_cells, n_actions))
        self.visit_count = np.zeros((n_cells, n_actions), dtype=int)
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.lp_init = lp_init

    def update(self, cell_index: int, action: int, error: float) -> None:
        if self.visit_count[cell_index, action] == 0:
            self.fast_ema[cell_index, action] = error
            self.slow_ema[cell_index, action] = error
        else:
            self.fast_ema[cell_index, action] = self.alpha_fast * error + (1 - self.alpha_fast) * self.fast_ema[cell_index, action]
            self.slow_ema[cell_index, action] = self.alpha_slow * error + (1 - self.alpha_slow) * self.slow_ema[cell_index, action]
        self.visit_count[cell_index, action] += 1

    def get_lp(self, cell_index: int, action: int) -> float:
        if self.visit_count[cell_index, action] == 0:
            return float(self.lp_init)
        else:
            return max(0.0, float(self.slow_ema[cell_index, action] - self.fast_ema[cell_index, action]))
