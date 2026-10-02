import numpy as np

class GridWorld:
    def __init__(self, size: int = 5, goal: tuple[int,int] = (4,4), seed: int = 0):
        self.size = size
        self.goal = goal
        self.rng = np.random.RandomState(seed)
        self.row = 0
        self.col = 0
        
    def reset(self) -> np.ndarray:
        while True:
            r = self.rng.randint(0, self.size)
            c = self.rng.randint(0, self.size)
            if (r, c) != self.goal:
                break
        self.row = r
        self.col = c
        return self._get_obs()
        
    def _get_obs(self) -> np.ndarray:
        return np.array([self.row / (self.size - 1), self.col / (self.size - 1)], dtype=np.float32)
        
    def step(self, action: int) -> tuple[np.ndarray, bool, bool]:
        # actions: 0=up(row-1), 1=down(row+1), 2=left(col-1), 3=right(col+1), 4=stay
        if action == 0:
            self.row = max(0, self.row - 1)
        elif action == 1:
            self.row = min(self.size - 1, self.row + 1)
        elif action == 2:
            self.col = max(0, self.col - 1)
        elif action == 3:
            self.col = min(self.size - 1, self.col + 1)
        elif action == 4:
            pass # stay
            
        obs = self._get_obs()
        success = (self.row, self.col) == self.goal
        done = success # Timeout logic will be handled externally by max_steps
        return obs, done, success
