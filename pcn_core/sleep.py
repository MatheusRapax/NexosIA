import numpy as np

class SleepConsolidator:
    def __init__(self, obs_dim: int = 2, n_actions: int = 5,
                 n_dreams: int = 20, prune_rate: float = 0.01, alpha: float = 0.02, seed: int = 0):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_dreams = n_dreams
        self.prune_rate = prune_rate
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        
        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim) * 0.1
        self.chronic_error_ema = None
        self.dream_memory = []
        
    def observe(self, x_obs: np.ndarray, energy: float) -> None:
        alpha = self.alpha
        self.obs_mean = alpha * x_obs + (1 - alpha) * self.obs_mean
        self.obs_std = alpha * np.abs(x_obs - self.obs_mean) + (1 - alpha) * self.obs_std
        
        if self.chronic_error_ema is None:
            self.chronic_error_ema = energy
        else:
            self.chronic_error_ema = alpha * energy + (1 - alpha) * self.chronic_error_ema
            
    def sleep_cycle(self, world_model) -> None:
        self.dream_memory = []
        for _ in range(self.n_dreams):
            obs_dream = np.clip(self.obs_mean + self.obs_std * self.rng.randn(self.obs_dim), 0.0, 1.0)
            action_dream = self.rng.randint(0, self.n_actions)
            onehot_acao = np.zeros(self.n_actions)
            onehot_acao[action_dream] = 1.0
            
            x0_dream = np.concatenate([obs_dream, onehot_acao])
            x_L_dream = world_model.predict(x0_dream).copy()
            self.dream_memory.append((x0_dream, x_L_dream))
            
        if self.chronic_error_ema is not None:
            prune_factor = max(0.0, 1.0 - self.prune_rate * self.chronic_error_ema)
            for l in range(1, len(world_model.W)):
                world_model.W[l] *= prune_factor
                
    def rehearse(self, world_model) -> None:
        for x0_dream, x_L_dream in self.dream_memory:
            world_model.train_step(x0_dream, x_L_dream)
