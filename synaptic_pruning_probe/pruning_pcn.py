import numpy as np
from pcn_core.model import PredictiveCodingNetwork

class LeakyRecyclingPCN(PredictiveCodingNetwork):
    def __init__(self, layer_sizes: list[int], inference_lr: float = 0.1,
                 inference_steps: int = 20, weight_lr: float = 0.01, seed: int = 0,
                 decay_lambda: float = 0.001, recycle_every: int = 200, recycle_frac: float = 0.05, trace_beta: float = 0.99):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.decay_lambda = decay_lambda
        self.recycle_every = recycle_every
        self.recycle_frac = recycle_frac
        self.trace_beta = trace_beta
        self.short_ema = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]
        self.historical_peak = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]
        self.step_count = 0
        self.recycle_log = []  # lista de (step_count, layer_idx, n_recycled)
        self._rng_recycle = np.random.RandomState(seed + 999)

    def train_step(self, x_input, y_onehot):
        energy = super().train_step(x_input, y_onehot)  # forward/inferencia/aprendizado normal; preenche self.last_local_grad

        for l in range(1, self.L + 1):
            # Traco de gradiente local (EMA curto)
            self.short_ema[l] = self.trace_beta * self.short_ema[l] + (1 - self.trace_beta) * np.abs(self.last_local_grad[l])
            # Pico historico do traco
            self.historical_peak[l] = np.maximum(self.historical_peak[l], self.short_ema[l])
            # Vazamento sinaptico continuo, aplicado TODO passo (nao so no ciclo de reciclagem)
            self.W[l] *= (1.0 - self.decay_lambda)

        self.step_count += 1
        if self.step_count % self.recycle_every == 0:
            for l in range(1, self.L + 1):
                threshold = np.percentile(self.historical_peak[l].flatten(), self.recycle_frac * 100)
                recycle_mask = self.historical_peak[l] <= threshold
                n_recycled = int(np.sum(recycle_mask))
                if n_recycled > 0:
                    limit = np.sqrt(6 / (self.layer_sizes[l-1] + self.layer_sizes[l]))
                    fresh = self._rng_recycle.uniform(-limit, limit, self.W[l].shape)
                    self.W[l] = np.where(recycle_mask, fresh, self.W[l])
                    self.historical_peak[l] = np.where(recycle_mask, 0.0, self.historical_peak[l])
                    self.short_ema[l] = np.where(recycle_mask, 0.0, self.short_ema[l])
                    self.recycle_log.append((self.step_count, l, n_recycled))
                    # Optionally log the recycled coordinates for distinct coords calculation
                    if not hasattr(self, 'recycled_coords_history'):
                        self.recycled_coords_history = [set() for _ in range(self.L + 1)]
                    coords = np.argwhere(recycle_mask)
                    for c in coords:
                        self.recycled_coords_history[l].add(tuple(c))

        return energy
