import numpy as np

class HardMaskLocalTracker:
    def __init__(self, layer_sizes: list[int], gamma: float = 0.9, beta_ema: float = 0.1, k_frac: float = 0.5, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.gamma = gamma
        self.beta_ema = beta_ema
        self.k_frac = k_frac
        self.L = len(layer_sizes) - 1
        self.base_seed = seed
        self.changepoint_counter = 0
        
        self.F_ema = [None]
        self.F_total = [None]
        self.w_anchor = [None]
        self.frozen_mask = [None]
        
        for l in range(1, self.L + 1):
            shape = (layer_sizes[l], layer_sizes[l-1])
            self.F_ema.append(np.zeros(shape))
            self.F_total.append(np.zeros(shape))
            self.w_anchor.append(np.zeros(shape))
            self.frozen_mask.append(np.zeros(shape, dtype=bool))

    def update_ema(self, local_grad: list[np.ndarray]) -> None:
        offset = 1 if len(local_grad) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            grad = local_grad[l - 1 + offset]
            self.F_ema[l] = self.beta_ema * (grad**2) + (1 - self.beta_ema) * self.F_ema[l]

    def on_changepoint(self, W_current: list[np.ndarray]) -> None:
        self.changepoint_counter += 1
        offset = 1 if len(W_current) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            W = W_current[l - 1 + offset]
            
            # Acumula F_total
            F_tarefa_atual = self.F_ema[l].copy()
            self.F_total[l] = self.gamma * self.F_total[l] + F_tarefa_atual
            
            # Calcula mascara do changepoint atual
            if self.k_frac > 0:
                flat = self.F_total[l].flatten()
                N = flat.size
                n_freeze = int(np.ceil(self.k_frac * N))
                if n_freeze > 0:
                    order = np.argsort(-flat, kind='stable')
                    top_idx = order[:n_freeze]
                    candidates_flat = np.zeros(N, dtype=bool)
                    candidates_flat[top_idx] = True
                    candidates = candidates_flat.reshape(self.F_total[l].shape)
                else:
                    candidates = np.zeros_like(self.frozen_mask[l])
            else:
                candidates = np.zeros_like(self.frozen_mask[l])

            # Atualiza mascara acumulada (OR logico)
            self.frozen_mask[l] = self.frozen_mask[l] | candidates
            
            # Reseta (Xavier) as posicoes de W_current FORA da frozen_mask cumulativa
            fan_in = self.layer_sizes[l-1]
            fan_out = self.layer_sizes[l]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            # RNG deterministico
            rng = np.random.RandomState(self.base_seed + l * 10000 + self.changepoint_counter)
            
            free_mask = ~self.frozen_mask[l]
            reset_vals = rng.uniform(-limit, limit, size=W.shape)
            W[free_mask] = reset_vals[free_mask]
            
            # w_anchor[l] = W_current[l].copy() (pos-reset)
            self.w_anchor[l] = W.copy()

    def apply_penalty(self, W_current: list[np.ndarray]) -> None:
        offset = 1 if len(W_current) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            W = W_current[l - 1 + offset]
            mask = self.frozen_mask[l]
            W[mask] = self.w_anchor[l][mask]
