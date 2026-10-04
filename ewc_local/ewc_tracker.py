import numpy as np

class EWCLocalTracker:
    def __init__(self, layer_sizes: list[int], gamma: float = 0.9, lam: float = 1.0, beta_ema: float = 0.1, epsilon_c: float = 1e-6):
        self.layer_sizes = layer_sizes
        self.gamma = gamma
        self.lam = lam
        self.beta_ema = beta_ema
        self.epsilon_c = epsilon_c
        self.L = len(layer_sizes) - 1
        
        self.F_ema = [None]
        self.F_total = [None]
        self.w_anchor = [None]
        self.c_layer = [None] + [self.epsilon_c] * self.L
        
        for l in range(1, self.L + 1):
            shape = (layer_sizes[l], layer_sizes[l-1])
            self.F_ema.append(np.zeros(shape))
            self.F_total.append(np.zeros(shape))
            self.w_anchor.append(np.zeros(shape))

    def update_ema(self, local_grad: list[np.ndarray]) -> None:
        offset = 1 if len(local_grad) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            grad = local_grad[l - 1 + offset]
            self.F_ema[l] = self.beta_ema * (grad**2) + (1 - self.beta_ema) * self.F_ema[l]

    def on_changepoint(self, W_current: list[np.ndarray]) -> None:
        offset = 1 if len(W_current) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            W = W_current[l - 1 + offset]
            F_tarefa_atual = self.F_ema[l].copy()
            F_old_decayed = self.gamma * self.F_total[l]
            self.w_anchor[l] = (F_old_decayed * self.w_anchor[l] + F_tarefa_atual * W) / (F_old_decayed + F_tarefa_atual + 1e-12)
            self.F_total[l] = F_old_decayed + F_tarefa_atual
            self.c_layer[l] = float(np.max(self.F_total[l])) + self.epsilon_c

    def apply_penalty(self, W_current: list[np.ndarray]) -> None:
        offset = 1 if len(W_current) == self.L + 1 else 0
        for l in range(1, self.L + 1):
            W = W_current[l - 1 + offset]
            ratio = self.F_total[l] / (self.F_total[l] + self.c_layer[l])
            W -= self.lam * ratio * (W - self.w_anchor[l])
