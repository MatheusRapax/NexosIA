import numpy as np
from pcn_core.model import PredictiveCodingNetwork

class GatedPredictiveCodingNetwork(PredictiveCodingNetwork):
    """
    Subclasse de PredictiveCodingNetwork com Context-dependent Gating.
    Debito tecnico: duplicacao de `train_step` e `predict` necessaria porque o 
    modelo base nao expoe ganchos para interceptar as ativacoes `x[l]`.
    """
    def __init__(self, layer_sizes: list[int], inference_lr: float = 0.1, inference_steps: int = 20, weight_lr: float = 0.01, seed: int = 0):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.gate = [None] + [np.ones((layer_sizes[l], 1)) for l in range(1, self.L)]

    def train_step(self, x_input, y_onehot) -> float:
        x_0 = x_input.reshape(-1, 1)
        y_target = y_onehot.reshape(-1, 1)
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            if l < self.L:
                mu_l = mu_l * self.gate[l]            # [GATE]
            x.append(mu_l)
        x[self.L] = y_target

        for step in range(self.inference_steps):
            e = [None] * (self.L + 1)
            mu = [None] * (self.L + 1)
            for l in range(1, self.L + 1):
                mu[l] = self._tanh(self.W[l] @ x[l-1] + self.b[l])
                e[l] = x[l] - mu[l]
                if l < self.L:
                    e[l] = e[l] * self.gate[l]        # [GATE]
            dx = [None] * (self.L + 1)
            for l in range(1, self.L):
                deriv = self._tanh_deriv(self.W[l] @ x[l-1] + self.b[l])
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv
            for l in range(1, self.L):
                x[l] += self.inference_lr * dx[l]
                x[l] = x[l] * self.gate[l]            # [GATE]

        e = [None] * (self.L + 1)
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            e[l] = x[l] - mu_l
            if l < self.L:
                e[l] = e[l] * self.gate[l]            # [GATE]
        energy = sum(0.5 * np.sum(e_l**2) for e_l in e[1:])
        self.last_local_grad = [None] + [np.outer(e[l], x[l-1]) for l in range(1, self.L + 1)]
        for l in range(1, self.L + 1):
            dW_l = self.weight_lr * np.outer(e[l], x[l-1])
            db_l = self.weight_lr * e[l]
            self.W[l] += dW_l
            self.b[l] += db_l
        return float(energy)

    def predict(self, x_input) -> np.ndarray:
        x_0 = x_input.reshape(-1, 1)
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            if l < self.L:
                mu_l = mu_l * self.gate[l]
            x.append(mu_l)

        for step in range(self.inference_steps):
            e = [None] * (self.L + 1)
            mu = [None] * (self.L + 1)
            for l in range(1, self.L + 1):
                mu[l] = self._tanh(self.W[l] @ x[l-1] + self.b[l])
                e[l] = x[l] - mu[l]
                if l < self.L:
                    e[l] = e[l] * self.gate[l]
            dx = [None] * (self.L + 1)
            for l in range(1, self.L):
                deriv = self._tanh_deriv(self.W[l] @ x[l-1] + self.b[l])
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv
            dx[self.L] = -e[self.L]
            for l in range(1, self.L + 1):
                x[l] += self.inference_lr * dx[l]
                if l < self.L:
                    x[l] = x[l] * self.gate[l]
        return x[self.L].flatten()
