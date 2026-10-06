import numpy as np
from gating_local.gated_pcn import GatedPredictiveCodingNetwork

class ContextGatedLateralInhibitionPCN(GatedPredictiveCodingNetwork):
    def __init__(self, layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0, gamma=0.01):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.gamma = gamma

    def train_step(self, x_input, y_onehot) -> float:
        x_0 = x_input.reshape(-1, 1)
        y_target = y_onehot.reshape(-1, 1)
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            if l < self.L:
                mu_l = mu_l * self.gate[l]
            x.append(mu_l)
        x[self.L] = y_target

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
                gated_x = x[l] * self.gate[l]
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv - 2 * self.gamma * self.gate[l] * (np.sum(gated_x) - gated_x)
            for l in range(1, self.L):
                x[l] += self.inference_lr * dx[l]
                x[l] = x[l] * self.gate[l]

        e = [None] * (self.L + 1)
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            e[l] = x[l] - mu_l
            if l < self.L:
                e[l] = e[l] * self.gate[l]
        energy = sum(0.5 * np.sum(e_l**2) for e_l in e[1:])
        self.last_local_grad = [None] + [np.outer(e[l], x[l-1]) for l in range(1, self.L + 1)]
        for l in range(1, self.L + 1):
            dW_l = self.weight_lr * np.outer(e[l], x[l-1])
            db_l = self.weight_lr * e[l]
            self.W[l] += dW_l
            self.b[l] += db_l
        return float(energy)
