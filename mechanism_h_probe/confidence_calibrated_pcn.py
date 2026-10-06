import numpy as np
from pcn_core.model import PredictiveCodingNetwork

class ConfidenceCalibratedPCN(PredictiveCodingNetwork):
    def __init__(self, layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0,
                 decay_lambda=0.01, beta=0.99, epsilon=1e-6):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.decay_lambda = decay_lambda
        self.beta = beta
        self.epsilon = epsilon
        self.grad_mean = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]
        self.grad_var = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]
        self.historical_peak = [None] + [np.zeros_like(self.W[l]) for l in range(1, self.L + 1)]

    def train_step(self, x_input, y_onehot) -> float:
        energy = super().train_step(x_input, y_onehot)  # forward/inferencia/aprendizado normal; preenche self.last_local_grad
        for l in range(1, self.L + 1):
            grad = self.last_local_grad[l]
            self.grad_mean[l] = self.beta * self.grad_mean[l] + (1 - self.beta) * grad
            self.grad_var[l] = self.beta * self.grad_var[l] + (1 - self.beta) * (grad - self.grad_mean[l])**2
            self.historical_peak[l] = np.maximum(self.historical_peak[l], np.abs(self.grad_mean[l]))
            confidence = self.historical_peak[l] / (np.sqrt(self.grad_var[l]) + self.epsilon)
            decay_rate = self.decay_lambda / (1.0 + confidence)
            self.W[l] *= (1.0 - decay_rate)
        return energy
