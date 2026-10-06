import numpy as np
from pcn_core.model import PredictiveCodingNetwork

class L1ProbePCN(PredictiveCodingNetwork):
    def __init__(self, layer_sizes: list[int], inference_lr: float = 0.1,
                 inference_steps: int = 20, weight_lr: float = 0.01, seed: int = 0,
                 l1_lambda: float = 0.01):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.l1_lambda = l1_lambda

    def train_step(self, x_input: np.ndarray, y_onehot: np.ndarray) -> float:
        # Reshape to column vectors
        x_0 = x_input.reshape(-1, 1)
        y_target = y_onehot.reshape(-1, 1)
        
        # Initialize x_l values with a forward pass
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            x.append(mu_l)
            
        # Clamp output
        x[self.L] = y_target
        
        # Inference phase
        for step in range(self.inference_steps):
            e = [None] * (self.L + 1)
            mu = [None] * (self.L + 1)
            
            # Compute errors
            for l in range(1, self.L + 1):
                mu[l] = self._tanh(self.W[l] @ x[l-1] + self.b[l])
                e[l] = x[l] - mu[l]
            
            # Update x_l (hidden layers only, output is clamped)
            dx = [None] * (self.L + 1)
            for l in range(1, self.L):
                deriv = self._tanh_deriv(self.W[l] @ x[l-1] + self.b[l])
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv - self.l1_lambda * np.sign(x[l])
            
            for l in range(1, self.L):
                x[l] += self.inference_lr * dx[l]
        
        # Final errors after inference
        e = [None] * (self.L + 1)
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            e[l] = x[l] - mu_l
            
        # Total energy
        energy = sum(0.5 * np.sum(e_l**2) for e_l in e[1:])
        
        self.last_local_grad = [None] + [np.outer(e[l], x[l-1]) for l in range(1, self.L + 1)]
        
        # Learning phase
        for l in range(1, self.L + 1):
            dW_l = self.weight_lr * np.outer(e[l], x[l-1])
            db_l = self.weight_lr * e[l]
            self.W[l] += dW_l
            self.b[l] += db_l
            
        return float(energy)


class LateralInhibitionProbePCN(PredictiveCodingNetwork):
    def __init__(self, layer_sizes: list[int], inference_lr: float = 0.1,
                 inference_steps: int = 20, weight_lr: float = 0.01, seed: int = 0,
                 gamma: float = 0.01):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.gamma = gamma

    def train_step(self, x_input: np.ndarray, y_onehot: np.ndarray) -> float:
        # Reshape to column vectors
        x_0 = x_input.reshape(-1, 1)
        y_target = y_onehot.reshape(-1, 1)
        
        # Initialize x_l values with a forward pass
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            x.append(mu_l)
            
        # Clamp output
        x[self.L] = y_target
        
        # Inference phase
        for step in range(self.inference_steps):
            e = [None] * (self.L + 1)
            mu = [None] * (self.L + 1)
            
            # Compute errors
            for l in range(1, self.L + 1):
                mu[l] = self._tanh(self.W[l] @ x[l-1] + self.b[l])
                e[l] = x[l] - mu[l]
            
            # Update x_l (hidden layers only, output is clamped)
            dx = [None] * (self.L + 1)
            for l in range(1, self.L):
                deriv = self._tanh_deriv(self.W[l] @ x[l-1] + self.b[l])
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv - 2 * self.gamma * (np.sum(x[l]) - x[l])
            
            for l in range(1, self.L):
                x[l] += self.inference_lr * dx[l]
        
        # Final errors after inference
        e = [None] * (self.L + 1)
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            e[l] = x[l] - mu_l
            
        # Total energy
        energy = sum(0.5 * np.sum(e_l**2) for e_l in e[1:])
        
        self.last_local_grad = [None] + [np.outer(e[l], x[l-1]) for l in range(1, self.L + 1)]
        
        # Learning phase
        for l in range(1, self.L + 1):
            dW_l = self.weight_lr * np.outer(e[l], x[l-1])
            db_l = self.weight_lr * e[l]
            self.W[l] += dW_l
            self.b[l] += db_l
            
        return float(energy)
