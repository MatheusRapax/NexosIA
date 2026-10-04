import numpy as np

class PredictiveCodingNetwork:
    def __init__(self, layer_sizes: list[int], inference_lr: float = 0.1,
                 inference_steps: int = 20, weight_lr: float = 0.01, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.inference_lr = inference_lr
        self.inference_steps = inference_steps
        self.weight_lr = weight_lr
        
        rng = np.random.RandomState(seed)
        
        self.W = [None]
        self.b = [None]
        for l in range(1, self.L + 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (layer_sizes[l] + layer_sizes[l-1]))
            self.W.append(rng.uniform(-limit, limit, (layer_sizes[l], layer_sizes[l-1])))
            self.b.append(np.zeros((layer_sizes[l], 1)))
            
    def _tanh(self, x):
        return np.tanh(x)
        
    def _tanh_deriv(self, x):
        return 1.0 - np.tanh(x)**2

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
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv
            
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

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        x_0 = x_input.reshape(-1, 1)
        
        x = [x_0]
        for l in range(1, self.L + 1):
            mu_l = self._tanh(self.W[l] @ x[l-1] + self.b[l])
            x.append(mu_l)
            
        # Inference phase with output free
        for step in range(self.inference_steps):
            e = [None] * (self.L + 1)
            mu = [None] * (self.L + 1)
            
            for l in range(1, self.L + 1):
                mu[l] = self._tanh(self.W[l] @ x[l-1] + self.b[l])
                e[l] = x[l] - mu[l]
            
            dx = [None] * (self.L + 1)
            for l in range(1, self.L):
                deriv = self._tanh_deriv(self.W[l] @ x[l-1] + self.b[l])
                dx[l] = -e[l] + (self.W[l+1].T @ e[l+1]) * deriv
            
            # x_L is also free, so we update it based on its error (no l+1 layer above it)
            dx[self.L] = -e[self.L]
            
            for l in range(1, self.L + 1):
                x[l] += self.inference_lr * dx[l]
                
        return x[self.L].flatten()


