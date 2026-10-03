import numpy as np
from pcn_core.model import PredictiveCodingNetwork

def _energia(rede: PredictiveCodingNetwork, x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * np.sum((rede.predict(x) - y.flatten())**2)

class DualWeightPCN:
    def __init__(self, layer_sizes: list[int], beta_slow: float = 0.02, beta_delta: float = 0.1,
                 inference_lr: float = 0.1, inference_steps: int = 10, weight_lr: float = 0.05, seed: int = 0):
        self.fast = PredictiveCodingNetwork(
            layer_sizes=layer_sizes,
            inference_lr=inference_lr,
            inference_steps=inference_steps,
            weight_lr=weight_lr,
            seed=seed
        )
        self.slow = PredictiveCodingNetwork(
            layer_sizes=layer_sizes,
            inference_lr=inference_lr,
            inference_steps=inference_steps,
            weight_lr=weight_lr,
            seed=seed
        )
        
        self.beta_slow = beta_slow
        self.beta_delta = beta_delta
        self.smoothed_delta = 0.0
        self.L = len(layer_sizes) - 1
        
    def step(self, x_input: np.ndarray, y_target: np.ndarray) -> dict:
        energia_fast_antes = _energia(self.fast, x_input, y_target)
        
        energia_fast_train_step = self.fast.train_step(x_input, y_target)
        
        energia_slow = _energia(self.slow, x_input, y_target)
        
        delta = energia_slow - energia_fast_antes
        self.smoothed_delta = self.beta_delta * delta + (1 - self.beta_delta) * self.smoothed_delta
        
        for l in range(1, self.L + 1):
            self.slow.W[l] = self.beta_slow * self.fast.W[l] + (1 - self.beta_slow) * self.slow.W[l]
            self.slow.b[l] = self.beta_slow * self.fast.b[l] + (1 - self.beta_slow) * self.slow.b[l]
            
        return {
            'delta': delta,
            'smoothed_delta': self.smoothed_delta,
            'energia_fast': energia_fast_antes,
            'energia_slow': energia_slow,
            'energia_fast_train_step': energia_fast_train_step
        }
        
    def should_sleep(self, threshold: float) -> bool:
        return self.smoothed_delta > threshold
