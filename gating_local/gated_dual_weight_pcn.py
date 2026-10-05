from dual_weight.dual_weight_pcn import DualWeightPCN
from gating_local.gated_pcn import GatedPredictiveCodingNetwork
from pcn_core.model import PredictiveCodingNetwork

class GatedDualWeightPCN(DualWeightPCN):
    def __init__(self, layer_sizes: list[int], beta_slow: float = 0.001, beta_delta: float = 0.1, inference_lr: float = 0.1, inference_steps: int = 10, weight_lr: float = 0.01, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.beta_slow = beta_slow
        self.beta_delta = beta_delta
        self.smoothed_delta = 0.0
        self.L = len(layer_sizes) - 1
        
        self.fast = GatedPredictiveCodingNetwork(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self.slow = PredictiveCodingNetwork(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        
        self.slow.W = [None] + [w.copy() for w in self.fast.W[1:]]
        self.slow.b = [None] + [b.copy() for b in self.fast.b[1:]]
