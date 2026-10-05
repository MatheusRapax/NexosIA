import numpy as np

class GatingLocalTracker:
    def __init__(self, layer_sizes: list[int], gate_frac: float = 0.5, overlap_frac: float = 0.0, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.gate_frac = gate_frac
        self.overlap_frac = overlap_frac
        self.base_seed = seed
        self.changepoint_counter = 0
        self.gate = [None] + [np.ones((layer_sizes[l], 1)) for l in range(1, self.L)]  # so ocultas

    def on_changepoint(self) -> None:
        self.changepoint_counter += 1
        for l in range(1, self.L):
            n_units = self.layer_sizes[l]
            n_active = max(1, round(self.gate_frac * n_units))
            rng = np.random.RandomState(self.base_seed + l * 10000 + self.changepoint_counter)
            prev_active_idx = np.flatnonzero(self.gate[l].flatten() > 0.5)
            n_from_prev = min(len(prev_active_idx), round(self.overlap_frac * n_active))
            chosen = rng.choice(prev_active_idx, size=n_from_prev, replace=False) if n_from_prev > 0 else np.array([], dtype=int)
            n_remaining = n_active - len(chosen)
            remaining_pool = np.setdiff1d(np.arange(n_units), prev_active_idx)
            if len(remaining_pool) < n_remaining:
                remaining_pool = np.setdiff1d(np.arange(n_units), chosen)
            extra = rng.choice(remaining_pool, size=n_remaining, replace=False)
            new_active = np.concatenate([chosen, extra]).astype(int)
            new_gate = np.zeros((n_units, 1))
            new_gate[new_active, 0] = 1.0
            self.gate[l] = new_gate

    def apply_gate(self, network: "GatedPredictiveCodingNetwork") -> None:
        network.gate = self.gate
