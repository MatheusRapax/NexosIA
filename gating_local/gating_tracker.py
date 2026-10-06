import numpy as np

class GatingLocalTracker:
    """
    Tracker for Gating layer activation over time.
    Note: `overlap_frac` is maintained for signature compatibility but is unused in the current
    cumulative mask + fallback logic.
    """
    def __init__(self, layer_sizes: list[int], gate_frac: float = 0.5, overlap_frac: float = 0.0, seed: int = 0):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.gate_frac = gate_frac
        self.overlap_frac = overlap_frac
        self.base_seed = seed
        self.changepoint_counter = 0
        self.gate = [None] + [np.ones((layer_sizes[l], 1)) for l in range(1, self.L)]  # so ocultas
        self.ever_active_mask = [None] + [np.zeros((layer_sizes[l], 1), dtype=bool) for l in range(1, self.L)]
        self.fallback_first_triggered = {}   # {layer_idx: changepoint_counter da 1a vez que o fallback (tier b ou c) foi usado}
        self.last_resort_triggered = {}      # {layer_idx: set de changepoint_counter onde o tier (c) (ultimo recurso) foi usado}

    def on_changepoint(self) -> None:
        self.changepoint_counter += 1
        for l in range(1, self.L):
            n_units = self.layer_sizes[l]
            n_active = max(1, round(self.gate_frac * n_units))
            rng = np.random.RandomState(self.base_seed + l * 10000 + self.changepoint_counter)
            
            prev_gate_idx = np.flatnonzero(self.gate[l].flatten() > 0.5)          # gate ATUAL (vira "ultimo gate" apos este update)
            never_used_idx = np.flatnonzero(~self.ever_active_mask[l].flatten())  # unidades fora da mascara cumulativa
            
            if len(never_used_idx) >= n_active:
                # Tier (a): capacidade nunca-usada suficiente -- comportamento ideal, sem fallback
                chosen = rng.choice(never_used_idx, size=n_active, replace=False)
            else:
                chosen_never = never_used_idx
                n_remaining = n_active - len(chosen_never)
                pool_b = np.setdiff1d(np.flatnonzero(self.ever_active_mask[l].flatten()), prev_gate_idx)
                if len(pool_b) >= n_remaining:
                    # Tier (b): fallback aleatorio, excluindo o ultimo gate (protege a tarefa imediatamente anterior)
                    extra = rng.choice(pool_b, size=n_remaining, replace=False)
                    chosen = np.concatenate([chosen_never, extra]).astype(int)
                else:
                    # Tier (c): ultimo recurso -- nem excluindo o ultimo gate sobra pool suficiente. Inclui o ultimo gate tambem.
                    n_remaining_c = n_remaining - len(pool_b)
                    extra_c = rng.choice(prev_gate_idx, size=min(n_remaining_c, len(prev_gate_idx)), replace=False)
                    chosen = np.concatenate([chosen_never, pool_b, extra_c]).astype(int)
                    self.last_resort_triggered.setdefault(l, set()).add(self.changepoint_counter)
                self.fallback_first_triggered.setdefault(l, self.changepoint_counter)
            
            new_gate = np.zeros((n_units, 1))
            new_gate[chosen, 0] = 1.0
            self.gate[l] = new_gate
            self.ever_active_mask[l] = np.logical_or(self.ever_active_mask[l], new_gate > 0.5)

    def apply_gate(self, network: "GatedPredictiveCodingNetwork") -> None:
        network.gate = self.gate
