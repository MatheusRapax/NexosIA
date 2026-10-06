import numpy as np
from gating_local.gated_pcn import GatedPredictiveCodingNetwork

class FixedAllocationPCN(GatedPredictiveCodingNetwork):
    TASKS = ['A', 'B', 'C']

    def __init__(self, layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0):
        super().__init__(layer_sizes, inference_lr, inference_steps, weight_lr, seed)
        self._task_blocks = {}
        rng = np.random.RandomState(seed)
        for l in range(1, self.L):
            n_units = layer_sizes[l]
            perm = rng.permutation(n_units)
            splits = np.array_split(perm, len(self.TASKS))
            for task, idx in zip(self.TASKS, splits):
                self._task_blocks.setdefault(task, {})[l] = idx

    def set_task_gate(self, task_id):
        for l in range(1, self.L):
            mask = np.zeros((self.layer_sizes[l], 1))
            mask[self._task_blocks[task_id][l], 0] = 1.0
            self.gate[l] = mask
