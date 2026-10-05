import numpy as np
from hard_mask_local.ablation_zero_tracker import HardMaskZeroTracker

def test_apply_penalty_zeros_frozen():
    tracker = HardMaskZeroTracker(layer_sizes=[10, 10], k_frac=0.3)
    
    grad = np.arange(100).reshape((10, 10)).astype(float)
    local_grad = [None, grad]
    tracker.update_ema(local_grad)
    
    W_current = [None, np.ones((10, 10))]
    tracker.on_changepoint(W_current)
    
    # Alter everything
    W_new = [None, np.ones((10, 10)) * 5.0]
    W_new_copy = W_new[1].copy()
    
    tracker.apply_penalty(W_new)
    
    frozen_mask = tracker.frozen_mask[1]
    free_mask = ~frozen_mask
    
    # Frozen positions should be zeroed
    assert np.all(W_new[1][frozen_mask] == 0.0)
    
    # Free positions should not be touched by apply_penalty
    assert np.allclose(W_new[1][free_mask], W_new_copy[free_mask])
