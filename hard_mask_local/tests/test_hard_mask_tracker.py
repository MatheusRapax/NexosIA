import numpy as np
from hard_mask_local.hard_mask_tracker import HardMaskLocalTracker

def test_multiple_changepoints_decay():
    tracker = HardMaskLocalTracker(layer_sizes=[2, 3], gamma=0.5, beta_ema=0.1, k_frac=0.5)
    
    local_grad1 = [None, np.ones((3, 2))]
    for _ in range(10):
        tracker.update_ema(local_grad1)
        
    F_ema_task1 = tracker.F_ema[1].copy()
    tracker.on_changepoint([None, np.ones((3, 2)) * 0.1])
    
    F_total_after_1 = tracker.F_total[1].copy()
    assert np.allclose(F_total_after_1, F_ema_task1)
    
    local_grad2 = [None, np.ones((3, 2)) * 2.0]
    for _ in range(10):
        tracker.update_ema(local_grad2)
        
    F_ema_task2 = tracker.F_ema[1].copy()
    tracker.on_changepoint([None, np.ones((3, 2)) * 0.2])
    
    F_total_after_2 = tracker.F_total[1].copy()
    
    # F_total should be gamma * F_old_decayed + F_tarefa_atual
    expected_F_total = 0.5 * F_total_after_1 + F_ema_task2
    assert np.allclose(F_total_after_2, expected_F_total)

def test_fraction_frozen():
    tracker = HardMaskLocalTracker(layer_sizes=[10, 10], k_frac=0.3)
    
    # Create distinct gradients so we don't have ties that mess up the percentile
    grad = np.arange(100).reshape((10, 10)).astype(float)
    local_grad = [None, grad]
    
    tracker.update_ema(local_grad)
    W_current = [None, np.ones((10, 10))]
    tracker.on_changepoint(W_current)
    
    frozen_mask = tracker.frozen_mask[1]
    import math
    assert frozen_mask.sum() == math.ceil(0.3 * frozen_mask.size)

def test_fraction_frozen_ties():
    tracker = HardMaskLocalTracker(layer_sizes=[10, 10], k_frac=0.1)
    
    # Create gradients with many zeros to force ties
    grad = np.zeros((10, 10))
    grad.flat[0:20] = 1.0 # 20% > 0, 80% == 0
    local_grad = [None, grad]
    
    tracker.update_ema(local_grad)
    W_current = [None, np.ones((10, 10))]
    tracker.on_changepoint(W_current)
    
    frozen_mask = tracker.frozen_mask[1]
    import math
    assert frozen_mask.sum() == math.ceil(0.1 * frozen_mask.size)

def test_reset_behavior():
    tracker = HardMaskLocalTracker(layer_sizes=[10, 10], k_frac=0.3)
    
    grad = np.arange(100).reshape((10, 10)).astype(float)
    local_grad = [None, grad]
    
    tracker.update_ema(local_grad)
    
    W_orig = np.ones((10, 10))
    W_current = [None, W_orig.copy()]
    
    tracker.on_changepoint(W_current)
    
    frozen_mask = tracker.frozen_mask[1]
    free_mask = ~frozen_mask
    
    W_after = W_current[1]
    
    # Frozen positions should NOT change
    assert np.all(W_after[frozen_mask] == W_orig[frozen_mask])
    
    # Free positions should change (reset to uniform)
    assert np.any(W_after[free_mask] != W_orig[free_mask])
    
    # Limit check: sqrt(6 / 20) = sqrt(0.3) approx 0.5477
    limit = np.sqrt(6.0 / 20.0)
    assert np.all(W_after[free_mask] >= -limit)
    assert np.all(W_after[free_mask] <= limit)

def test_apply_penalty_replaces_frozen():
    tracker = HardMaskLocalTracker(layer_sizes=[10, 10], k_frac=0.3)
    
    grad = np.arange(100).reshape((10, 10)).astype(float)
    local_grad = [None, grad]
    tracker.update_ema(local_grad)
    
    W_current = [None, np.ones((10, 10))]
    tracker.on_changepoint(W_current)
    
    # Alter everything
    W_new = [None, np.zeros((10, 10))]
    W_new_copy = W_new[1].copy()
    
    tracker.apply_penalty(W_new)
    
    frozen_mask = tracker.frozen_mask[1]
    free_mask = ~frozen_mask
    
    # Frozen positions should be clamped back to w_anchor
    assert np.allclose(W_new[1][frozen_mask], tracker.w_anchor[1][frozen_mask])
    
    # Free positions should not be touched by apply_penalty
    assert np.allclose(W_new[1][free_mask], W_new_copy[free_mask])

def test_monotonicity():
    tracker = HardMaskLocalTracker(layer_sizes=[10, 10], k_frac=0.3)
    
    # Changepoint 1
    grad1 = np.arange(100).reshape((10, 10)).astype(float)
    tracker.update_ema([None, grad1])
    W_current = [None, np.ones((10, 10))]
    tracker.on_changepoint(W_current)
    
    frozen1 = tracker.frozen_mask[1].copy()
    w_anchor1 = tracker.w_anchor[1].copy()
    
    # Changepoint 2 - completely different gradients
    grad2 = np.flip(grad1) # High gradients in different places
    tracker.update_ema([None, grad2])
    tracker.on_changepoint(W_current)
    
    frozen2 = tracker.frozen_mask[1]
    w_anchor2 = tracker.w_anchor[1]
    
    # All frozen from 1 should remain frozen in 2
    assert np.all(frozen2[frozen1] == True)
    
    # w_anchor for previously frozen should not have changed
    assert np.allclose(w_anchor2[frozen1], w_anchor1[frozen1])
    
    # Total frozen should be larger
    assert frozen2.sum() > frozen1.sum()
