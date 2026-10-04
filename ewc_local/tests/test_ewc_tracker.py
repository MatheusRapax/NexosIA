import numpy as np
from ewc_local.ewc_tracker import EWCLocalTracker

def test_first_changepoint():
    tracker = EWCLocalTracker(layer_sizes=[2, 3], gamma=0.9, lam=1.0, beta_ema=0.1)
    
    W_current = [None, np.ones((3, 2)) * 0.5]
    local_grad = [None, np.ones((3, 2))]
    
    # Simulate some training
    for _ in range(5):
        tracker.update_ema(local_grad)
        
    tracker.on_changepoint(W_current)
    
    # After first changepoint, w_anchor should be exactly W_current
    assert np.allclose(tracker.w_anchor[1], W_current[1])
    assert np.allclose(tracker.F_total[1], tracker.F_ema[1])

def test_multiple_changepoints_decay():
    tracker = EWCLocalTracker(layer_sizes=[2, 3], gamma=0.5, lam=1.0, beta_ema=0.1)
    
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

def test_apply_penalty_direction():
    tracker = EWCLocalTracker(layer_sizes=[2, 3], gamma=0.9, lam=1.0, beta_ema=0.1)
    
    W_current = [None, np.ones((3, 2)) * 1.0]
    local_grad = [None, np.ones((3, 2))]
    
    tracker.update_ema(local_grad)
    tracker.on_changepoint(W_current)
    
    assert np.all(tracker.F_total[1] > 0)
    
    # Now W drifts away
    W_new = [None, np.ones((3, 2)) * 2.0]
    
    W_new_copy = [w.copy() if w is not None else None for w in W_new]
    tracker.apply_penalty(W_new)
    
    # W_new should move towards w_anchor (which is 1.0)
    # So W_new should be less than W_new_copy
    assert np.all(W_new[1] < W_new_copy[1])
    assert np.all(W_new[1] >= tracker.w_anchor[1]) # Doesn't overshoot w_anchor normally for lam <= 1/F

def test_apply_penalty_zero_F():
    tracker = EWCLocalTracker(layer_sizes=[2, 3], gamma=0.9, lam=1.0, beta_ema=0.1)
    
    # No changepoint called, so F_total is zero
    assert np.allclose(tracker.F_total[1], 0.0)
    
    W_new = [None, np.ones((3, 2)) * 2.0]
    W_new_copy = [w.copy() if w is not None else None for w in W_new]
    
    tracker.apply_penalty(W_new)
    
    # Should not change
    assert np.allclose(W_new[1], W_new_copy[1])

def test_apply_penalty_saturates():
    tracker = EWCLocalTracker(layer_sizes=[2, 3], gamma=0.9, lam=1.0, beta_ema=1.0)
    
    local_grad = [None, np.ones((3, 2)) * 100.0]
    for _ in range(5):
        tracker.update_ema(local_grad)
        
    tracker.on_changepoint([None, np.ones((3, 2))])
    
    W_new = [None, np.ones((3, 2)) * 2.0]
    W_new_copy = [w.copy() if w is not None else None for w in W_new]
    
    tracker.apply_penalty(W_new)
    
    ratio = (W_new_copy[1] - W_new[1]) / (tracker.lam * (W_new_copy[1] - tracker.w_anchor[1]))
    assert np.all(ratio < 1.0)

def test_apply_penalty_sparse_no_nan():
    tracker = EWCLocalTracker(layer_sizes=[4, 4], gamma=0.9, lam=1.0, beta_ema=1.0)
    
    grad = np.zeros((4, 4))
    grad[0, 0] = 10.0
    grad[1, 1] = 10.0
    local_grad = [None, grad]
    
    tracker.update_ema(local_grad)
    tracker.on_changepoint([None, np.ones((4, 4))])
    
    assert tracker.c_layer[1] > 0
    assert not np.isnan(tracker.c_layer[1])
    
    W_new = [None, np.ones((4, 4)) * 2.0]
    W_new_copy = [w.copy() if w is not None else None for w in W_new]
    
    tracker.apply_penalty(W_new)
    
    assert not np.any(np.isnan(W_new[1]))
    
    zero_mask = (tracker.F_total[1] == 0)
    assert np.all(W_new[1][zero_mask] == W_new_copy[1][zero_mask])
