import numpy as np
from gating_local.gating_tracker import GatingLocalTracker

def test_gating_tracker_fraction():
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, overlap_frac=0.0)
    tracker.on_changepoint()
    
    gate1 = tracker.gate[1]
    n_active = gate1.sum()
    assert n_active == round(0.4 * 10)

def test_gating_tracker_overlap_1():
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, overlap_frac=1.0)
    tracker.on_changepoint()
    gate_cp1 = tracker.gate[1].copy()
    
    tracker.on_changepoint()
    gate_cp2 = tracker.gate[1].copy()
    
    # Com overlap 1.0, deve reproduzir o mesmo gate
    assert np.all(gate_cp1 == gate_cp2)

def test_gating_tracker_overlap_0():
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, overlap_frac=0.0)
    tracker.on_changepoint()
    gate_cp1 = tracker.gate[1].copy()
    
    tracker.on_changepoint()
    gate_cp2 = tracker.gate[1].copy()
    
    # Com overlap 0.0 e capacidade suficiente (2 * 4 = 8 <= 10), devem ser disjuntos
    assert np.sum(gate_cp1 * gate_cp2) == 0

def test_gating_tracker_determinism():
    tracker1 = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, seed=42)
    tracker1.on_changepoint()
    tracker1.on_changepoint()
    
    tracker2 = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, seed=42)
    tracker2.on_changepoint()
    tracker2.on_changepoint()
    
    assert np.all(tracker1.gate[1] == tracker2.gate[1])
