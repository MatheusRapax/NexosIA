import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pcn_core"))

from pcn_core.model import PredictiveCodingNetwork
from synaptic_pruning_probe.pruning_pcn import LeakyRecyclingPCN

def test_zero_decay_no_recycle_equivalent_to_baseline():
    layer_sizes = [7, 64, 32, 2]
    net_base = PredictiveCodingNetwork(layer_sizes, seed=42)
    net_prune = LeakyRecyclingPCN(layer_sizes, seed=42, decay_lambda=0.0, recycle_every=999)
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.randn(7)
        y = np.array([1, 0])
        net_base.train_step(x, y)
        net_prune.train_step(x, y)
        
    for w_b, w_p in zip(net_base.W[1:], net_prune.W[1:]):
        np.testing.assert_allclose(w_b, w_p, atol=1e-6)

def test_decay_shrinks_weights():
    layer_sizes = [7, 64, 32, 2]
    net_base = PredictiveCodingNetwork(layer_sizes, seed=42)
    net_prune = LeakyRecyclingPCN(layer_sizes, seed=42, decay_lambda=0.05, recycle_every=999)
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.randn(7)
        y = np.array([1, 0])
        net_base.train_step(x, y)
        net_prune.train_step(x, y)
        
    for w_b, w_p in zip(net_base.W[1:], net_prune.W[1:]):
        assert np.linalg.norm(w_p) < np.linalg.norm(w_b)

def test_recycling_logic():
    layer_sizes = [7, 64, 32, 2]
    net_prune = LeakyRecyclingPCN(layer_sizes, seed=42, decay_lambda=0.0, recycle_every=3, recycle_frac=0.5)
    
    rng = np.random.RandomState(42)
    
    for i in range(2):
        x = rng.randn(7)
        y = np.array([1, 0])
        net_prune.train_step(x, y)
    
    assert len(net_prune.recycle_log) == 0
    
    # Store old W to compare
    W_old = [np.copy(w) if w is not None else None for w in net_prune.W]
    peak_old = [np.copy(g) if g is not None else None for g in net_prune.historical_peak]
    
    # Step 3 triggers recycle
    x = rng.randn(7)
    y = np.array([1, 0])
    net_prune.train_step(x, y)
    
    assert len(net_prune.recycle_log) > 0
    assert net_prune.recycle_log[0][0] == 3 # step_count
    
    # Verify that in layer 1, some weights were recycled
    # Specifically, those with historical_peak <= threshold
    l = 1
    assert len(np.unique(net_prune.W[l])) > 1
    recycled_mask = (net_prune.historical_peak[l] == 0.0)
    assert np.any(recycled_mask)
    
    limit = np.sqrt(6 / (layer_sizes[l-1] + layer_sizes[l]))
    # Check that the fresh values are within limits
    assert np.all(np.abs(net_prune.W[l][recycled_mask]) <= limit + 1e-6)

def test_historical_peak_monotonicity():
    layer_sizes = [7, 64, 32, 2]
    # No recycling so we can observe monotonicity clearly
    net_prune = LeakyRecyclingPCN(layer_sizes, seed=42, decay_lambda=0.0, recycle_every=999)
    
    rng = np.random.RandomState(42)
    
    x_high = rng.randn(7) * 10
    x_low = rng.randn(7) * 0.1
    y = np.array([1, 0])
    
    net_prune.train_step(x_high, y)
    peak_after_high = [np.copy(p) if p is not None else None for p in net_prune.historical_peak]
    short_ema_after_high = [np.copy(p) if p is not None else None for p in net_prune.short_ema]
    
    # Train with low inputs, short_ema should drop, but historical_peak should not decrease
    for _ in range(5):
        net_prune.train_step(x_low, y)
        
    for l in range(1, net_prune.L + 1):
        # Peak never decreases
        assert np.all(net_prune.historical_peak[l] >= peak_after_high[l])
        # After many low steps, short_ema is likely much lower than historical peak for many coords
        assert np.any(net_prune.short_ema[l] < net_prune.historical_peak[l])

def test_determinism():
    layer_sizes = [7, 64, 32, 2]
    rng = np.random.RandomState(42)
    inputs = [(rng.randn(7), np.array([1, 0])) for _ in range(5)]
    
    def run_net():
        net = LeakyRecyclingPCN(layer_sizes, seed=123, decay_lambda=0.01, recycle_every=2, recycle_frac=0.1)
        for x, y in inputs:
            net.train_step(x, y)
        return net.W[1], net.recycle_log
        
    w1, log1 = run_net()
    w2, log2 = run_net()
    
    np.testing.assert_allclose(w1, w2)
    assert log1 == log2

def test_run_probe_integration():
    import subprocess
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_probe.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["TEST_MODE"] = "1" 
    result = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
