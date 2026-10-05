import numpy as np
from gating_local.gated_pcn import GatedPredictiveCodingNetwork
from pcn_core.model import PredictiveCodingNetwork

def test_gated_pcn_equivalence():
    layer_sizes = [2, 4, 2]
    np.random.seed(42)
    net_base = PredictiveCodingNetwork(layer_sizes, seed=42)
    net_gated = GatedPredictiveCodingNetwork(layer_sizes, seed=42)
    
    x = np.random.randn(2)
    y = np.random.randn(2)
    
    e_base = net_base.train_step(x, y)
    e_gated = net_gated.train_step(x, y)
    
    assert np.isclose(e_base, e_gated)
    
    for l in range(1, len(layer_sizes)):
        assert np.allclose(net_base.W[l], net_gated.W[l])
        assert np.allclose(net_base.b[l], net_gated.b[l])
        assert np.allclose(net_base.last_local_grad[l], net_gated.last_local_grad[l])
        
    p_base = net_base.predict(x)
    p_gated = net_gated.predict(x)
    assert np.allclose(p_base, p_gated)

def test_gated_pcn_disabled_unit():
    layer_sizes = [2, 4, 2]
    net_gated = GatedPredictiveCodingNetwork(layer_sizes, seed=42)
    
    # Disable first unit of hidden layer 1
    net_gated.gate[1][0, 0] = 0.0
    
    x = np.random.randn(2)
    y = np.random.randn(2)
    
    W_orig = net_gated.W[1].copy()
    
    net_gated.train_step(x, y)
    
    # Check that gradient for unit 0 is exactly 0
    assert np.all(net_gated.last_local_grad[1][0, :] == 0.0)
    
    # Check that W for unit 0 hasn't changed
    assert np.all(net_gated.W[1][0, :] == W_orig[0, :])
    
    # Check that W for unit 1 HAS changed (it's active)
    assert not np.all(net_gated.W[1][1, :] == W_orig[1, :])
