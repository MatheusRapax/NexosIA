import numpy as np
from pcn_core.model import PredictiveCodingNetwork

def test_last_local_grad_equivalence():
    net = PredictiveCodingNetwork([2, 10, 5], seed=123)
    np.random.seed(42)
    x = np.random.randn(2)
    y = np.random.randn(5)
    
    energy = net.train_step(x, y)
    
    # Check if the outputs match the expected values
    assert np.isclose(energy, 1.9842130061275653)
    assert np.isclose(net.W[1][0, 0], 0.27629977811362405)
    
    # Also verify that last_local_grad was properly populated and has correct shape
    assert net.last_local_grad is not None
    assert len(net.last_local_grad) == 3
    assert net.last_local_grad[0] is None
    assert net.last_local_grad[1].shape == (10, 2)
    assert net.last_local_grad[2].shape == (5, 10)
