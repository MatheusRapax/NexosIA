import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pcn_core"))

from pcn_core.model import PredictiveCodingNetwork
from decorrelation_probe.probe_pcn import L1ProbePCN, LateralInhibitionProbePCN

def test_zero_params_equivalent_to_baseline():
    layer_sizes = [7, 64, 32, 2]
    x = np.random.randn(7)
    y = np.array([1, 0])
    
    net_base = PredictiveCodingNetwork(layer_sizes, seed=42)
    net_l1 = L1ProbePCN(layer_sizes, seed=42, l1_lambda=0.0)
    net_lat = LateralInhibitionProbePCN(layer_sizes, seed=42, gamma=0.0)
    
    net_base.train_step(x, y)
    net_l1.train_step(x, y)
    net_lat.train_step(x, y)
    
    for w_base, w_l1, w_lat in zip(net_base.W[1:], net_l1.W[1:], net_lat.W[1:]):
        np.testing.assert_allclose(w_base, w_l1, atol=1e-6)
        np.testing.assert_allclose(w_base, w_lat, atol=1e-6)
        
    for b_base, b_l1, b_lat in zip(net_base.b[1:], net_l1.b[1:], net_lat.b[1:]):
        np.testing.assert_allclose(b_base, b_l1, atol=1e-6)
        np.testing.assert_allclose(b_base, b_lat, atol=1e-6)

def test_gamma_greater_than_zero_differs_from_baseline():
    layer_sizes = [7, 64, 32, 2]
    x = np.random.randn(7)
    y = np.array([1, 0])
    
    net_base = PredictiveCodingNetwork(layer_sizes, seed=42)
    net_lat = LateralInhibitionProbePCN(layer_sizes, seed=42, gamma=0.1)
    
    net_base.train_step(x, y)
    net_lat.train_step(x, y)
    
    # Verify that the weights are different
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_allclose(net_base.W[1], net_lat.W[1], atol=1e-6)

def test_determinism():
    layer_sizes = [7, 64, 32, 2]
    x = np.random.randn(7)
    y = np.array([1, 0])
    
    def run_net(cls, **kwargs):
        net = cls(layer_sizes, seed=123, **kwargs)
        net.train_step(x, y)
        return net.W[1]
    
    w_base1 = run_net(PredictiveCodingNetwork)
    w_base2 = run_net(PredictiveCodingNetwork)
    np.testing.assert_allclose(w_base1, w_base2)
    
    w_l1_1 = run_net(L1ProbePCN, l1_lambda=0.01)
    w_l1_2 = run_net(L1ProbePCN, l1_lambda=0.01)
    np.testing.assert_allclose(w_l1_1, w_l1_2)
    
    w_lat_1 = run_net(LateralInhibitionProbePCN, gamma=0.01)
    w_lat_2 = run_net(LateralInhibitionProbePCN, gamma=0.01)
    np.testing.assert_allclose(w_lat_1, w_lat_2)

def test_run_probe_integration():
    import subprocess
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_probe.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use a small flag or env var for test speedup if implemented, or just rely on the script being fast enough
    env["TEST_MODE"] = "1" 
    result = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
