import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_integration_experiment import run_experiment

def test_smoke_scaled():
    res = run_experiment(layer_sizes=[7,64,32,2], sleep_enabled=True, gating_enabled=True, gate_frac=0.5, num_eps_per_region=5)
    assert isinstance(res, tuple)
    assert len(res) == 5
    for r in res:
        assert isinstance(r, float) or isinstance(r, np.floating)

def test_different_layer_sizes():
    res = run_experiment(layer_sizes=[7,16,8,2], sleep_enabled=True, gating_enabled=False, num_eps_per_region=5)
    assert isinstance(res, tuple)
    assert len(res) == 5

def test_numerical_regression():
    # Import the old version
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from integration_v7.run_integration_experiment import run_experiment as run_experiment_v7
    
    res_v7 = run_experiment_v7(sleep_enabled=True, gating_enabled=True, gate_frac=0.5, num_eps_per_region=5)
    res_v8 = run_experiment(layer_sizes=[7,16,8,2], sleep_enabled=True, gating_enabled=True, gate_frac=0.5, num_eps_per_region=5)
    
    assert np.allclose(res_v7, res_v8)
