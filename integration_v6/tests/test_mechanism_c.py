import numpy as np
from integration_v6.run_integration_experiment import run_experiment

def test_mechanism_c_runs():
    # Run a reduced version of the experiment to ensure Mechanism C works without crashing
    res = run_experiment(sleep_enabled=True, hm_enabled=True, hm_k_frac=0.1, hm_gamma=0.9, num_eps_per_region=10)
    
    # We just assert that it returns the 5 MSE values as floats
    assert len(res) == 5
    for val in res:
        assert isinstance(val, float)
        assert not np.isnan(val)
