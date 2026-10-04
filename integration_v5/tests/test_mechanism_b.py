import numpy as np
from integration_v5.run_integration_experiment import run_experiment

def test_mechanism_b_runs():
    # Run a reduced version of the experiment to ensure EWC mechanism works without crashing
    res = run_experiment(sleep_enabled=True, ewc_enabled=True, ewc_lam=1.0, ewc_gamma=0.9, num_eps_per_region=10)
    
    # We just assert that it returns the 5 MSE values as floats
    assert len(res) == 5
    for val in res:
        assert isinstance(val, float)
        assert not np.isnan(val)
