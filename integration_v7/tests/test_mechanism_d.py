import numpy as np
from integration_v7.run_integration_experiment import run_experiment

def test_mechanism_d_smoke():
    # Smoke test end-to-end
    res = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=0.5, num_eps_per_region=5)
    assert len(res) == 5
    for r in res:
        assert isinstance(r, float)

from integration_v7.run_integration_experiment import calc_mse_with_gate, calc_mse, oracle_metrics_history
from gating_local.gated_pcn import GatedPredictiveCodingNetwork

def test_calc_mse_with_gate_restores_gate():
    model = GatedPredictiveCodingNetwork([7, 4, 2])
    original_gate = [None, np.ones((4, 1)), np.ones((2, 1))]
    model.gate = original_gate
    
    override = [None, np.zeros((4, 1)), np.zeros((2, 1))]
    
    # transitions format: list of (obs, action, next_obs)
    # mock it
    transitions = [(np.array([0.5, 0.5]), 0, np.array([0.5, 0.5]))]
    
    calc_mse_with_gate(model, transitions, override)
    
    # Should restore to the same object / value
    assert model.gate is original_gate

def test_calc_mse_with_gate_equal_gate():
    model = GatedPredictiveCodingNetwork([7, 4, 2])
    original_gate = [None, np.ones((4, 1)), np.ones((2, 1))]
    model.gate = original_gate
    
    # transitions format: list of (obs, action, next_obs)
    transitions = [(np.array([0.5, 0.5]), 0, np.array([0.5, 0.5]))]
    
    mse_direct = calc_mse(model, transitions)
    mse_oracle = calc_mse_with_gate(model, transitions, original_gate)
    
    assert np.isclose(mse_direct, mse_oracle)

def test_calc_mse_with_gate_different_gate():
    model = GatedPredictiveCodingNetwork([7, 4, 2])
    original_gate = [None, np.ones((4, 1)), np.ones((2, 1))]
    model.gate = original_gate
    
    # transitions format: list of (obs, action, next_obs)
    transitions = [(np.array([0.5, 0.5]), 0, np.array([0.5, 0.5]))]
    
    mse_direct = calc_mse(model, transitions)
    
    override_gate = [None, np.zeros((4, 1)), np.zeros((2, 1))]
    mse_oracle = calc_mse_with_gate(model, transitions, override_gate)
    
    assert not np.isclose(mse_direct, mse_oracle)

def test_oracle_smoke():
    res = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=0.5, num_eps_per_region=2)
    assert len(res) == 5
    assert 0.5 in oracle_metrics_history
    om = oracle_metrics_history[0.5]
    
    s_ab, o_ab = om['A_pos_B']
    assert s_ab is not None and o_ab is not None
    s_ac, o_ac = om['A_pos_C']
    assert s_ac is not None and o_ac is not None
    s_bc, o_bc = om['B_pos_C']
    assert s_bc is not None and o_bc is not None
