import numpy as np
from integration_v3.run_integration_experiment import sample_dream_cell_action, cell_to_obs

def test_smoothed_proportional_sampling():
    n_cells = 25
    n_actions = 5
    visit_snapshot = np.zeros((n_cells, n_actions), dtype=int)
    
    visit_snapshot[0, 0] = 1000
    visit_snapshot[24, 4] = 10
    
    rng = np.random.RandomState(42)
    
    counts = {}
    n_samples = 2000
    for _ in range(n_samples):
        cell, action = sample_dream_cell_action(visit_snapshot, rng, alpha=0.5, epsilon=1.0)
        key = (cell, action)
        counts[key] = counts.get(key, 0) + 1
        
    # Check that the most visited cell is still sampled more
    assert counts.get((0, 0), 0) > counts.get((24, 4), 0), "Most visited should still be > less visited"
    
    # Check that a 0-visit cell is sampled at least once in 2000 tries
    # Since epsilon=1.0, adjusted value is 1^0.5 = 1.
    # Total adjusted sum is roughly sqrt(1001) + sqrt(11) + 123*1 = 31.6 + 3.3 + 123 = 157.9
    # Probability of picking a specific 0-visit cell is 1/157.9 = 0.63%
    # Expected number of hits in 2000 tries = 12.6. Very likely to have at least 1.
    assert counts.get((12, 2), 0) > 0, "Cell-action with 0 visits should be sampled due to epsilon"
    
    # Calculate probability of most visited cell to report
    prob_most_visited = counts.get((0, 0), 0) / n_samples
    print(f"Prob most visited: {prob_most_visited:.2%}")
    assert prob_most_visited < 0.96, "Smoothing failed, it is still dominating"

def test_fallback_sampling():
    n_cells = 25
    n_actions = 5
    visit_snapshot = np.zeros((n_cells, n_actions), dtype=int)
    rng = np.random.RandomState(42)
    
    cell, action = sample_dream_cell_action(visit_snapshot, rng)
    
    assert 0 <= cell < n_cells
    assert 0 <= action < n_actions

def test_cell_to_obs():
    obs = cell_to_obs(0)
    assert np.allclose(obs, [0.0, 0.0])
    
    obs = cell_to_obs(24)
    assert np.allclose(obs, [1.0, 1.0])
    
    obs = cell_to_obs(12) # row 2, col 2 -> 2/4 = 0.5
    assert np.allclose(obs, [0.5, 0.5])
