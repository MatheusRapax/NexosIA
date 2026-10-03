import numpy as np
from dual_weight.dual_weight_pcn import DualWeightPCN

def generate_a():
    x = np.random.normal([0.2, 0.2], 0.05)
    return x, 2.0 * x

def generate_b():
    x = np.random.normal([0.8, 0.8], 0.05)
    return x, -2.0 * x

def generate_a_rare():
    x = np.array([0.2, 0.2]) + 3 * 0.05
    return x, 2.0 * x

def test_changepoint_detector_resistencia_intra_tarefa():
    np.random.seed(42)
    beta_slow = 0.02
    beta_delta = 0.1
    threshold = 0.5
    weight_lr = 0.05
    
    pcn_a = DualWeightPCN(layer_sizes=[2, 8, 4, 2], beta_slow=beta_slow, beta_delta=beta_delta, weight_lr=weight_lr)
    
    for _ in range(200):
        x, y = generate_a()
        pcn_a.step(x, y)
        
    deltas_a = []
    for i in range(50):
        if i % 10 == 0:
            x, y = generate_a_rare()
        else:
            x, y = generate_a()
        res = pcn_a.step(x, y)
        deltas_a.append(res['smoothed_delta'])
        
    pico_a = max(deltas_a)
    assert pico_a <= 0.5 * threshold

def test_changepoint_detector_deteccao_shift_real():
    np.random.seed(43)
    beta_slow = 0.02
    beta_delta = 0.1
    threshold = 0.5
    weight_lr = 0.05
    
    pcn_b = DualWeightPCN(layer_sizes=[2, 8, 4, 2], beta_slow=beta_slow, beta_delta=beta_delta, weight_lr=weight_lr)
    
    for _ in range(200):
        x, y = generate_a()
        pcn_b.step(x, y)
        
    deltas_b = []
    crossed_idx = -1
    for i in range(200):
        x, y = generate_b()
        res = pcn_b.step(x, y)
        deltas_b.append(res['smoothed_delta'])
        if res['smoothed_delta'] > threshold and crossed_idx == -1:
            crossed_idx = i
            
    assert crossed_idx != -1 and crossed_idx < 100
    assert max(deltas_b) >= 2 * threshold
