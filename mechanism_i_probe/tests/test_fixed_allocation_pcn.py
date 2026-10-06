import numpy as np
from mechanism_i_probe.fixed_allocation_pcn import FixedAllocationPCN
from probe_harness.train_and_measure import train_and_measure

def test_partitions_disjoint():
    net = FixedAllocationPCN([7, 64, 32, 2], seed=42)
    
    for l in range(1, net.L):
        # A, B, C lists
        a_idx = net._task_blocks['A'][l]
        b_idx = net._task_blocks['B'][l]
        c_idx = net._task_blocks['C'][l]
        
        # uniao deve cobrir tudo
        union = set(a_idx) | set(b_idx) | set(c_idx)
        assert len(union) == net.layer_sizes[l]
        assert len(union) == len(a_idx) + len(b_idx) + len(c_idx) # Sem overlap

def test_gates_mutually_exclusive():
    net = FixedAllocationPCN([7, 64, 32, 2], seed=42)
    
    net.set_task_gate('A')
    gate_A_l1 = net.gate[1].copy()
    
    net.set_task_gate('B')
    gate_B_l1 = net.gate[1].copy()
    
    # Produto Hadamard deve ser zero em tudo
    assert np.sum(gate_A_l1 * gate_B_l1) == 0.0
    
def test_weight_protection():
    net = FixedAllocationPCN([7, 64, 32, 2], seed=42)
    x = np.random.randn(7)
    y = np.array([1, 0])
    
    net.set_task_gate('A')
    for _ in range(5):
        net.train_step(x, y)
        
    W1_before = net.W[1].copy()
    
    # Vamos trocar para B e treinar
    net.set_task_gate('B')
    for _ in range(5):
        net.train_step(x, y)
        
    W1_after = net.W[1].copy()
    
    # As posicoes do gate A devem permanecer inalteradas
    # (a menos que tambem estivessem no gate B, mas sabemos que sao disjuntos)
    idx_A = net._task_blocks['A'][1]
    
    for idx in idx_A:
        assert np.allclose(W1_before[idx, :], W1_after[idx, :])

def test_determinism():
    res1 = train_and_measure(FixedAllocationPCN, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42, task_gate_fn=lambda net, task: net.set_task_gate(task))
    res2 = train_and_measure(FixedAllocationPCN, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42, task_gate_fn=lambda net, task: net.set_task_gate(task))
    assert res1 == res2
    
def test_run_probe_importable():
    import mechanism_i_probe.run_probe
    assert True
