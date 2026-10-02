import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcn_core.model import PredictiveCodingNetwork
from pcn_core.sleep import SleepConsolidator
from consolidation.run_phase4_experiment import get_transitions, train_epochs, measure_mse

def test_catastrophic_forgetting():
    np.random.seed(42)
    transitions_A, transitions_B = get_transitions()
    
    # 1. Sem sono
    model_no_sleep = PredictiveCodingNetwork([7, 16, 8, 2], seed=0)
    train_epochs(model_no_sleep, transitions_A, 50)
    erro_A_pos_A = measure_mse(model_no_sleep, transitions_A)
    
    train_epochs(model_no_sleep, transitions_B, 50)
    erro_A_pos_B_sem_sono = measure_mse(model_no_sleep, transitions_A)
    
    # Critério 3: O esquecimento catastrófico de fato ocorre
    assert erro_A_pos_B_sem_sono > erro_A_pos_A, f"Esquecimento não ocorreu: pos_A={erro_A_pos_A}, pos_B={erro_A_pos_B_sem_sono}"
    
    # 2. Com sono
    model_sleep = PredictiveCodingNetwork([7, 16, 8, 2], seed=0)
    sleep = SleepConsolidator(obs_dim=2, n_actions=5, n_dreams=20, prune_rate=0.01, seed=0)
    
    train_epochs(model_sleep, transitions_A, 50, sleep_consolidator=sleep, rehearse=False)
    sleep.sleep_cycle(model_sleep)
    
    train_epochs(model_sleep, transitions_B, 50, sleep_consolidator=sleep, rehearse=True)
    erro_A_pos_B_com_sono = measure_mse(model_sleep, transitions_A)
    
    reducao = (erro_A_pos_B_sem_sono - erro_A_pos_B_com_sono) / erro_A_pos_B_sem_sono
    
    # Critério 4: Redução relativa >= 30%
    assert reducao >= 0.30, f"Redução falhou: {reducao*100:.2f}% (esperado >= 30%)"

def test_no_autograd():
    model_file = os.path.join(os.path.dirname(__file__), '..', '..', 'pcn_core', 'model.py')
    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "backward" not in content, "Uso de backward não permitido"
    assert "autograd" not in content, "Uso de autograd não permitido"
    assert "def generate" not in content, "generate() não deve existir"
    
    sleep_file = os.path.join(os.path.dirname(__file__), '..', '..', 'pcn_core', 'sleep.py')
    with open(sleep_file, 'r', encoding='utf-8') as f:
        sleep_content = f.read()
    assert "backward" not in sleep_content, "Uso de backward não permitido"
    assert "autograd" not in sleep_content, "Uso de autograd não permitido"
