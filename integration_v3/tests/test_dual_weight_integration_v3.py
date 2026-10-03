import pytest
import numpy as np
from integration_v3.run_integration_experiment import run_experiment

def test_integration_criteria():
    erro_A_pos_A_sem, erro_A_pos_B_sem, succ_A_pos_A_sem, succ_A_pos_B_sem = run_experiment(sleep_enabled=False)
    
    assert erro_A_pos_B_sem > erro_A_pos_A_sem, "Esquecimento nao ocorreu na baseline"
    
    erro_A_pos_A_com, erro_A_pos_B_com, succ_A_pos_A_com, succ_A_pos_B_com = run_experiment(sleep_enabled=True, threshold=0.015)
    
    reducao = (erro_A_pos_B_sem - erro_A_pos_B_com) / erro_A_pos_B_sem
    
    assert reducao >= 0.30, f"Reducao de erro foi apenas de {reducao:.2%}"
