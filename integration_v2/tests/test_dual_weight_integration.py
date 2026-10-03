import pytest
import numpy as np
from integration_v2.run_integration_experiment import run_experiment

def test_integration_criteria():
    erro_A_pos_A_sem, erro_A_pos_B_sem, succ_A_pos_A_sem, succ_A_pos_B_sem = run_experiment(sleep_enabled=False)
    
    # Critério 5: Verificar se há esquecimento
    assert erro_A_pos_B_sem > erro_A_pos_A_sem, "Esquecimento não ocorreu na baseline"
    
    erro_A_pos_A_com, erro_A_pos_B_com, succ_A_pos_A_com, succ_A_pos_B_com = run_experiment(sleep_enabled=True, threshold=0.015, n_dreams=500)
    
    reducao = (erro_A_pos_B_sem - erro_A_pos_B_com) / erro_A_pos_B_sem
    
    # Critério 6 (Vai falhar propositalmente, pois o design atual atinge ~2%)
    assert reducao >= 0.30, f"Redução de erro foi apenas de {reducao:.2%}"
