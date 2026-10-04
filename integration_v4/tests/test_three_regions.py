import pytest
from integration_v4.run_integration_experiment import run_experiment

def test_catastrophic_forgetting_baseline():
    sem_A_pos_A, sem_A_pos_B, sem_B_pos_B, sem_A_pos_C, sem_B_pos_C = run_experiment(sleep_enabled=False)
    
    # Verifica esquecimento na Regiao A (erro deve aumentar de A para B, e possivelmente de B para C)
    assert sem_A_pos_B > sem_A_pos_A, "Esquecimento nao ocorreu na Regiao A apos B"
    assert sem_A_pos_C > sem_A_pos_A, "Esquecimento nao ocorreu na Regiao A apos C"
    
    # Verifica esquecimento na Regiao B
    assert sem_B_pos_C > sem_B_pos_B, "Esquecimento nao ocorreu na Regiao B apos C"

def test_three_regions_with_sleep():
    com_A_pos_A, com_A_pos_B, com_B_pos_B, com_A_pos_C, com_B_pos_C = run_experiment(sleep_enabled=True)
    # Apenas assegura que o experimento completo executa sem erros, como especificado
    assert True
