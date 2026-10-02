import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'active_inference'))

from integration.run_unified_agent import run_experiment

def test_unified_agent():
    # Roda condicao sem sono
    taxa_A_sem, taxa_eval_sem, mse_A_sem, mse_B_sem = run_experiment(sleep_enabled=False)
    
    # Roda condicao com sono
    taxa_A_com, taxa_eval_com, mse_A_com, mse_B_com = run_experiment(sleep_enabled=True)
    
    # Criterio 2: Taxa de sucesso na Fase A >= 70% em ambas
    assert taxa_A_sem >= 0.70, f"Fase A sem sono falhou: {taxa_A_sem*100:.2f}%"
    assert taxa_A_com >= 0.70, f"Fase A com sono falhou: {taxa_A_com*100:.2f}%"
    
    # Criterio 3: Reducao MSE >= 30%
    reducao = (mse_B_sem - mse_B_com) / mse_B_sem
    assert reducao >= 0.30, f"Diferenca de reducao de MSE falhou: {reducao*100:.2f}% (esperado >= 30%)"

def test_no_autograd():
    run_file = os.path.join(os.path.dirname(__file__), '..', 'run_unified_agent.py')
    with open(run_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "backward" not in content, "Uso de backward não permitido"
    assert "autograd" not in content, "Uso de autograd não permitido"
