import numpy as np
from dream_sampling_probe.run_probe import run_experiment_accumulated
from integration_v4.run_integration_experiment import run_experiment
import dream_sampling_probe.run_probe as run_probe_module

def test_gamma_zero_equivalence():
    res_orig = run_experiment(sleep_enabled=True)
    res_acc = run_experiment_accumulated(gamma_importance=0.0, num_eps_per_region=225)
    np.testing.assert_allclose(res_orig, res_acc, rtol=1e-5, atol=1e-5)

def test_sanity_check_phase9():
    res_sem = run_experiment(sleep_enabled=False)
    res_com = run_experiment(sleep_enabled=True)
    # Erro em A_pos_C e MAIOR (pior) com sono do que sem sono na Fase 9
    assert res_com[3] > res_sem[3], "A_pos_C deveria ser pior com sono original"

def test_accumulates_importance(monkeypatch):
    importances_vistas = []
    original_sample = run_probe_module.sample_dream_cell_action
    
    def spy_sample(importance, rng, alpha=0.5, epsilon=1.0):
        importances_vistas.append(importance.copy())
        return original_sample(importance, rng, alpha, epsilon)
        
    monkeypatch.setattr(run_probe_module, "sample_dream_cell_action", spy_sample)
    
    run_experiment_accumulated(gamma_importance=0.9, num_eps_per_region=30, threshold=0.015)
    
    assert len(importances_vistas) > 0, "Nenhum sonho disparado"
    
    # Pega as importancias do primeiro e ultimo bloco de sonhos
    # Como sao 4 sonhos por step apos changepoint, importances_vistas tera multiplos itens iguais no inicio.
    imp_start = importances_vistas[0]
    imp_end = importances_vistas[-1]
    
    # Se acumula, imp_end deve conter influencia do inicio, nao deve ser so os ultimos passos.
    # Nao precisamos validar os zeros perfeitos, mas sim que e diferente de gamma_importance=0.0
    importances_vistas.clear()
    
    run_experiment_accumulated(gamma_importance=0.0, num_eps_per_region=30, threshold=0.015)
    
    imp_end_gamma0 = importances_vistas[-1]
    
    # Confirma que imp_end com acumulo e diferente de imp_end sem acumulo (que seria so do segundo changepoint isolado)
    assert not np.allclose(imp_end, imp_end_gamma0), "Importancia nao acumulou"

def test_determinism():
    res_1 = run_experiment_accumulated(gamma_importance=0.5, num_eps_per_region=10)
    res_2 = run_experiment_accumulated(gamma_importance=0.5, num_eps_per_region=10)
    np.testing.assert_allclose(res_1, res_2, rtol=1e-5, atol=1e-5)

def test_run_probe_lightweight():
    # Roda o script rapidamente via API
    res = run_experiment_accumulated(gamma_importance=0.9, num_eps_per_region=5)
    assert len(res) == 5
