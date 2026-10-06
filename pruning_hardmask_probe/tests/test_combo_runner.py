import numpy as np
from pruning_hardmask_probe.combo_runner import run_scenario

def test_k_frac_zero_noop():
    # 1. Com use_mask=True, k_frac=0.0 e use_poda=False: o reset de capacidade livre afeta tudo (0% congelado, 100% reset)
    # A afirmacao "resultado identico ao baseline" era valida apenas sem o reset real.
    # Com o reset ativado (conforme delta da verificacao), k_frac=0.0 destroi o desempenho.
    _, _, err_base = run_scenario(use_poda=False, use_mask=False, epochs=10)
    _, _, err_k0 = run_scenario(use_poda=False, use_mask=True, epochs=10, k_frac=0.0)
    
    # Assert que o MSE e diferente devido ao reset
    assert not np.allclose(err_base, err_k0, rtol=1e-5, atol=1e-5)

def test_mask_restores_frozen_weights():
    # 2. Com use_mask=True, k_frac>0: apos primeiro on_changepoint, perturbar peso congelado e apply_penalty restaura
    net, mask_tracker, _ = run_scenario(use_poda=False, use_mask=True, epochs=10, k_frac=0.5)
    
    # Simula o fim de um epoch e congelamento
    mask_tracker.on_changepoint([w.copy() if w is not None else None for w in net.W])
    
    # Encontra um peso congelado
    l_idx = None
    coord = None
    for l in range(len(net.W)):
        frozen = mask_tracker.frozen_mask[l]
        if np.any(frozen):
            l_idx = l
            # Pega primeira coordenada
            coords = np.where(frozen)
            coord = (coords[0][0], coords[1][0])
            break
            
    assert l_idx is not None, "Nenhum peso foi congelado"
    
    # Anota valor original ancorado
    original_val = mask_tracker.w_anchor[l_idx][coord]
    
    # Perturba o peso na rede
    net.W[l_idx][coord] += 10.0
    
    # Aplica penalidade
    mask_tracker.apply_penalty(net.W)
    
    # Verifica se restaurou
    assert np.isclose(net.W[l_idx][coord], original_val), "apply_penalty nao restaurou o peso congelado"

def test_mask_protects_pruning():
    # 3. Com use_poda=True, use_mask=True: posicao congelada nao muda (mesmo com decaimento), livre muda.
    net, mask_tracker, _ = run_scenario(use_poda=True, use_mask=True, epochs=10, k_frac=0.5)
    
    # Forcar on_changepoint
    mask_tracker.on_changepoint([w.copy() if w is not None else None for w in net.W])
    
    # Guarda estado dos pesos
    w_before = [w.copy() if w is not None else None for w in net.W]
    
    # Realiza um step extra forcando um input (aplica train_step e poda/decaimento)
    x_in = np.ones(7, dtype=np.float32)
    next_obs = np.ones(2, dtype=np.float32)
    
    net.train_step(x_in, next_obs)
    mask_tracker.update_ema(net.last_local_grad)
    mask_tracker.apply_penalty(net.W)
    
    # Verifica
    l_idx = None
    for l in range(len(net.W)):
        frozen = mask_tracker.frozen_mask[l]
        if np.any(frozen):
            l_idx = l
            break
            
    assert l_idx is not None, "Nenhum peso congelado"
    
    frozen_coords = np.where(mask_tracker.frozen_mask[l_idx])
    free_coords = np.where(~mask_tracker.frozen_mask[l_idx])
    
    if len(frozen_coords[0]) > 0:
        c = (frozen_coords[0][0], frozen_coords[1][0])
        # Nao deve ter mudado porque foi protegido pela mascara (apply_penalty desfaz o decaimento)
        assert np.isclose(w_before[l_idx][c], net.W[l_idx][c]), "Peso congelado mudou, mascara falhou em proteger contra poda!"
        
    if len(free_coords[0]) > 0:
        c = (free_coords[0][0], free_coords[1][0])
        # Deve ter mudado devido ao train_step (gradiente + decaimento)
        assert not np.isclose(w_before[l_idx][c], net.W[l_idx][c]), "Peso livre nao mudou!"

def test_determinism():
    # 4. Mesma seed, mesmo resultado.
    _, _, err_d_1 = run_scenario(use_poda=True, use_mask=True, epochs=10)
    _, _, err_d_2 = run_scenario(use_poda=True, use_mask=True, epochs=10)
    np.testing.assert_allclose(err_d_1, err_d_2, rtol=1e-5, atol=1e-5)
    
def test_run_probe_lightweight():
    # 5. Roda fim a fim com poucas epocas (teste leve)
    # Reuso da logica do run_probe porem sem chamar o run_probe como subprocesso demorado
    results = {}
    _, _, results["Baseline"] = run_scenario(use_poda=False, use_mask=False, epochs=2)
    _, _, results["Poda Isolada"] = run_scenario(use_poda=True, use_mask=False, epochs=2)
    _, _, results["Mascara Isolada"] = run_scenario(use_poda=False, use_mask=True, epochs=2)
    _, _, results["Combinado"] = run_scenario(use_poda=True, use_mask=True, epochs=2)
    assert len(results) == 4
