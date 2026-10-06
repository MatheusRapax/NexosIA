import numpy as np
from gating_local.gating_tracker import GatingLocalTracker

def test_gating_tracker_fraction():
    # 1. Regressao: tier a, fracao ativa
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4, overlap_frac=0.0)
    tracker.on_changepoint()
    gate1 = tracker.gate[1]
    n_active = gate1.sum()
    assert n_active == round(0.4 * 10)

def test_gating_tracker_ever_active_grows():
    # 2. ever_active_mask cresce corretamente e nunca encolhe
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4)
    tracker.on_changepoint()
    mask1 = tracker.ever_active_mask[1].copy()
    
    tracker.on_changepoint()
    mask2 = tracker.ever_active_mask[1].copy()
    
    # mask2 deve conter todos os elementos de mask1
    assert np.all(mask2[mask1 > 0.5])
    # E deve ter crescido (ou mantido, mas neste caso gate_frac=0.4 x 2 > 0.4)
    assert mask2.sum() > mask1.sum()

def test_gating_tracker_tier_b():
    # 3. Tier (a) esgota, tier (b) eh acionado excluindo o ultimo gate
    # layer_sizes pequeno, gate_frac alto
    tracker = GatingLocalTracker(layer_sizes=[2, 10, 2], gate_frac=0.4) # 4 unidades por cp
    
    # CP1: usa 4 unidades (0 a 9)
    tracker.on_changepoint()
    gate1 = tracker.gate[1].copy()
    assert 1 not in tracker.fallback_first_triggered
    
    # CP2: usa 4 unidades
    tracker.on_changepoint()
    gate2 = tracker.gate[1].copy()
    assert 1 not in tracker.fallback_first_triggered
    
    # total usadas ate agora = 8. restam 2.
    # CP3: precisa de 4. Tem 2 nao-usadas. 
    # Vai pegar as 2 nao-usadas e mais 2 do pool (unidades da ever_active_mask que NAO estao em gate2).
    tracker.on_changepoint()
    gate3 = tracker.gate[1].copy()
    
    assert 1 in tracker.fallback_first_triggered
    assert tracker.fallback_first_triggered[1] == 3
    
    # Verifica exclusao do ultimo gate (gate2)
    # intersection of gate3 and gate2 should be empty
    assert np.sum(gate3 * gate2) == 0

def test_gating_tracker_tier_c():
    # 4. Tier (c): excluir o ultimo gate nao deixa pool suficiente.
    tracker = GatingLocalTracker(layer_sizes=[2, 4, 2], gate_frac=0.75) # 3 unidades por cp de 4
    
    # CP1: usa 3 de 4. (sobra 1 nunca usada)
    tracker.on_changepoint()
    gate1 = tracker.gate[1].copy()
    
    # CP2: precisa de 3. tem 1 nunca usada. precisa pegar 2 do pool.
    # pool = unidades ja usadas (3), mas excluir o ultimo gate (3) deixa 0!
    # entao tier c aciona!
    tracker.on_changepoint()
    gate2 = tracker.gate[1].copy()
    
    assert 1 in tracker.fallback_first_triggered
    assert tracker.fallback_first_triggered[1] == 2
    assert 1 in tracker.last_resort_triggered
    assert 2 in tracker.last_resort_triggered[1]
    
    # o gate resultante INCLUI unidades do ultimo gate (pois era impossivel evitar)
    assert np.sum(gate2 * gate1) > 0

def test_gating_tracker_determinism():
    # 5. Determinismo com mesma seed e fallbacks
    tracker1 = GatingLocalTracker(layer_sizes=[2, 4, 2], gate_frac=0.75, seed=42)
    tracker1.on_changepoint()
    tracker1.on_changepoint()
    tracker1.on_changepoint()
    
    tracker2 = GatingLocalTracker(layer_sizes=[2, 4, 2], gate_frac=0.75, seed=42)
    tracker2.on_changepoint()
    tracker2.on_changepoint()
    tracker2.on_changepoint()
    
    assert np.all(tracker1.gate[1] == tracker2.gate[1])
    assert tracker1.fallback_first_triggered == tracker2.fallback_first_triggered
    assert tracker1.last_resort_triggered == tracker2.last_resort_triggered

def test_gating_tracker_fallback_first_triggered_once():
    # 6. fallback_first_triggered so registra a primeira vez
    tracker = GatingLocalTracker(layer_sizes=[2, 4, 2], gate_frac=0.75)
    
    tracker.on_changepoint() # CP1
    tracker.on_changepoint() # CP2 - aciona tier C
    
    assert tracker.fallback_first_triggered[1] == 2
    
    tracker.on_changepoint() # CP3 - aciona tier C dnv
    
    # Deve continuar como 2
    assert tracker.fallback_first_triggered[1] == 2
