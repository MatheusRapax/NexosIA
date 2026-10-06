import numpy as np
from pcn_core.model import PredictiveCodingNetwork
from gating_local.gated_pcn import GatedPredictiveCodingNetwork
from gating_local.gating_tracker import GatingLocalTracker
from mechanism_g_probe.lateral_inhibition_windowed import LateralInhibitionWindowedPCN
from mechanism_g_probe.context_gated_lateral_inhibition import ContextGatedLateralInhibitionPCN
from probe_harness.train_and_measure import train_and_measure

def test_glite_gamma_0():
    # G-lite with gamma=0.0 should be identical to baseline
    res_baseline = train_and_measure(
        model_class=PredictiveCodingNetwork,
        model_kwargs={'layer_sizes': [7, 8, 2]},
        epochs=1,
        seed=42
    )
    res_glite = train_and_measure(
        model_class=LateralInhibitionWindowedPCN,
        model_kwargs={'layer_sizes': [7, 8, 2], 'gamma': 0.0},
        epochs=1,
        seed=42
    )
    for a, b in zip(res_baseline, res_glite):
        assert abs(a - b) < 1e-6

def test_glite_window():
    net1 = LateralInhibitionWindowedPCN([7, 8, 2], gamma=0.0, seed=42)
    net2 = LateralInhibitionWindowedPCN([7, 8, 2], gamma=1.0, seed=42)
    x = np.random.randn(7)
    y = np.array([1, 0])
    
    net1.on_changepoint()
    net2.on_changepoint()
    
    # Dentro da janela
    e1 = net1.train_step(x, y)
    e2 = net2.train_step(x, y)
    
    # Com gamma > 0, os pesos/energia serao diferentes
    assert abs(e1 - e2) > 1e-6
    
    # Fora da janela
    net1.steps_since_boundary = 200
    net2.steps_since_boundary = 200
    e1_out = net1.train_step(x, y)
    e2_out = net2.train_step(x, y)
    # inibicao desligada, mesma energia se os pesos iniciais deste passo fossem iguais
    # mas ja divergiram. Entao recriamos
    
    net3 = LateralInhibitionWindowedPCN([7, 8, 2], gamma=0.0, seed=42)
    net4 = LateralInhibitionWindowedPCN([7, 8, 2], gamma=1.0, seed=42)
    net3.steps_since_boundary = 200
    net4.steps_since_boundary = 200
    
    e3_out = net3.train_step(x, y)
    e4_out = net4.train_step(x, y)
    assert abs(e3_out - e4_out) < 1e-6


def test_gv2_gamma_0():
    res_gating = train_and_measure(
        model_class=GatedPredictiveCodingNetwork,
        model_kwargs={'layer_sizes': [7, 8, 2]},
        tracker_class=GatingLocalTracker,
        tracker_kwargs={'layer_sizes': [7, 8, 2], 'gate_frac': 0.5, 'seed': 42},
        epochs=1,
        seed=42
    )
    res_gv2 = train_and_measure(
        model_class=ContextGatedLateralInhibitionPCN,
        model_kwargs={'layer_sizes': [7, 8, 2], 'gamma': 0.0},
        tracker_class=GatingLocalTracker,
        tracker_kwargs={'layer_sizes': [7, 8, 2], 'gate_frac': 0.5, 'seed': 42},
        epochs=1,
        seed=42
    )
    for a, b in zip(res_gating, res_gv2):
        assert abs(a - b) < 1e-6

def test_gv2_inside_gate():
    net = ContextGatedLateralInhibitionPCN([7, 8, 2], gamma=1.0, seed=42)
    tracker = GatingLocalTracker([7, 8, 2], gate_frac=0.5, seed=42)
    tracker.apply_gate(net)
    
    # Unidades fora do gate tem ativacao zero
    x = np.random.randn(7)
    y = np.array([1, 0])
    net.train_step(x, y)
    
    # No testamos energia diretamente, testamos se o mecanismo RODA
    pass

def test_determinism():
    res1 = train_and_measure(PredictiveCodingNetwork, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42)
    res2 = train_and_measure(PredictiveCodingNetwork, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42)
    assert res1 == res2
    
def test_run_probe():
    from mechanism_g_probe.run_probe import run_probe
    # apenas certificar que o script pode ser importado
    assert True
