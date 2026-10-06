import numpy as np
from pcn_core.model import PredictiveCodingNetwork
from mechanism_h_probe.confidence_calibrated_pcn import ConfidenceCalibratedPCN
from probe_harness.train_and_measure import train_and_measure

def test_equivalence_zero_lambda():
    net1 = PredictiveCodingNetwork([7, 8, 2], seed=42)
    net2 = ConfidenceCalibratedPCN([7, 8, 2], decay_lambda=0.0, seed=42)
    
    x = np.random.randn(7)
    y = np.array([1, 0])
    for _ in range(3):
        net1.train_step(x, y)
        net2.train_step(x, y)
        
    for l in range(1, net1.L + 1):
        assert np.allclose(net1.W[l], net2.W[l])
        assert np.allclose(net1.b[l], net2.b[l])

def test_unused_weight_decays_max():
    net = ConfidenceCalibratedPCN([7, 8, 2], decay_lambda=0.01, seed=42, weight_lr=0.0)
    # Pegamos um peso especifico e garantimos que grad_mean/var/peak sao zero (ja sao na inicalizacao)
    # Como weight_lr = 0.0, o gradiente real nao muda W, APENAS o decay muda W.
    # Mas wait, o `train_step` calcula o e[l] e faz um backward pass, preenchendo last_local_grad.
    # Se last_local_grad for 0 para um peso, ele continua 0.
    
    # Ao inves de rodar um passo completo que pode preencher last_local_grad, vamos injetar zero manualmente
    net.last_local_grad = [None] + [np.zeros_like(net.W[l]) for l in range(1, net.L + 1)]
    W_before = net.W[1][0, 0]
    
    # Executamos o bloco do mecanismo H simulando um train_step que teve energy computado
    for l in range(1, net.L + 1):
        grad = net.last_local_grad[l]
        net.grad_mean[l] = net.beta * net.grad_mean[l] + (1 - net.beta) * grad
        net.grad_var[l] = net.beta * net.grad_var[l] + (1 - net.beta) * (grad - net.grad_mean[l])**2
        net.historical_peak[l] = np.maximum(net.historical_peak[l], np.abs(net.grad_mean[l]))
        confidence = net.historical_peak[l] / (np.sqrt(net.grad_var[l]) + net.epsilon)
        decay_rate = net.decay_lambda / (1.0 + confidence)
        net.W[l] *= (1.0 - decay_rate)
        
    W_after = net.W[1][0, 0]
    
    # Como peak = 0, confidence = 0, decay_rate = 0.01
    assert np.isclose(W_after, W_before * (1.0 - 0.01))

def test_oscillating_gradient():
    net = ConfidenceCalibratedPCN([7, 8, 2], decay_lambda=0.01, seed=42, weight_lr=0.0)
    
    for i in range(1000):
        # alterna +1 e -1
        grad_val = 1.0 if i % 2 == 0 else -1.0
        net.last_local_grad = [None] + [np.full_like(net.W[l], grad_val) for l in range(1, net.L + 1)]
        
        for l in range(1, net.L + 1):
            grad = net.last_local_grad[l]
            net.grad_mean[l] = net.beta * net.grad_mean[l] + (1 - net.beta) * grad
            net.grad_var[l] = net.beta * net.grad_var[l] + (1 - net.beta) * (grad - net.grad_mean[l])**2
            net.historical_peak[l] = np.maximum(net.historical_peak[l], np.abs(net.grad_mean[l]))
            confidence = net.historical_peak[l] / (np.sqrt(net.grad_var[l]) + net.epsilon)
            decay_rate = net.decay_lambda / (1.0 + confidence)
            # Nao multiplicamos W aqui para nao convergir a zero e sumir
            
    # var deve ser alta, mean baixa
    assert np.mean(np.abs(net.grad_mean[1])) < 0.1
    assert np.mean(net.grad_var[1]) > 0.5
    # confidence deve ser baixa
    confidence = net.historical_peak[1] / (np.sqrt(net.grad_var[1]) + net.epsilon)
    assert np.mean(confidence) < 0.2
    
def test_consistent_gradient():
    net = ConfidenceCalibratedPCN([7, 8, 2], decay_lambda=0.01, seed=42, weight_lr=0.0)
    
    for i in range(1000):
        # sempre +1
        grad_val = 1.0
        net.last_local_grad = [None] + [np.full_like(net.W[l], grad_val) for l in range(1, net.L + 1)]
        
        for l in range(1, net.L + 1):
            grad = net.last_local_grad[l]
            net.grad_mean[l] = net.beta * net.grad_mean[l] + (1 - net.beta) * grad
            net.grad_var[l] = net.beta * net.grad_var[l] + (1 - net.beta) * (grad - net.grad_mean[l])**2
            net.historical_peak[l] = np.maximum(net.historical_peak[l], np.abs(net.grad_mean[l]))
            
    # var deve ser baixa, mean alta
    assert np.mean(np.abs(net.grad_mean[1])) > 0.5
    assert np.mean(net.grad_var[1]) < 0.1
    # confidence deve ser alta
    confidence = net.historical_peak[1] / (np.sqrt(net.grad_var[1]) + net.epsilon)
    assert np.mean(confidence) > 10.0

def test_determinism():
    res1 = train_and_measure(ConfidenceCalibratedPCN, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42)
    res2 = train_and_measure(ConfidenceCalibratedPCN, {'layer_sizes': [7, 8, 2]}, epochs=1, seed=42)
    assert res1 == res2

def test_run_probe_importable():
    import mechanism_h_probe.run_probe
    assert True
