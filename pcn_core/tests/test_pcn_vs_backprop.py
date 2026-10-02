import os
import pytest
import numpy as np
from pcn_core.model import PredictiveCodingNetwork
from pcn_core.baseline import BackpropMLP
from pcn_core.data import load_and_preprocess_data

def test_no_autograd_used_in_pcn():
    # Inspeciona o arquivo model.py pra ter certeza que nao tem import do torch ou torch.autograd ou backward()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.py')
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "import torch" not in content, "PCN class should not import torch"
    assert ".backward()" not in content, "PCN class must not use .backward()"
    assert "autograd" not in content.lower(), "PCN class must not use autograd"
    assert "torch.no_grad" not in content, "PCN class should not even need torch.no_grad if it uses pure numpy"

def test_pcn_vs_backprop_accuracy():
    # Roda as duas redes na configuracao completa
    X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test = load_and_preprocess_data(seed=42)
    
    layer_sizes = [64, 32, 16, 10]
    epochs = 30
    
    pcn = PredictiveCodingNetwork(layer_sizes=layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0)
    for epoch in range(epochs):
        for i in range(len(X_train)):
            pcn.train_step(X_train[i], y_train_onehot[i])
            
    correct_pcn = 0
    for i in range(len(X_test)):
        pred = pcn.predict(X_test[i])
        if np.argmax(pred) == y_test[i]:
            correct_pcn += 1
    acc_pcn = correct_pcn / len(X_test)
    
    baseline = BackpropMLP(layer_sizes=layer_sizes, lr=0.01, seed=0)
    for epoch in range(epochs):
        for i in range(len(X_train)):
            baseline.train_step(X_train[i], y_train_onehot[i])
            
    correct_base = 0
    for i in range(len(X_test)):
        pred = baseline.predict(X_test[i])
        if np.argmax(pred) == y_test[i]:
            correct_base += 1
    acc_base = correct_base / len(X_test)
    
    # Check if PCN accuracy is within 5 percentage points of baseline
    assert acc_pcn >= (acc_base - 0.05), f"PCN acc {acc_pcn:.4f} is not within 5% of Baseline acc {acc_base:.4f}"
