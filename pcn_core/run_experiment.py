import time
import numpy as np
from data import load_and_preprocess_data
from model import PredictiveCodingNetwork
from baseline import BackpropMLP

def main():
    print("Loading data...")
    X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test = load_and_preprocess_data(seed=42)
    
    layer_sizes = [64, 32, 16, 10]
    epochs = 30
    
    print("\n--- Training Predictive Coding Network (PCN) ---")
    pcn = PredictiveCodingNetwork(layer_sizes=layer_sizes, inference_lr=0.1, inference_steps=20, weight_lr=0.01, seed=0)
    
    start_time = time.time()
    for epoch in range(epochs):
        energy_sum = 0
        for i in range(len(X_train)):
            energy = pcn.train_step(X_train[i], y_train_onehot[i])
            energy_sum += energy
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Energy: {energy_sum/len(X_train):.4f}")
    pcn_time = time.time() - start_time
    
    correct_pcn = 0
    for i in range(len(X_test)):
        pred = pcn.predict(X_test[i])
        if np.argmax(pred) == y_test[i]:
            correct_pcn += 1
    acc_pcn = correct_pcn / len(X_test)
    
    print("\n--- Training Backprop MLP Baseline ---")
    baseline = BackpropMLP(layer_sizes=layer_sizes, lr=0.01, seed=0)
    
    start_time = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for i in range(len(X_train)):
            loss = baseline.train_step(X_train[i], y_train_onehot[i])
            loss_sum += loss
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_sum/len(X_train):.4f}")
    base_time = time.time() - start_time
    
    correct_base = 0
    for i in range(len(X_test)):
        pred = baseline.predict(X_test[i])
        if np.argmax(pred) == y_test[i]:
            correct_base += 1
    acc_base = correct_base / len(X_test)
    
    print("\n" + "="*50)
    print("RESULTADOS DO EXPERIMENTO")
    print("="*50)
    print(f"Modelo         | Acuracia | Tempo Treino")
    print("-" * 50)
    print(f"PCN (Numpy)    | {acc_pcn*100:6.2f}%  | {pcn_time:6.2f}s")
    print(f"Baseline (PT)  | {acc_base*100:6.2f}%  | {base_time:6.2f}s")
    print("-" * 50)
    diff = acc_pcn - acc_base
    print(f"Diferenca: {diff*100:+.2f} pontos percentuais")
    
    if diff >= -0.05:
        print("=> SUCESSO: Acuracia da PCN ficou dentro de 5% do Baseline!")
    else:
        print("=> FALHA: Acuracia da PCN ficou abaixo do limite aceitavel de -5%.")

if __name__ == "__main__":
    main()
