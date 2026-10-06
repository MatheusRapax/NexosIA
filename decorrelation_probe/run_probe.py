import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))

from pcn_core.model import PredictiveCodingNetwork
from decorrelation_probe.probe_pcn import L1ProbePCN, LateralInhibitionProbePCN
from active_inference.environment import GridWorld
from integration_v8.run_integration_experiment import get_region_transitions

def get_hidden_activation(net, x_input):
    x_0 = x_input.reshape(-1, 1)
    x = [x_0]
    for l in range(1, net.L + 1):
        x.append(np.tanh(net.W[l] @ x[l-1] + net.b[l]))
        
    for step in range(net.inference_steps):
        e = [None] * (net.L + 1)
        mu = [None] * (net.L + 1)
        
        for l in range(1, net.L + 1):
            mu[l] = np.tanh(net.W[l] @ x[l-1] + net.b[l])
            e[l] = x[l] - mu[l]
        
        dx = [None] * (net.L + 1)
        for l in range(1, net.L):
            deriv = 1.0 - np.tanh(net.W[l] @ x[l-1] + net.b[l])**2
            # Utilizamos sempre a dinamica de inferencia PADRAO (vanilla) para medir a ativacao.
            # O termo de penalidade (L1/Inibicao Lateral) so e aplicado durante o TREINO (train_step)
            # para moldar a representacao aprendida nos pesos. A medicao reflete o efeito residual
            # nos pesos W/b, e nao uma supressao ativa forçada em tempo de inferencia.
            dx[l] = -e[l] + (net.W[l+1].T @ e[l+1]) * deriv
            
        dx[net.L] = -e[net.L]
        
        for l in range(1, net.L + 1):
            x[l] += net.inference_lr * dx[l]
            
    return x[1].flatten()

def train_and_evaluate(model_class, **kwargs):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    
    net = model_class(layer_sizes=[7, 64, 32, 2], seed=42, **kwargs)
    
    epochs = 10 if os.environ.get("TEST_MODE") else 50
    
    # Train A
    for _ in range(epochs):
        np.random.shuffle(transitions_A)
        for obs, a, next_obs in transitions_A:
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            net.train_step(x_input, next_obs)
            
    # Train B
    for _ in range(epochs):
        np.random.shuffle(transitions_B)
        for obs, a, next_obs in transitions_B:
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            net.train_step(x_input, next_obs)
            
    # Collect activations
    act_A = []
    for obs, a, next_obs in transitions_A:
        a_onehot = np.zeros(5, dtype=np.float32)
        a_onehot[a] = 1.0
        x_input = np.concatenate([obs, a_onehot])
        act_A.append(get_hidden_activation(net, x_input))
        
    act_B = []
    for obs, a, next_obs in transitions_B:
        a_onehot = np.zeros(5, dtype=np.float32)
        a_onehot[a] = 1.0
        x_input = np.concatenate([obs, a_onehot])
        act_B.append(get_hidden_activation(net, x_input))
        
    act_A = np.array(act_A)
    act_B = np.array(act_B)
    
    mean_A = np.mean(act_A, axis=0)
    mean_B = np.mean(act_B, axis=0)
    
    # Cosine similarity
    norm_A = np.linalg.norm(mean_A)
    norm_B = np.linalg.norm(mean_B)
    cos_sim = np.dot(mean_A, mean_B) / (norm_A * norm_B + 1e-8)
    
    # Jaccard index
    # We use 75th percentile of magnitude across all activations in that region as threshold
    mag_A = np.abs(mean_A)
    mag_B = np.abs(mean_B)
    threshold = np.percentile(np.concatenate([mag_A, mag_B]), 75)
    
    active_A = set(np.where(mag_A > threshold)[0])
    active_B = set(np.where(mag_B > threshold)[0])
    
    intersection = len(active_A.intersection(active_B))
    union = len(active_A.union(active_B))
    jaccard = intersection / union if union > 0 else 0.0
    
    return cos_sim, jaccard

def main():
    print("Running Baseline...")
    cos_base, jac_base = train_and_evaluate(PredictiveCodingNetwork)
    
    print("Running L1 Probe...")
    cos_l1, jac_l1 = train_and_evaluate(L1ProbePCN, l1_lambda=0.01)
    
    print("Running Lateral Inhibition Probe...")
    cos_lat, jac_lat = train_and_evaluate(LateralInhibitionProbePCN, gamma=0.01)
    
    print("\n--- RESULTS ---")
    print(f"{'Model':<20} | {'Cosine Sim':<12} | {'Jaccard Index':<12}")
    print("-" * 50)
    print(f"{'Baseline':<20} | {cos_base:<12.4f} | {jac_base:<12.4f}")
    print(f"{'L1 (0.01)':<20} | {cos_l1:<12.4f} | {jac_l1:<12.4f}")
    print(f"{'Lateral Inh (0.01)':<20} | {cos_lat:<12.4f} | {jac_lat:<12.4f}")

if __name__ == "__main__":
    main()
