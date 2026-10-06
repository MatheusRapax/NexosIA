import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))

from pcn_core.model import PredictiveCodingNetwork
from synaptic_pruning_probe.pruning_pcn import LeakyRecyclingPCN
from active_inference.environment import GridWorld
from integration_v8.run_integration_experiment import get_region_transitions, calc_mse

def run_scenario(model_class, epochs, **kwargs):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    transitions_C = get_region_transitions(env, 'C')
    
    net = model_class(layer_sizes=[7, 64, 32, 2], seed=42, **kwargs)
    
    # Treino A
    for _ in range(epochs):
        np.random.shuffle(transitions_A)
        for obs, a, next_obs in transitions_A:
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            net.train_step(x_input, next_obs)
            
    erro_A_pos_A = calc_mse(net, transitions_A)
    
    # Treino B
    for _ in range(epochs):
        np.random.shuffle(transitions_B)
        for obs, a, next_obs in transitions_B:
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            net.train_step(x_input, next_obs)
            
    erro_A_pos_B = calc_mse(net, transitions_A)
    erro_B_pos_B = calc_mse(net, transitions_B)
    
    # Treino C
    for _ in range(epochs):
        np.random.shuffle(transitions_C)
        for obs, a, next_obs in transitions_C:
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            net.train_step(x_input, next_obs)
            
    erro_A_pos_C = calc_mse(net, transitions_A)
    erro_B_pos_C = calc_mse(net, transitions_B)
    
    return net, (erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C)

def main():
    epochs = 10 if os.environ.get("TEST_MODE") else 50
    
    print("Running Baseline...")
    net_base, errors_base = run_scenario(PredictiveCodingNetwork, epochs)
    
    # Defaults according to spec
    decay = 0.001
    recycle_every = 200
    recycle_frac = 0.05
    trace_beta = 0.99
    
    print(f"Running LeakyRecyclingPCN (decay={decay}, every={recycle_every}, frac={recycle_frac})...")
    net_prune, errors_prune = run_scenario(LeakyRecyclingPCN, epochs,
                                           decay_lambda=decay, recycle_every=recycle_every,
                                           recycle_frac=recycle_frac, trace_beta=trace_beta)
    
    # Calc reductions
    # (base[idx] - r[idx]) / base[idx]
    def calc_red(r, base):
        red_A_C = (base[3] - r[3]) / base[3]
        red_B_C = (base[4] - r[4]) / base[4]
        return red_A_C, red_B_C

    red_A_C, red_B_C = calc_red(errors_prune, errors_base)
    
    print("\n--- RESULTS ---")
    print(f"{'Model':<15} | {'A pos A':<8} | {'A pos B':<8} | {'B pos B':<8} | {'A pos C':<8} | {'B pos C':<8} | {'Red A-pos-C':<12} | {'Red B-pos-C':<12}")
    print("-" * 110)
    print(f"{'Baseline':<15} | {errors_base[0]:.4f}   | {errors_base[1]:.4f}   | {errors_base[2]:.4f}   | {errors_base[3]:.4f}   | {errors_base[4]:.4f}   | {'-':<12} | {'-':<12}")
    print(f"{'Poda Isolada':<15} | {errors_prune[0]:.4f}   | {errors_prune[1]:.4f}   | {errors_prune[2]:.4f}   | {errors_prune[3]:.4f}   | {errors_prune[4]:.4f}   | {red_A_C:>11.2%} | {red_B_C:>11.2%}")
    
    # Diagnostico de poda
    print("\n--- DIAGNOSTICO DE PODA ---")
    print(f"Total de eventos de reciclagem disparados: {len(net_prune.recycle_log)}")
    if hasattr(net_prune, 'recycled_coords_history'):
        for l in range(1, net_prune.L + 1):
            n_distinct = len(net_prune.recycled_coords_history[l])
            total_params = net_prune.W[l].size
            frac = n_distinct / total_params
            print(f"Camada {l}: {n_distinct}/{total_params} parametros distintos reciclados ({frac:.2%})")

if __name__ == "__main__":
    main()
