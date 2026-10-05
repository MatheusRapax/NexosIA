import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))
import numpy as np

from gating_local.gated_dual_weight_pcn import GatedDualWeightPCN
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker
from active_inference.environment import GridWorld

from gating_local.gating_tracker import GatingLocalTracker

def sample_dream_cell_action(visit_snapshot: np.ndarray, rng, alpha: float = 0.5, epsilon: float = 1.0) -> tuple[int, int]:
    adjusted = (visit_snapshot.astype(float) + epsilon) ** alpha
    total = np.sum(adjusted)
    flat_probs = adjusted.flatten() / total
    
    n_cells, n_actions = visit_snapshot.shape
    flat_index = rng.choice(n_cells * n_actions, p=flat_probs)
    cell = int(flat_index // n_actions)
    action = int(flat_index % n_actions)
    return cell, action

def cell_to_obs(cell: int) -> np.ndarray:
    row = cell // 5
    col = cell % 5
    return np.array([row/4.0, col/4.0], dtype=np.float32)

def get_region_transitions(env, region):
    transitions = []
    for r in range(env.size):
        for c in range(env.size):
            if region == 'A' and r in [0,1] and c in [0,1]:
                pass
            elif region == 'B' and r in [3,4] and c in [3,4]:
                pass
            elif region == 'C' and r in [0,1] and c in [3,4]:
                pass
            else:
                continue
                
            saved_r, saved_c = env.row, env.col
            for a in range(5):
                env.row, env.col = r, c
                obs = env._get_obs()
                next_obs, _, _ = env.step(a)
                transitions.append((obs, a, next_obs))
            
            env.row, env.col = saved_r, saved_c
    return transitions

def calc_mse(fast_model, transitions):
    total_error = 0.0
    for obs, a, next_obs in transitions:
        a_onehot = np.zeros(5, dtype=np.float32)
        a_onehot[a] = 1.0
        x_input = np.concatenate([obs, a_onehot])
        pred = fast_model.predict(x_input)
        total_error += np.mean((pred - next_obs)**2)
    return total_error / len(transitions)

def rejection_sample_reset(env, region):
    while True:
        obs = env.reset()
        if region == 'A' and env.row in [0,1] and env.col in [0,1]:
            return obs
        elif region == 'B' and env.row in [3,4] and env.col in [3,4]:
            return obs
        elif region == 'C' and env.row in [0,1] and env.col in [3,4]:
            return obs

metrics_B_history = {}
oracle_metrics_history = {}

def calc_mse_with_gate(fast_model, transitions, gate_override):
    original_gate = fast_model.gate
    fast_model.gate = gate_override
    try:
        return calc_mse(fast_model, transitions)
    finally:
        fast_model.gate = original_gate

def snapshot_gate(gate):
    return [None] + [g.copy() for g in gate[1:]]

def run_experiment(sleep_enabled=False, gating_enabled=False, beta_slow=0.00005, beta_delta=0.1, threshold=0.015, gate_frac=0.5, num_eps_per_region=225):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    
    dual_weight = GatedDualWeightPCN(layer_sizes=[7, 16, 8, 2], beta_slow=beta_slow, beta_delta=beta_delta, 
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)
                                
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)
    
    gating_tracker = None
    if gating_enabled:
        gating_tracker = GatingLocalTracker(layer_sizes=dual_weight.fast.layer_sizes, gate_frac=gate_frac, overlap_frac=0.0, seed=42)
        gating_tracker.apply_gate(dual_weight.fast)
    
    was_above_threshold = False
    cooldown_counter = 0
    protecting = False
    visit_snapshot = None
    
    def run_episodes(num_eps, region):
        nonlocal was_above_threshold, cooldown_counter, protecting, visit_snapshot
        for ep in range(num_eps):
            obs = rejection_sample_reset(env, region)
            for step in range(30):
                action = agent.select_action(obs, ep)
                next_obs, done, success = env.step(action)
                
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[action] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                
                res = dual_weight.step(x_input, next_obs)
                
                cell = int(round(obs[0]*4)*5 + round(obs[1]*4))
                agent.tracker.update(cell, action, res['energia_fast_train_step'])
                
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    
                if sleep_enabled or gating_enabled:
                    smoothed_delta = res['smoothed_delta']
                    if smoothed_delta > threshold and not was_above_threshold and cooldown_counter == 0:
                        visit_snapshot = agent.tracker.visit_count.copy()
                        protecting = True
                        was_above_threshold = True
                        cooldown_counter = 100
                        
                        if gating_enabled:
                            gating_tracker.on_changepoint()
                            gating_tracker.apply_gate(dual_weight.fast)
                    elif smoothed_delta <= threshold:
                        was_above_threshold = False
                        
                    if protecting and visit_snapshot is not None and sleep_enabled:
                        for _ in range(4):
                            cell_dream, action_dream = sample_dream_cell_action(visit_snapshot, np.random)
                            obs_dream = cell_to_obs(cell_dream)
                            ad_onehot = np.zeros(5, dtype=np.float32)
                            ad_onehot[action_dream] = 1.0
                            x0_dream = np.concatenate([obs_dream, ad_onehot])
                            y_dream = dual_weight.slow.predict(x0_dream)
                            dual_weight.fast.train_step(x0_dream, y_dream)
                        
                obs = next_obs
                if success:
                    break
            if ep % 50 == 0:
                print(f"Region {region} Episode {ep} done")
                    
    gate_history = {}

    # Train Region A
    run_episodes(num_eps_per_region, 'A')
    if gating_enabled:
        gate_history['A'] = snapshot_gate(gating_tracker.gate)
    erro_A_pos_A = calc_mse(dual_weight.fast, transitions_A)
    
    # Switch to Region B
    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(num_eps_per_region, 'B')
    
    if gating_enabled:
        gate_history['B'] = snapshot_gate(gating_tracker.gate)

    erro_A_pos_B = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_B = calc_mse(dual_weight.fast, transitions_B)
    
    erro_A_pos_B_oracle = None
    if gating_enabled:
        erro_A_pos_B_oracle = calc_mse_with_gate(dual_weight.fast, transitions_A, gate_history['A'])

    if gating_enabled:
        metrics_B = []
        for l in range(1, dual_weight.fast.L): # Only hidden layers
            mask = gating_tracker.gate[l]
            frac_active = float(np.mean(mask > 0.5))
            metrics_B.append(frac_active)
        metrics_B_history[gate_frac] = metrics_B
    
    # Switch to Region C
    env.goal = (0,4)
    agent.goal_obs = np.array([0.0, 1.0])
    run_episodes(num_eps_per_region, 'C')
    
    erro_A_pos_C = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_C = calc_mse(dual_weight.fast, transitions_B)
    
    erro_A_pos_C_oracle = None
    erro_B_pos_C_oracle = None
    
    if gating_enabled:
        erro_A_pos_C_oracle = calc_mse_with_gate(dual_weight.fast, transitions_A, gate_history['A'])
        erro_B_pos_C_oracle = calc_mse_with_gate(dual_weight.fast, transitions_B, gate_history['B'])
        
        oracle_metrics_history[gate_frac] = {
            'A_pos_B': (erro_A_pos_B, erro_A_pos_B_oracle),
            'A_pos_C': (erro_A_pos_C, erro_A_pos_C_oracle),
            'B_pos_C': (erro_B_pos_C, erro_B_pos_C_oracle)
        }
    
    return erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C

def main():
    # Known baselines from previous run
    res_a = (0.0583, 0.4256, 0.0494, 0.2078, 0.0528)
    res_b = (0.1848, 0.1437, 0.3229, 0.2436, 0.0258)
    
    gate_frac_values = [0.25, 0.5, 0.75]
    results_d = {}
    
    for g in gate_frac_values:
        print(f"\n=== Rodando Mecanismo D com gate_frac={g} ===")
        res_tuple = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=g)
        results_d[g] = res_tuple
        
    print("\n--- TABELA DE CALIBRACAO (Mecanismo D) ---")
    print(f"{'Gate':<5} | {'A pos A':<8} | {'A pos B':<8} | {'B pos B':<8} | {'A pos C':<8} | {'B pos C':<8} | {'Red A-apos-B':<12} | {'Red A-apos-C':<12} | {'Red B-apos-C':<12}")
    
    def calc_red(r, base):
        red_A_B = (base[1] - r[1]) / base[1]
        red_A_C = (base[3] - r[3]) / base[3]
        red_B_C = (base[4] - r[4]) / base[4]
        return red_A_B, red_A_C, red_B_C
        
    # Print Mechanism A for comparison
    r_A_B, r_A_C, r_B_C = calc_red(res_b, res_a)
    print(f"{'A(b)':<5} | {res_b[0]:.4f}   | {res_b[1]:.4f}   | {res_b[2]:.4f}   | {res_b[3]:.4f}   | {res_b[4]:.4f}   | {r_A_B:>11.2%} | {r_A_C:>11.2%} | {r_B_C:>11.2%}")
    
    for g in gate_frac_values:
        r = results_d[g]
        red_A_B, red_A_C, red_B_C = calc_red(r, res_a)
        print(f"{g:<5} | {r[0]:.4f}   | {r[1]:.4f}   | {r[2]:.4f}   | {r[3]:.4f}   | {r[4]:.4f}   | {red_A_B:>11.2%} | {red_A_C:>11.2%} | {red_B_C:>11.2%}")

    print("\n--- ORACULO DE TAREFA vs GATE MAIS RECENTE ---")
    print(f"{'Gate':<5} | {'A-pos-B (sem/com)':<17} | {'A-pos-C (sem/com)':<17} | {'B-pos-C (sem/com)':<17}")
    for g in gate_frac_values:
        if g in oracle_metrics_history:
            om = oracle_metrics_history[g]
            s_ab, o_ab = om['A_pos_B']
            s_ac, o_ac = om['A_pos_C']
            s_bc, o_bc = om['B_pos_C']
            print(f"{g:<5} | {s_ab:.4f} / {o_ab:.4f} | {s_ac:.4f} / {o_ac:.4f} | {s_bc:.4f} / {o_bc:.4f}")

    print("\n--- METRICAS DIAGNOSTICAS (Apos Treino B) ---")
    for g in gate_frac_values:
        print(f"gate_frac={g}:")
        mets = metrics_B_history.get(g)
        if mets:
            for l, frac_active in enumerate(mets, 1):
                print(f"  Camada {l}: fracao ativa = {frac_active:.2%}")

if __name__ == '__main__':
    main()
