import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))
import numpy as np

from dual_weight.dual_weight_pcn import DualWeightPCN
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker
from active_inference.environment import GridWorld

# Substitui EWCLocalTracker por HardMaskLocalTracker
from hard_mask_local.hard_mask_tracker import HardMaskLocalTracker

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

def run_experiment(sleep_enabled=False, hm_enabled=False, beta_slow=0.00005, beta_delta=0.1, threshold=0.015, hm_gamma=0.9, hm_k_frac=0.5, hm_beta_ema=0.1, num_eps_per_region=225):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    
    dual_weight = DualWeightPCN(layer_sizes=[7, 16, 8, 2], beta_slow=beta_slow, beta_delta=beta_delta, 
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)
                                
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)
    
    hm_tracker = None
    if hm_enabled:
        hm_tracker = HardMaskLocalTracker(layer_sizes=dual_weight.fast.layer_sizes, gamma=hm_gamma, beta_ema=hm_beta_ema, k_frac=hm_k_frac, seed=42)
    
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
                
                if hm_enabled:
                    hm_tracker.update_ema(dual_weight.fast.last_local_grad[1:])
                    hm_tracker.apply_penalty(dual_weight.fast.W[1:])
                
                cell = int(round(obs[0]*4)*5 + round(obs[1]*4))
                agent.tracker.update(cell, action, res['energia_fast_train_step'])
                
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    
                if sleep_enabled or hm_enabled:
                    smoothed_delta = res['smoothed_delta']
                    if smoothed_delta > threshold and not was_above_threshold and cooldown_counter == 0:
                        # UPDATE SNAPSHOT ON EVERY CHANGEOVER RISING EDGE
                        visit_snapshot = agent.tracker.visit_count.copy()
                        protecting = True
                        was_above_threshold = True
                        cooldown_counter = 100  # Debounce
                        
                        if hm_enabled:
                            hm_tracker.on_changepoint(dual_weight.fast.W[1:])
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
                            
                            if hm_enabled:
                                hm_tracker.update_ema(dual_weight.fast.last_local_grad[1:])
                                hm_tracker.apply_penalty(dual_weight.fast.W[1:])
                        
                obs = next_obs
                if success:
                    break
            if ep % 50 == 0:
                print(f"Region {region} Episode {ep} done")
                    
    # Train Region A
    run_episodes(num_eps_per_region, 'A')
    erro_A_pos_A = calc_mse(dual_weight.fast, transitions_A)
    
    # Switch to Region B
    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(num_eps_per_region, 'B')
    
    erro_A_pos_B = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_B = calc_mse(dual_weight.fast, transitions_B)
    
    if hm_enabled:
        metrics_B = []
        for l in range(1, dual_weight.fast.L + 1):
            W = dual_weight.fast.W[l]
            w_anchor = hm_tracker.w_anchor[l]
            mask = hm_tracker.frozen_mask[l]
            
            if np.any(mask):
                mean_diff_frozen = float(np.mean(np.abs(W[mask] - w_anchor[mask])))
            else:
                mean_diff_frozen = 0.0
                
            frac_frozen = float(np.mean(mask))
            frac_free = 1.0 - frac_frozen
            
            metrics_B.append((frac_frozen, mean_diff_frozen, frac_free))
        metrics_B_history[hm_k_frac] = metrics_B
    
    # Switch to Region C
    env.goal = (0,4)
    agent.goal_obs = np.array([0.0, 1.0])
    run_episodes(num_eps_per_region, 'C')
    
    erro_A_pos_C = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_C = calc_mse(dual_weight.fast, transitions_B)
    
    return erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C

def main():
    # Known baselines from previous run
    res_a = (0.0583, 0.4256, 0.0494, 0.2078, 0.0528)
    res_b = (0.1848, 0.1437, 0.3229, 0.2436, 0.0258)
    
    k_values = [0.1, 0.25, 0.5, 0.75]
    results_c = {}
    
    for k in k_values:
        print(f"\n=== Rodando Condicao (c) com k_frac={k} ===")
        res_tuple = run_experiment(sleep_enabled=True, hm_enabled=True, hm_k_frac=k, hm_gamma=0.9)
        results_c[k] = res_tuple
        
    print("\n--- TABELA DE CALIBRACAO (Condicao C) ---")
    print(f"{'K':<5} | {'A pos A':<8} | {'A pos B':<8} | {'B pos B':<8} | {'A pos C':<8} | {'B pos C':<8} | {'Red A-apos-B':<12} | {'Red A-apos-C':<12} | {'Red B-apos-C':<12}")
    
    def calc_red(r, base):
        red_A_B = (base[1] - r[1]) / base[1]
        red_A_C = (base[3] - r[3]) / base[3]
        red_B_C = (base[4] - r[4]) / base[4]
        return red_A_B, red_A_C, red_B_C
        
    # Print Mechanism A for comparison
    r_A_B, r_A_C, r_B_C = calc_red(res_b, res_a)
    print(f"{'A(b)':<5} | {res_b[0]:.4f}   | {res_b[1]:.4f}   | {res_b[2]:.4f}   | {res_b[3]:.4f}   | {res_b[4]:.4f}   | {r_A_B:>11.2%} | {r_A_C:>11.2%} | {r_B_C:>11.2%}")
    
    for k in k_values:
        r = results_c[k]
        red_A_B, red_A_C, red_B_C = calc_red(r, res_a)
        print(f"{k:<5} | {r[0]:.4f}   | {r[1]:.4f}   | {r[2]:.4f}   | {r[3]:.4f}   | {r[4]:.4f}   | {red_A_B:>11.2%} | {red_A_C:>11.2%} | {red_B_C:>11.2%}")

    print("\n--- METRICAS DIAGNOSTICAS (Apos Treino B) ---")
    for k in k_values:
        print(f"K={k}:")
        mets = metrics_B_history.get(k)
        if mets:
            for l, (frac_frozen, mean_diff_frozen, frac_free) in enumerate(mets, 1):
                print(f"  Camada {l}: fracao congelada = {frac_frozen:.2%}, media |W - w_anchor| congelada = {mean_diff_frozen:.2e}, posicoes livres = {frac_free:.2%}")

if __name__ == '__main__':
    main()
