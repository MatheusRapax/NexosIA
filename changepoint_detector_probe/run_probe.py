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
from integration_v8.run_integration_experiment import (
    get_region_transitions,
    calc_mse,
    rejection_sample_reset,
    sample_dream_cell_action,
    cell_to_obs
)

from changepoint_detector_probe.page_hinkley import PageHinkleyDetector

changepoints_fired = 0
changepoint_log_history = {}

def run_experiment(layer_sizes: list[int] = [7, 64, 32, 2], sleep_enabled=False, gating_enabled=False, beta_slow=0.00005, beta_delta=0.1, threshold=0.015, gate_frac=0.5, num_eps_per_region=225, delta_tol=0.005, lambda_threshold=0.05):
    global changepoints_fired
    changepoints_fired = 0
    log_key = gate_frac if gating_enabled else ('sleep_only' if sleep_enabled else 'naive')
    changepoint_log_history[log_key] = []
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    
    dual_weight = GatedDualWeightPCN(layer_sizes=layer_sizes, beta_slow=beta_slow, beta_delta=beta_delta, 
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)
                                
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)
    
    gating_tracker = None
    if gating_enabled:
        gating_tracker = GatingLocalTracker(layer_sizes=dual_weight.fast.layer_sizes, gate_frac=gate_frac, overlap_frac=0.0, seed=42)
        gating_tracker.apply_gate(dual_weight.fast)
    
    cooldown_counter = 0
    protecting = False
    visit_snapshot = None
    
    ph_detector = PageHinkleyDetector(delta_tol=delta_tol, lambda_threshold=lambda_threshold)
    
    def run_episodes(num_eps, region):
        nonlocal cooldown_counter, protecting, visit_snapshot
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
                    
                    if cooldown_counter == 0 and ph_detector.update(smoothed_delta):
                        visit_snapshot = agent.tracker.visit_count.copy()
                        protecting = True
                        cooldown_counter = 100
                        
                        global changepoints_fired
                        changepoints_fired += 1
                        changepoint_log_history[log_key].append((region, ep, step))
                        if gating_enabled:
                            gating_tracker.on_changepoint()
                            gating_tracker.apply_gate(dual_weight.fast)
                        
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
                    
    # Train Region A
    run_episodes(num_eps_per_region, 'A')
    
    # Switch to Region B
    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(num_eps_per_region, 'B')
    
    # Switch to Region C
    env.goal = (0,4)
    agent.goal_obs = np.array([0.0, 1.0])
    run_episodes(num_eps_per_region, 'C')
    
    return changepoints_fired

if __name__ == '__main__':
    print("Running 4 scenarios from Fase 17 with Page-Hinkley detector...")
    
    # Configure delta_tol and lambda_threshold based on testing (suggested defaults: 0.005, 0.05)
    d_tol = 0.005
    l_thresh = 0.05
    
    cp_sleep_only = run_experiment(sleep_enabled=True, gating_enabled=False, delta_tol=d_tol, lambda_threshold=l_thresh)
    
    cp_gate_25 = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=0.25, delta_tol=d_tol, lambda_threshold=l_thresh)
    cp_gate_50 = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=0.50, delta_tol=d_tol, lambda_threshold=l_thresh)
    cp_gate_75 = run_experiment(sleep_enabled=True, gating_enabled=True, gate_frac=0.75, delta_tol=d_tol, lambda_threshold=l_thresh)
    
    print("\n--- COMPARATIVO DE CONTAGEM DE CHANGEPOINTS ---")
    print(f"{'Cenario':<15} | {'Original (Fase 17)':<20} | {'Novo (Page-Hinkley)':<20}")
    print("-" * 62)
    print(f"{'Sleep Only':<15} | {4:<20} | {cp_sleep_only:<20}")
    print(f"{'Gate=0.25':<15} | {12:<20} | {cp_gate_25:<20}")
    print(f"{'Gate=0.50':<15} | {4:<20} | {cp_gate_50:<20}")
    print(f"{'Gate=0.75':<15} | {5:<20} | {cp_gate_75:<20}")
    
    print("\nLogs detalhados:")
    for k, log in changepoint_log_history.items():
        print(f"{k}: {log}")
