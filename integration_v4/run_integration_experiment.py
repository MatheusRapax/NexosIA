import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))
import numpy as np

from dual_weight.dual_weight_pcn import DualWeightPCN
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker
from active_inference.environment import GridWorld

def sample_dream_cell_action(visit_snapshot: np.ndarray, rng, alpha: float = 0.5, epsilon: float = 1.0) -> tuple[int, int]:
    adjusted = (visit_snapshot.astype(float) + epsilon) ** alpha
    total = np.sum(adjusted)
    flat_probs = adjusted.flatten() / total
    
    n_cells, n_actions = visit_snapshot.shape
    flat_index = rng.choice(n_cells * n_actions, p=flat_probs)
    cell = int(flat_index // n_actions)
    action = int(flat_index % n_actions)
    return cell, action

def cell_to_obs(cell: int, size: int = 5) -> np.ndarray:
    row = cell // size
    col = cell % size
    return np.array([row / (size - 1), col / (size - 1)])

def get_region_transitions(env, region):
    transitions = []
    if region == 'A':
        cells = [(0,0), (0,1), (1,0), (1,1)]
    elif region == 'B':
        cells = [(3,3), (3,4), (4,3), (4,4)]
    elif region == 'C':
        cells = [(0,3), (0,4), (1,3), (1,4)]
        
    for r, c in cells:
        for a in range(5):
            env.row = r
            env.col = c
            obs = env._get_obs()
            
            saved_r, saved_c = env.row, env.col
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

def run_experiment(sleep_enabled=False, beta_slow=0.00005, beta_delta=0.1, threshold=0.015):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    
    dual_weight = DualWeightPCN(layer_sizes=[7, 16, 8, 2], beta_slow=beta_slow, beta_delta=beta_delta, 
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)
                                
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)
    
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
                    
                if sleep_enabled:
                    smoothed_delta = res['smoothed_delta']
                    if smoothed_delta > threshold and not was_above_threshold and cooldown_counter == 0:
                        # UPDATE SNAPSHOT ON EVERY CHANGEOVER RISING EDGE
                        visit_snapshot = agent.tracker.visit_count.copy()
                        protecting = True
                        was_above_threshold = True
                        cooldown_counter = 100  # Debounce
                    elif smoothed_delta <= threshold:
                        was_above_threshold = False
                        
                    if protecting and visit_snapshot is not None:
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
                    
    # Train Region A
    run_episodes(225, 'A')
    erro_A_pos_A = calc_mse(dual_weight.fast, transitions_A)
    
    # Switch to Region B
    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(225, 'B')
    
    erro_A_pos_B = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_B = calc_mse(dual_weight.fast, transitions_B)
    
    # Switch to Region C
    env.goal = (0,4)
    agent.goal_obs = np.array([0.0, 1.0])
    run_episodes(225, 'C')
    
    erro_A_pos_C = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_C = calc_mse(dual_weight.fast, transitions_B)
    
    return erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C

if __name__ == '__main__':
    print("Running Baseline (Sem Sono)...")
    sem_A_pos_A, sem_A_pos_B, sem_B_pos_B, sem_A_pos_C, sem_B_pos_C = run_experiment(sleep_enabled=False)
    
    print("\nRunning Experiment (Com Sono - Mecanismo A Suavizado)...")
    com_A_pos_A, com_A_pos_B, com_B_pos_B, com_A_pos_C, com_B_pos_C = run_experiment(sleep_enabled=True)
    
    print("\n--- RESULTADOS ---")
    print(f"MSE A pos A: {sem_A_pos_A:.4f}")
    
    print(f"\n[Sem Sono]")
    print(f"MSE A pos B: {sem_A_pos_B:.4f}")
    print(f"MSE B pos B: {sem_B_pos_B:.4f}")
    print(f"MSE A pos C: {sem_A_pos_C:.4f}")
    print(f"MSE B pos C: {sem_B_pos_C:.4f}")
    
    print(f"\n[Com Sono]")
    print(f"MSE A pos B: {com_A_pos_B:.4f}")
    print(f"MSE B pos B: {com_B_pos_B:.4f}")
    print(f"MSE A pos C: {com_A_pos_C:.4f}")
    print(f"MSE B pos C: {com_B_pos_C:.4f}")
    
    reducao_A_apos_B = (sem_A_pos_B - com_A_pos_B) / sem_A_pos_B
    reducao_A_apos_C = (sem_A_pos_C - com_A_pos_C) / sem_A_pos_C
    reducao_B_apos_C = (sem_B_pos_C - com_B_pos_C) / sem_B_pos_C
    
    print(f"\n--- REDUCOES ---")
    print(f"Reducao em A apos B: {reducao_A_apos_B:.2%} (Checagem de consistencia, ref ~64.9%)")
    print(f"Reducao em A apos C: {reducao_A_apos_C:.2%} (Sobrevivencia da 1a tarefa apos 2 transicoes)")
    print(f"Reducao em B apos C: {reducao_B_apos_C:.2%} (Protecao da 2a tarefa)")
