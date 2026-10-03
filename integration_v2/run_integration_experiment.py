import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from dual_weight.dual_weight_pcn import DualWeightPCN
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker
from active_inference.environment import GridWorld

def get_region_a_transitions(env):
    transitions = []
    cells = [(0,0), (0,1), (1,0), (1,1)]
    for r, c in cells:
        for a in range(5):
            env.row = r
            env.col = c
            obs = env._get_obs()
            
            # Save state
            saved_r, saved_c = env.row, env.col
            next_obs, _, _ = env.step(a)
            
            transitions.append((obs, a, next_obs))
            
            # Restore state
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

def eval_success_rate(agent, env, region, episodes=50):
    successes = 0
    for _ in range(episodes):
        obs = rejection_sample_reset(env, region)
        for _ in range(30):
            a = agent.select_action(obs, 0)
            obs, done, success = env.step(a)
            if success:
                successes += 1
                break
    return successes / episodes

def run_experiment(sleep_enabled=False, beta_slow=0.00005, beta_delta=0.1, threshold=0.015):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_a_transitions(env)
    
    dual_weight = DualWeightPCN(layer_sizes=[7, 16, 8, 2], beta_slow=beta_slow, beta_delta=beta_delta, 
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)
                                
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)
    
    obs_mean = np.zeros(2)
    obs_std = np.zeros(2)
    alpha_obs = 0.02
    was_above_threshold = False
    cooldown_counter = 0
    protecting = False
    
    def run_episodes(num_eps, region):
        nonlocal obs_mean, obs_std, was_above_threshold, cooldown_counter, protecting
        for ep in range(num_eps):
            obs = rejection_sample_reset(env, region)
            for step in range(30):
                action = agent.select_action(obs, ep)
                next_obs, done, success = env.step(action)
                
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[action] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                
                res = dual_weight.step(x_input, next_obs)
                
                cell = round(obs[0]*4)*5 + round(obs[1]*4)
                agent.tracker.update(cell, action, res['energia_fast_train_step'])
                
                # update obs stats
                obs_mean = alpha_obs * obs + (1 - alpha_obs) * obs_mean
                # basic std deviation tracking approximation
                obs_std = alpha_obs * np.abs(obs - obs_mean) + (1 - alpha_obs) * obs_std
                
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    
                if sleep_enabled:
                    smoothed_delta = res['smoothed_delta']
                    if smoothed_delta > threshold and not was_above_threshold and cooldown_counter == 0:
                        protecting = True
                        was_above_threshold = True
                        cooldown_counter = 100  # Debounce
                    elif smoothed_delta <= threshold:
                        was_above_threshold = False
                        
                    if protecting:
                        for _ in range(4):
                            obs_dream = np.clip(obs_mean + obs_std * np.random.randn(2), 0, 1)
                            action_dream = np.random.randint(0, 5)
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
    succ_A_pos_A = eval_success_rate(agent, env, 'A')
    
    # Switch to Region B
    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(225, 'B')
    
    erro_A_pos_B = calc_mse(dual_weight.fast, transitions_A)
    succ_A_pos_B = eval_success_rate(agent, env, 'A')
    
    return erro_A_pos_A, erro_A_pos_B, succ_A_pos_A, succ_A_pos_B

if __name__ == '__main__':
    print("Running Baseline (Sem Sono)...")
    erro_A_pos_A_sem, erro_A_pos_B_sem, succ_A_pos_A_sem, succ_A_pos_B_sem = run_experiment(sleep_enabled=False)
    
    print("Running Experiment (Com Sono)...")
    # Will calibrate threshold later if needed
    erro_A_pos_A_com, erro_A_pos_B_com, succ_A_pos_A_com, succ_A_pos_B_com = run_experiment(sleep_enabled=True, threshold=0.015)
    
    print("\n--- RESULTADOS ---")
    print(f"MSE A pos A: {erro_A_pos_A_sem:.4f}")
    print(f"MSE A pos B (sem sono): {erro_A_pos_B_sem:.4f}")
    print(f"MSE A pos B (com sono): {erro_A_pos_B_com:.4f}")
    
    if erro_A_pos_B_sem > erro_A_pos_A_sem:
        print("Esquecimento catastrofico CONFIRMADO na baseline.")
    else:
        print("ALERTA: Esquecimento nao ocorreu na baseline!")
        
    reducao = (erro_A_pos_B_sem - erro_A_pos_B_com) / erro_A_pos_B_sem
    print(f"Reducao relativa de erro: {reducao:.2%}")
    
    print(f"\nTaxa Sucesso Regiao A (antes de B): {succ_A_pos_A_sem:.2%}")
    print(f"Taxa Sucesso Regiao A (pos B, sem sono): {succ_A_pos_B_sem:.2%}")
    print(f"Taxa Sucesso Regiao A (pos B, com sono): {succ_A_pos_B_com:.2%}")
