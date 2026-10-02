import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'active_inference'))

from pcn_core.model import PredictiveCodingNetwork
from pcn_core.sleep import SleepConsolidator
from active_inference.environment import GridWorld
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker

def rejection_sample_reset(env, region='A'):
    while True:
        obs = env.reset()
        r = int(obs[0] * 4)
        c = int(obs[1] * 4)
        if region == 'A' and r in [0, 1] and c in [0, 1]:
            return obs
        elif region == 'B' and r in [3, 4] and c in [3, 4]:
            return obs

def measure_mse_region_A(world_model):
    env = GridWorld(size=5, goal=(4,4), seed=0)
    transitions_A = []
    for r in [0, 1]:
        for c in [0, 1]:
            for a in range(5):
                env.row, env.col = r, c
                obs = np.array([r/4.0, c/4.0], dtype=np.float32)
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[a] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                next_obs, _, _ = env.step(a)
                transitions_A.append((x_input, next_obs))
    
    errors = []
    for x_input, next_obs in transitions_A:
        pred = world_model.predict(x_input)
        errors.append(np.sum((pred - next_obs)**2))
    return np.mean(errors)

def run_experiment(sleep_enabled=False):
    np.random.seed(0)
    
    world_model = PredictiveCodingNetwork([7, 16, 8, 2], inference_steps=10, weight_lr=0.05, seed=0)
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    
    goal_A = np.array([0.0/4.0, 0.0/4.0], dtype=np.float32)
    agent = ActiveInferenceAgent(world_model, goal_obs=goal_A, tracker=tracker)
    
    sleep = None
    if sleep_enabled:
        sleep = SleepConsolidator(obs_dim=2, n_actions=5, n_dreams=50, prune_rate=0.01, seed=0)
        
    env = GridWorld(size=5, goal=(0, 0), seed=0)
    
    # FASE A (225 episodios, regiao A)
    successes_A = []
    
    for ep in range(1, 226):
        obs = rejection_sample_reset(env, region='A')
        done = False
        step_count = 0
        success = False
        
        while not done and step_count < 30:
            action = agent.select_action(obs, ep)
            next_obs, done, success = env.step(action)
            
            energy = agent.learn(obs, action, next_obs)
            
            if sleep_enabled:
                sleep.observe(obs, energy)
                
            obs = next_obs
            step_count += 1
                
        successes_A.append(success)
        
        if sleep_enabled and ep % 25 == 0:
            sleep.sleep_cycle(world_model)
            sleep.rehearse(world_model)
            
    taxa_A = np.mean(successes_A[-50:])
    mse_pos_A = measure_mse_region_A(world_model)
    
    # FASE B (225 episodios, regiao B, objetivo (4,4))
    env.goal = (4, 4)
    agent.goal_obs = np.array([4.0/4.0, 4.0/4.0], dtype=np.float32)
    
    for ep in range(1, 226):
        obs = rejection_sample_reset(env, region='B')
        done = False
        step_count = 0
        
        while not done and step_count < 30:
            action = agent.select_action(obs, ep)
            next_obs, done, success = env.step(action)
            
            energy = agent.learn(obs, action, next_obs)
            
            if sleep_enabled:
                sleep.observe(obs, energy)
                
            obs = next_obs
            step_count += 1
            
        if sleep_enabled and ep % 25 == 0:
            sleep.sleep_cycle(world_model)
            sleep.rehearse(world_model)
            
    mse_pos_B = measure_mse_region_A(world_model)
            
    # AVALIACAO FINAL (50 episodios, Regiao A, objetivo (0,0))
    env.goal = (0, 0)
    agent.goal_obs = np.array([0.0/4.0, 0.0/4.0], dtype=np.float32)
    
    eval_successes = []
    for ep in range(50):
        obs = rejection_sample_reset(env, region='A')
        done = False
        step_count = 0
        success = False
        
        while not done and step_count < 30:
            action = agent.select_action(obs, ep)
            next_obs, done, success = env.step(action)
            obs = next_obs
            step_count += 1
                
        eval_successes.append(success)
        
    taxa_eval = np.mean(eval_successes)
    
    return taxa_A, taxa_eval, mse_pos_A, mse_pos_B

def run_regression_test():
    np.random.seed(0)
    world_model = PredictiveCodingNetwork([7, 16, 8, 2], inference_steps=10, weight_lr=0.05, seed=0)
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    goal_A = np.array([0.0/4.0, 0.0/4.0], dtype=np.float32)
    agent = ActiveInferenceAgent(world_model, goal_obs=goal_A, tracker=tracker)
    
    env = GridWorld(size=5, goal=(0, 0), seed=0)
    successes = []
    
    for ep in range(250):
        obs = rejection_sample_reset(env, region='A')
        done = False
        step_count = 0
        success = False
        
        while step_count < 30 and not done:
            action = agent.select_action(obs, ep)
            next_obs, done, success = env.step(action)
            agent.learn(obs, action, next_obs)
            obs = next_obs
            step_count += 1
                
        successes.append(success)
        
    return np.mean(successes[-50:])

if __name__ == "__main__":
    print("Executando regressao (so Regiao A)...")
    taxa_regressao = run_regression_test()
    print(f"Taxa sucesso regressao (ultimos 50 ep): {taxa_regressao*100:.2f}%")
    
    print("\nExecutando CONDICAO SEM SONO...")
    taxa_A_sem, taxa_eval_sem, mse_A_sem, mse_B_sem = run_experiment(sleep_enabled=False)
    
    print("\nExecutando CONDICAO COM SONO...")
    taxa_A_com, taxa_eval_com, mse_A_com, mse_B_com = run_experiment(sleep_enabled=True)
    
    print("\n==================================================")
    print("RESULTADOS DA INTEGRAÇÃO (FASE 5 - DELTA 2)")
    print("==================================================")
    print(f"Taxa Fase A (antes de mudar):")
    print(f" - Sem Sono: {taxa_A_sem*100:.2f}%")
    print(f" - Com Sono: {taxa_A_com*100:.2f}%")
    print("\nTaxa Avaliacao Final (Regiao A, pos-Regiao B) [Apenas Contexto]:")
    print(f" - Sem Sono: {taxa_eval_sem*100:.2f}%")
    print(f" - Com Sono: {taxa_eval_com*100:.2f}%")
    
    print(f"\nMSE na Regiao A (Apos treinar na Fase A): {mse_A_sem:.5f}")
    print(f"MSE na Regiao A (Apos treinar na Fase B - SEM SONO): {mse_B_sem:.5f}")
    print(f"MSE na Regiao A (Apos treinar na Fase B - COM SONO): {mse_B_com:.5f}")
    
    reducao = (mse_B_sem - mse_B_com) / mse_B_sem
    print(f"\nReducao relativa de MSE: {reducao*100:.2f}%")
