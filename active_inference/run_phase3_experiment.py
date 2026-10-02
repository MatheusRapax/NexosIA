import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pcn_core.model import PredictiveCodingNetwork
from environment import GridWorld
from agent import ActiveInferenceAgent
from random_baseline import RandomAgent

def run_experiment():
    n_episodes = 250
    max_steps = 30
    
    goal = (4, 4)
    goal_obs = np.array([goal[0]/4.0, goal[1]/4.0], dtype=np.float32)
    
    print("Treinando Active Inference Agent (Phase 3 com Learning Progress)...")
    env = GridWorld(size=5, goal=goal, seed=0)
    world_model = PredictiveCodingNetwork(layer_sizes=[7, 16, 8, 2], inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=0)
    
    agent = ActiveInferenceAgent(world_model=world_model, goal_obs=goal_obs)
    
    pcn_success = []
    pcn_energy = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        success = False
        ep_energy = []
        
        while steps < max_steps and not done:
            action = agent.select_action(obs, ep)
            next_obs, done, success = env.step(action)
            energy = agent.learn(obs, action, next_obs)
            ep_energy.append(energy)
            obs = next_obs
            steps += 1
            
        pcn_success.append(success)
        pcn_energy.append(np.mean(ep_energy) if ep_energy else 0.0)
        if (ep + 1) % 50 == 0:
            print(f"Episodio {ep+1}/{n_episodes} - Sucesso recente: {np.mean(pcn_success[-50:])*100:.1f}%")
            
    print("\nTestando Random Agent...")
    env_rand = GridWorld(size=5, goal=goal, seed=0)
    agent_rand = RandomAgent()
    
    rand_success = []
    for ep in range(n_episodes):
        obs = env_rand.reset()
        done = False
        steps = 0
        success = False
        
        while steps < max_steps and not done:
            action = agent_rand.select_action(obs)
            next_obs, done, success = env_rand.step(action)
            obs = next_obs
            steps += 1
            
        rand_success.append(success)

    pcn_inicio = np.mean(pcn_success[:50]) * 100
    pcn_fim = np.mean(pcn_success[-50:]) * 100
    rand_inicio = np.mean(rand_success[:50]) * 100
    rand_fim = np.mean(rand_success[-50:]) * 100
    
    energia_inicio = np.mean(pcn_energy[:50])
    energia_fim = np.mean(pcn_energy[-50:])
    
    print("\n==================================================")
    print("RESULTADOS DA FASE 3: LEARNING PROGRESS")
    print("==================================================")
    print(f"Agente          | 50 Ep. Iniciais | 50 Ep. Finais")
    print("-" * 50)
    print(f"Random          | {rand_inicio:13.2f}%  | {rand_fim:11.2f}%")
    print(f"PCN-AIF (P3)    | {pcn_inicio:13.2f}%  | {pcn_fim:11.2f}%")
    print("-" * 50)
    print(f"Energia World Model (Inicio): {energia_inicio:.5f}")
    print(f"Energia World Model (Fim):    {energia_fim:.5f}")
    diff_agente = pcn_fim - rand_fim
    print(f"Vantagem final vs Random:     {diff_agente:+.2f} pp")

if __name__ == "__main__":
    np.random.seed(42)
    run_experiment()
