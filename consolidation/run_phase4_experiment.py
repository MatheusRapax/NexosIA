import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcn_core.model import PredictiveCodingNetwork
from pcn_core.sleep import SleepConsolidator
from active_inference.environment import GridWorld

def get_transitions():
    env = GridWorld(size=5, goal=(4,4), seed=0)
    transitions_A = []
    transitions_B = []
    for r in range(5):
        for c in range(5):
            for a in range(5):
                env.row, env.col = r, c
                obs = np.array([r/4.0, c/4.0], dtype=np.float32)
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[a] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                next_obs, _, _ = env.step(a)
                
                if c in [0, 1]:
                    transitions_A.append((x_input, next_obs))
                elif c in [3, 4]:
                    transitions_B.append((x_input, next_obs))
    return transitions_A, transitions_B

def train_epochs(world_model, transitions, epochs=50, sleep_consolidator=None, rehearse=False):
    for ep in range(epochs):
        np.random.shuffle(transitions)
        for x_input, next_obs in transitions:
            energy = world_model.train_step(x_input, next_obs)
            if sleep_consolidator and not rehearse:
                obs_only = x_input[:2]
                sleep_consolidator.observe(obs_only, energy)
        
        if rehearse and sleep_consolidator:
            sleep_consolidator.rehearse(world_model)

def measure_mse(world_model, transitions):
    errors = []
    for x_input, next_obs in transitions:
        pred = world_model.predict(x_input)
        errors.append(np.sum((pred - next_obs)**2))
    return np.mean(errors)

def run():
    np.random.seed(42)
    transitions_A, transitions_B = get_transitions()
    
    print("=== CONDICAO SEM SONO ===")
    model_no_sleep = PredictiveCodingNetwork([7, 16, 8, 2], seed=0)
    train_epochs(model_no_sleep, transitions_A, 50)
    erro_A_pos_A = measure_mse(model_no_sleep, transitions_A)
    print(f"Erro em A apos treinar em A: {erro_A_pos_A:.5f}")
    
    train_epochs(model_no_sleep, transitions_B, 50)
    erro_A_pos_B_sem_sono = measure_mse(model_no_sleep, transitions_A)
    print(f"Erro em A apos treinar em B (sem sono): {erro_A_pos_B_sem_sono:.5f}")
    
    print("\n=== CONDICAO COM SONO ===")
    model_sleep = PredictiveCodingNetwork([7, 16, 8, 2], seed=0)
    sleep = SleepConsolidator(obs_dim=2, n_actions=5, n_dreams=20, prune_rate=0.01, seed=0)
    
    train_epochs(model_sleep, transitions_A, 50, sleep_consolidator=sleep, rehearse=False)
    sleep.sleep_cycle(model_sleep)
    
    train_epochs(model_sleep, transitions_B, 50, sleep_consolidator=sleep, rehearse=True)
    erro_A_pos_B_com_sono = measure_mse(model_sleep, transitions_A)
    print(f"Erro em A apos treinar em B (com sono): {erro_A_pos_B_com_sono:.5f}")
    
    reducao = (erro_A_pos_B_sem_sono - erro_A_pos_B_com_sono) / erro_A_pos_B_sem_sono
    print("\n=== RESULTADOS ===")
    print(f"Reducao relativa de erro: {reducao*100:.2f}%")

if __name__ == "__main__":
    run()
