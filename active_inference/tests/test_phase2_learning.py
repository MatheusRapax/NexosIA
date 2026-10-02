import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcn_core.model import PredictiveCodingNetwork
from environment import GridWorld
from agent import ActiveInferenceAgent
from random_baseline import RandomAgent

def test_imports_pcn_from_core():
    # Verifica que o agent.py importa a classe da Fase 1 e nao reimplementa
    agent_file = os.path.join(os.path.dirname(__file__), '..', 'agent.py')
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "from pcn_core.model import PredictiveCodingNetwork" in content, "Faltou importar PCN da Fase 1"
    assert "class PredictiveCodingNetwork" not in content, "PCN foi reimplementada, nao pode"

def test_no_autograd():
    # Verifica agent.py e environment.py
    for filename in ['agent.py', 'environment.py', 'random_baseline.py']:
        filepath = os.path.join(os.path.dirname(__file__), '..', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "autograd" not in content.lower()
        assert "backward" not in content.lower()

def test_no_buffer_used():
    # Verifica que o agente so passa (obs, acao, next_obs) sem listas
    agent_file = os.path.join(os.path.dirname(__file__), '..', 'agent.py')
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if there's any append that is not sys.path.append
    lines = content.split('\n')
    for line in lines:
        if "append" in line and "sys.path.append" not in line:
            assert False, "Não usar buffers (append)"
    
def test_learning_success_criteria():
    np.random.seed(42)
    n_episodes = 250
    max_steps = 30
    goal = (4, 4)
    goal_obs = np.array([goal[0]/4.0, goal[1]/4.0], dtype=np.float32)
    
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
        
    pcn_inicio = np.mean(pcn_success[:50])
    pcn_fim = np.mean(pcn_success[-50:])
    rand_fim = np.mean(rand_success[-50:])
    
    energia_inicio = np.mean(pcn_energy[:50])
    energia_fim = np.mean(pcn_energy[-50:])
    
    assert energia_fim <= 0.5 * energia_inicio, f"Energia não caiu pela metade: início {energia_inicio:.4f}, fim {energia_fim:.4f}"
    assert pcn_fim >= rand_fim + 0.20, f"Vantagem contra random falhou: {pcn_fim} vs {rand_fim}"
