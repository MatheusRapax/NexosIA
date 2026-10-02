import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_progress import LearningProgressTracker
from pcn_core.model import PredictiveCodingNetwork
from environment import GridWorld
from agent import ActiveInferenceAgent
from random_baseline import RandomAgent

def test_lp_detects_real_progress():
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    for t in range(30):
        error_t = 1.0 * (0.9 ** t)
        tracker.update(cell_index=0, action=0, error=error_t)
    lp = tracker.get_lp(0, 0)
    assert lp > 0.01, f"LP should be positive for real progress, got {lp}"

def test_lp_resists_irreducible_noise():
    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    np.random.seed(42)
    for t in range(30):
        error_t = 1.0 + np.random.normal(0, 0.05)
        tracker.update(cell_index=1, action=0, error=error_t)
    lp = tracker.get_lp(1, 0)
    assert abs(lp) < 0.05, f"LP should be near 0 for irreducible noise, got {lp}"

def test_lp_bootstrap_untouched_pair():
    tracker = LearningProgressTracker(n_cells=25, n_actions=5, lp_init=1.0)
    lp = tracker.get_lp(5, 2)
    assert lp == 1.0, f"LP for untouched should be exactly lp_init (1.0), got {lp}"

def test_select_action_no_randomness():
    agent_file = os.path.join(os.path.dirname(__file__), '..', 'agent.py')
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # Pega o corpo da funcao select_action
    func_start = content.find('def select_action(')
    func_end = content.find('def learn(')
    select_action_body = content[func_start:func_end]
    
    assert "np.random" not in select_action_body, "select_action must not use random"
    assert "random." not in select_action_body, "select_action must not use random"

def test_phase3_regression_vs_random():
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
