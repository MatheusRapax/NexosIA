import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pcn_core.model import PredictiveCodingNetwork
from learning_progress import LearningProgressTracker

class ActiveInferenceAgent:
    def __init__(self, world_model: PredictiveCodingNetwork, goal_obs: np.ndarray,
                 tracker: LearningProgressTracker = None, pragmatic_weight: float = 1.0, epistemic_weight: float = 1.0):
        self.world_model = world_model
        self.goal_obs = goal_obs
        # Fallback for Phase 2 test compatibility
        self.tracker = tracker if tracker is not None else LearningProgressTracker(n_cells=25, n_actions=5)
        self.pragmatic_weight = pragmatic_weight
        self.epistemic_weight = epistemic_weight
        
    def _cell_index(self, obs: np.ndarray) -> int:
        size = 5 # hardcoded for Phase 3 simplification
        row = round(obs[0] * (size - 1))
        col = round(obs[1] * (size - 1))
        return int(row * size + col)
        
    def select_action(self, obs: np.ndarray, episode: int) -> int:
        best_action = -1
        min_EFE = float('inf')
        
        cell = self._cell_index(obs)
        
        for a in range(5):
            a_onehot = np.zeros(5, dtype=np.float32)
            a_onehot[a] = 1.0
            x_input = np.concatenate([obs, a_onehot])
            
            # Prever o proximo estado baseado no modelo de mundo aprendido
            pred_next = self.world_model.predict(x_input)
            
            # Valor pragmatico: minimizar distancia para o objetivo (energia livre pragmatica)
            G = np.sum((pred_next - self.goal_obs)**2)
            
            # Valor epistemico
            LP = self.tracker.get_lp(cell, a)
            
            # Expected Free Energy
            EFE = self.pragmatic_weight * G - self.epistemic_weight * LP
            
            if EFE < min_EFE:
                min_EFE = EFE
                best_action = a
                
        return best_action
        
    def learn(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> float:
        a_onehot = np.zeros(5, dtype=np.float32)
        a_onehot[action] = 1.0
        x_input = np.concatenate([obs, a_onehot])
        
        # Treino on-line puro, sem replay buffer
        energy = self.world_model.train_step(x_input, next_obs)
        
        # Atualiza o rastreador de progresso
        cell = self._cell_index(obs)
        self.tracker.update(cell, action, float(energy))
        
        return energy
