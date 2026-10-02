import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BackpropMLP:
    def __init__(self, layer_sizes: list[int], lr: float = 0.01, seed: int = 0):
        torch.manual_seed(seed)
        
        layers = []
        for l in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[l-1], layer_sizes[l]))
            if l < len(layer_sizes) - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Tanh()) # Ensure the output matches the tanh activation of PCN
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='sum') # MSE matches the 0.5 * e^2 energy formula conceptually
        
        # We need to scale the loss by 0.5 to exactly match the energy formula of PCN
        
    def train_step(self, x_input: np.ndarray, y_onehot: np.ndarray) -> float:
        x_tensor = torch.FloatTensor(x_input).unsqueeze(0)
        y_tensor = torch.FloatTensor(y_onehot).unsqueeze(0)
        
        self.optimizer.zero_grad()
        output = self.model(x_tensor)
        
        # Energy = sum 0.5 * e^2, MSELoss computes mean, so we just manually compute energy
        loss = 0.5 * torch.sum((output - y_tensor) ** 2)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        x_tensor = torch.FloatTensor(x_input).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x_tensor)
        return output.squeeze(0).numpy()
