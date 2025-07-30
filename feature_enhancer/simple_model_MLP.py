import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPResidualHead(nn.Module):
    def __init__(self, dim=256, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return residual + x
