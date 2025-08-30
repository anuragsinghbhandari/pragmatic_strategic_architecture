# models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionDeceptionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):   # now accepts single vector
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


