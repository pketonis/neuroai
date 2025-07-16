# models/sparse_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SparseMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a1 = self.relu(self.fc1(x))
        a2 = self.relu(self.fc2(a1))
        out = self.fc3(a2)
        return out, [a1, a2]
