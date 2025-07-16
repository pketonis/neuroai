# models/gcn_masked.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNMasked(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_mask=None):
        if edge_mask is not None:
            edge_index = edge_index[:, edge_mask]
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
