import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from src.features import build_pair_features


class PairMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class DotDecoder(nn.Module):
    def forward(self, z, edges):
        return (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)


class PairMLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.mlp = PairMLP(latent_dim * 4, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, z, edges):
        features = build_pair_features(z, edges)
        return self.mlp(features)
