import math

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, SAGEConv, TransformerConv

from src.data_utils import build_pair_features


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
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2, residual="none"):
        super().__init__()
        if residual not in {"none", "concat", "add"}:
            raise ValueError(f"Unsupported residual mode: {residual}")
        if residual == "add" and input_dim != hidden_dim:
            raise ValueError("Residual mode 'add' requires input_dim == hidden_dim.")
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.output_dim = hidden_dim if residual != "concat" else hidden_dim + input_dim

    def forward(self, x, edge_index):
        x_input = x
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        if self.residual == "concat":
            return torch.cat([x, x_input], dim=-1)
        if self.residual == "add":
            return x + x_input
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, heads=4, layers=2, dropout=0.2, residual="none"):
        super().__init__()
        if residual not in {"none", "concat", "add"}:
            raise ValueError(f"Unsupported residual mode: {residual}")
        if residual == "add" and input_dim != hidden_dim:
            raise ValueError("Residual mode 'add' requires input_dim == hidden_dim.")

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = input_dim
        for _ in range(layers):
            self.layers.append(
                TransformerConv(
                    current_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.output_dim = hidden_dim if residual != "concat" else hidden_dim + input_dim

    def forward(self, x, edge_index):
        x_input = x
        h = x
        for layer, norm in zip(self.layers, self.norms):
            out = layer(h, edge_index)
            out = norm(out)
            out = torch.relu(out)
            out = self.dropout(out)
            if out.shape[-1] == h.shape[-1]:
                h = out + h
            else:
                h = out
        if self.residual == "concat":
            return torch.cat([h, x_input], dim=-1)
        if self.residual == "add":
            return h + x_input
        return h


class HGTEncoder(nn.Module):
    def __init__(
        self,
        protein_input_dim,
        hidden_dim=256,
        heads=4,
        layers=2,
        dropout=0.2,
        metadata=None,
        num_pathways=0,
        num_domains=0,
        node_type_counts=None,
    ):
        super().__init__()
        if metadata is None:
            raise ValueError()

        node_types, _ = metadata
        if node_type_counts is None:
            node_type_counts = {}
            if "pathway" in node_types:
                node_type_counts["pathway"] = num_pathways
            if "domain" in node_types:
                node_type_counts["domain"] = num_domains

        self.protein_proj = nn.Linear(protein_input_dim, hidden_dim)
        self.node_type_embeddings = nn.ModuleDict()
        for node_type, count in sorted(node_type_counts.items()):
            if node_type == "protein":
                continue
            self.node_type_embeddings[node_type] = nn.Embedding(max(1, count), hidden_dim)
        self.convs = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

        for module in self.node_type_embeddings.values():
            if not isinstance(module, nn.Embedding):
                raise TypeError(f"Expected nn.Embedding, got {type(module).__name__}")
            nn.init.normal_(module.weight, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, data):
        x_dict = {"protein": self.protein_proj(data["protein"].x)}
        for node_type, embedding in self.node_type_embeddings.items():
            x_dict[node_type] = embedding(data[node_type].node_id)

        for conv, norm in zip(self.convs, self.norms):
            out_dict = conv(x_dict, data.edge_index_dict)
            next_x_dict = {}
            for node_type, out in out_dict.items():
                out = norm(out)
                out = torch.relu(out)
                out = self.dropout(out)
                if out.shape[-1] == x_dict[node_type].shape[-1]:
                    out = out + x_dict[node_type]
                next_x_dict[node_type] = out
            x_dict = next_x_dict

        return x_dict["protein"]


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
