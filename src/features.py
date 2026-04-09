import torch


def build_pair_features(x, edges):
    x_u = x[edges[:, 0]]
    x_v = x[edges[:, 1]]
    return torch.cat([x_u, x_v, torch.abs(x_u - x_v), x_u * x_v], dim=-1)
