import numpy as np


def canonical_edge(u, v):
    return (u, v) if u < v else (v, u)


def edge_array_to_set(edges):
    if isinstance(edges, set):
        return {canonical_edge(int(u), int(v)) for u, v in edges}
    return {canonical_edge(int(u), int(v)) for u, v in np.asarray(edges)}


def sample_negative_edges(num_nodes, num_samples, positive_edges, rng=None, seed=None, forbidden_edges=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    positive_set = edge_array_to_set(positive_edges)
    forbidden_set = edge_array_to_set(forbidden_edges) if forbidden_edges is not None else set()

    max_edges = num_nodes * (num_nodes - 1) // 2
    if len(positive_set) + len(forbidden_set) + num_samples > max_edges:
        raise ValueError("Not enough candidate negative edges to sample from.")

    negatives = []
    seen = set()

    while len(negatives) < num_samples:
        u = int(rng.integers(num_nodes))
        v = int(rng.integers(num_nodes - 1))
        if v >= u:
            v += 1
        edge = canonical_edge(u, v)
        if edge in positive_set or edge in forbidden_set or edge in seen:
            continue
        negatives.append(edge)
        seen.add(edge)

    return np.asarray(negatives, dtype=np.int64)
