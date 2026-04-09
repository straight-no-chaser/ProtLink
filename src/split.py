import numpy as np


def split_positive_edges(edges, num_nodes, val_ratio=0.1, test_ratio=0.1, seed=0):
    edges = np.asarray(edges, dtype=np.int64)
    num_edges = len(edges)
    rng = np.random.default_rng(seed)

    target_val = int(round(num_edges * val_ratio))
    target_test = int(round(num_edges * test_ratio))
    if target_val + target_test >= num_edges:
        raise ValueError("Validation and test split sizes leave no training edges.")

    train_degree = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edges:
        train_degree[u] += 1
        train_degree[v] += 1

    assignment = np.zeros(num_edges, dtype=np.int8)
    order = rng.permutation(num_edges)

    def fill_bucket(bucket_value, target_count):
        count = 0
        for edge_idx in order:
            if count >= target_count:
                break
            if assignment[edge_idx] != 0:
                continue
            u, v = edges[edge_idx]
            if train_degree[u] > 1 and train_degree[v] > 1:
                assignment[edge_idx] = bucket_value
                train_degree[u] -= 1
                train_degree[v] -= 1
                count += 1

        for edge_idx in order:
            if count >= target_count:
                break
            if assignment[edge_idx] != 0:
                continue
            u, v = edges[edge_idx]
            assignment[edge_idx] = bucket_value
            train_degree[u] -= 1
            train_degree[v] -= 1
            count += 1

    fill_bucket(1, target_val)
    fill_bucket(2, target_test)

    train_pos = edges[assignment == 0]
    val_pos = edges[assignment == 1]
    test_pos = edges[assignment == 2]

    return train_pos, val_pos, test_pos
