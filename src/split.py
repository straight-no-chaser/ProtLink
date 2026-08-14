import numpy as np


def _assign_edge_split(edges, num_nodes, target_val, target_test, order):
    train_degree = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edges:
        train_degree[u] += 1
        train_degree[v] += 1

    assignment = np.zeros(len(edges), dtype=np.int8)

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


def _build_node_disjoint_order(edges, num_nodes, target_eval_edges, rng, candidate_nodes=None):
    if target_eval_edges <= 0:
        return rng.permutation(len(edges))

    incident_edges = [[] for _ in range(num_nodes)]
    for edge_idx, (u, v) in enumerate(edges):
        incident_edges[int(u)].append(edge_idx)
        incident_edges[int(v)].append(edge_idx)

    candidate_mask = np.zeros(len(edges), dtype=bool)
    candidate_count = 0

    if candidate_nodes is None:
        node_order = rng.permutation(num_nodes)
    else:
        node_order = np.asarray(candidate_nodes, dtype=np.int64)
        node_order = node_order[rng.permutation(len(node_order))]

    for node in node_order:
        for edge_idx in incident_edges[int(node)]:
            if not candidate_mask[edge_idx]:
                candidate_mask[edge_idx] = True
                candidate_count += 1
        if candidate_count >= target_eval_edges:
            break

    preferred = np.where(candidate_mask)[0]
    remaining = np.where(~candidate_mask)[0]
    rng.shuffle(preferred)
    rng.shuffle(remaining)
    return np.concatenate([preferred, remaining], axis=0)


def _candidate_nodes(num_nodes, candidate_nodes):
    if candidate_nodes is None:
        return np.arange(num_nodes, dtype=np.int64)
    nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
    nodes = np.unique(nodes)
    if len(nodes) == 0 or np.any(nodes < 0) or np.any(nodes >= num_nodes):
        raise ValueError("candidate_nodes indices are outside [0, num_nodes)")
    return nodes


def _node_target_count(num_eligible, ratio):
    count = int(round(num_eligible * ratio))
    if ratio > 0 and count == 0 and num_eligible >= 3:
        count = 1
    return count


def _edge_array(edge_list):
    if not edge_list:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edge_list, dtype=np.int64)


def _has_train_neighbor(node, neighbors, train_nodes):
    return any(neighbor in train_nodes for neighbor in neighbors[node])


def _validate_node_split(train_pos, val_pos, test_pos, train_nodes, val_nodes, test_nodes):
    if train_nodes.intersection(val_nodes) or train_nodes.intersection(test_nodes) or val_nodes.intersection(test_nodes):
        raise ValueError("node_inductive split produced overlapping node sets")

    for u, v in train_pos:
        if int(u) not in train_nodes or int(v) not in train_nodes:
            raise ValueError("node_inductive train_pos contains a held-out node")

    for u, v in val_pos:
        u = int(u)
        v = int(v)
        if not ((u in val_nodes and v in train_nodes) or (v in val_nodes and u in train_nodes)):
            raise ValueError("node_inductive val_pos must connect validation nodes to train nodes")

    for u, v in test_pos:
        u = int(u)
        v = int(v)
        if not ((u in test_nodes and v in train_nodes) or (v in test_nodes and u in train_nodes)):
            raise ValueError("node_inductive test_pos must connect test nodes to train nodes")

    seen = set()
    for bucket_name, bucket in [("train_pos", train_pos), ("val_pos", val_pos), ("test_pos", test_pos)]:
        for u, v in bucket:
            edge = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            if edge in seen:
                raise ValueError(f"node_inductive positive edge appears in multiple splits: {bucket_name}")
            seen.add(edge)


def split_nodes(edges, num_nodes, val_ratio=0.1, test_ratio=0.1, seed=0, candidate_nodes=None):
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) == 0:
        raise ValueError()

    rng = np.random.default_rng(seed)
    active_nodes = _candidate_nodes(num_nodes, candidate_nodes)
    active_node_set = set(int(node) for node in active_nodes)

    neighbors = {int(node): set() for node in active_nodes}
    usable_edges = []
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v or u not in active_node_set or v not in active_node_set:
            continue
        usable_edges.append((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)

    eligible_nodes = np.asarray([node for node in active_nodes if neighbors[int(node)]], dtype=np.int64)
    if len(eligible_nodes) < 3:
        raise ValueError("node_inductive split requires at least three non-isolated eligible nodes")

    target_val = _node_target_count(len(eligible_nodes), val_ratio)
    target_test = _node_target_count(len(eligible_nodes), test_ratio)
    while target_val + target_test >= len(eligible_nodes):
        if target_test >= target_val and target_test > 0:
            target_test -= 1
        elif target_val > 0:
            target_val -= 1
        else:
            break

    if val_ratio > 0 and target_val == 0:
        raise ValueError("node_inductive could not allocate validation nodes with the requested ratio")
    if test_ratio > 0 and target_test == 0:
        raise ValueError("node_inductive could not allocate test nodes with the requested ratio")

    order = eligible_nodes[rng.permutation(len(eligible_nodes))]
    val_nodes = set(int(node) for node in order[:target_val])
    test_nodes = set(int(node) for node in order[target_val : target_val + target_test])

    while True:
        train_nodes = active_node_set.difference(val_nodes).difference(test_nodes)
        kept_val_nodes = {node for node in val_nodes if _has_train_neighbor(node, neighbors, train_nodes)}
        kept_test_nodes = {node for node in test_nodes if _has_train_neighbor(node, neighbors, train_nodes)}
        if kept_val_nodes == val_nodes and kept_test_nodes == test_nodes:
            break
        val_nodes = kept_val_nodes
        test_nodes = kept_test_nodes

    train_nodes = active_node_set.difference(val_nodes).difference(test_nodes)
    train_pos = []
    val_pos = []
    test_pos = []

    for u, v in usable_edges:
        if u in train_nodes and v in train_nodes:
            train_pos.append((u, v))
        elif (u in val_nodes and v in train_nodes) or (v in val_nodes and u in train_nodes):
            val_pos.append((u, v))
        elif (u in test_nodes and v in train_nodes) or (v in test_nodes and u in train_nodes):
            test_pos.append((u, v))

    train_pos = _edge_array(train_pos)
    val_pos = _edge_array(val_pos)
    test_pos = _edge_array(test_pos)

    if val_ratio > 0 and len(val_pos) == 0:
        raise ValueError("node_inductive split produced no validation positive edges")
    if test_ratio > 0 and len(test_pos) == 0:
        raise ValueError("node_inductive split produced no test positive edges")
    if len(train_pos) == 0:
        raise ValueError("node_inductive split produced no training positive edges")

    _validate_node_split(train_pos, val_pos, test_pos, train_nodes, val_nodes, test_nodes)

    return {
        "train_pos": train_pos,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "train_nodes": np.asarray(sorted(train_nodes), dtype=np.int64),
        "val_nodes": np.asarray(sorted(val_nodes), dtype=np.int64),
        "test_nodes": np.asarray(sorted(test_nodes), dtype=np.int64),
    }


def split_positive_edges(edges, num_nodes, val_ratio=0.1, test_ratio=0.1, seed=0, mode="edge_random", candidate_nodes=None):
    edges = np.asarray(edges, dtype=np.int64)
    if mode == "node_inductive":
        return split_nodes(
            edges,
            num_nodes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            candidate_nodes=candidate_nodes,
        )

    num_edges = len(edges)
    rng = np.random.default_rng(seed)

    target_val = int(round(num_edges * val_ratio))
    target_test = int(round(num_edges * test_ratio))
    if target_val + target_test >= num_edges:
        raise ValueError()

    if mode == "edge_random":
        order = rng.permutation(num_edges)
        return _assign_edge_split(edges, num_nodes, target_val, target_test, order)

    if mode == "node_disjoint":
        order = _build_node_disjoint_order(edges, num_nodes, target_val + target_test, rng, candidate_nodes=candidate_nodes)
        return _assign_edge_split(edges, num_nodes, target_val, target_test, order)

    raise ValueError(f"Unsupported split mode: {mode}")
