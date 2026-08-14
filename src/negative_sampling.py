import numpy as np

from src.homo_graph_utils import canonical_edge


def edge_array_to_set(edges):
    if isinstance(edges, set):
        return {canonical_edge(int(u), int(v)) for u, v in edges}
    edges = np.asarray(edges)
    if edges.size == 0:
        return set()
    return {canonical_edge(int(u), int(v)) for u, v in edges}


def _candidate_node_array(num_nodes, candidate_nodes):
    if candidate_nodes is None:
        return None

    candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
    candidate_nodes = np.unique(candidate_nodes)
    if len(candidate_nodes) == 0:
        return candidate_nodes
    if np.any(candidate_nodes < 0) or np.any(candidate_nodes >= num_nodes):
        raise ValueError()
    return candidate_nodes


def _active_node_set(num_nodes, candidate_nodes):
    if candidate_nodes is None:
        return set(range(num_nodes))
    return set(int(node) for node in candidate_nodes)


def _resolve_candidate_sets(num_nodes, candidate_nodes=None, left_candidate_nodes=None, right_candidate_nodes=None):
    if left_candidate_nodes is None and right_candidate_nodes is None:
        candidate_nodes = _candidate_node_array(num_nodes, candidate_nodes)
        active_nodes = _active_node_set(num_nodes, candidate_nodes)
        return candidate_nodes, candidate_nodes, active_nodes, active_nodes

    if left_candidate_nodes is None:
        left_candidate_nodes = candidate_nodes
    if right_candidate_nodes is None:
        right_candidate_nodes = candidate_nodes

    left_candidate_nodes = _candidate_node_array(num_nodes, left_candidate_nodes)
    right_candidate_nodes = _candidate_node_array(num_nodes, right_candidate_nodes)
    left_nodes = _active_node_set(num_nodes, left_candidate_nodes)
    right_nodes = _active_node_set(num_nodes, right_candidate_nodes)
    return left_candidate_nodes, right_candidate_nodes, left_nodes, right_nodes


def _candidate_edge_set(edges, active_nodes):
    blocked = set()
    for u, v in edge_array_to_set(edges):
        if u == v:
            continue
        if u in active_nodes and v in active_nodes:
            blocked.add((u, v))
    return blocked


def _candidate_pair_edge_set(edges, left_nodes, right_nodes):
    blocked = set()
    for u, v in edge_array_to_set(edges):
        if u == v:
            continue
        if (u in left_nodes and v in right_nodes) or (v in left_nodes and u in right_nodes):
            blocked.add((u, v))
    return blocked


def _candidate_pair_capacity(left_nodes, right_nodes):
    overlap = len(left_nodes.intersection(right_nodes))
    return len(left_nodes) * len(right_nodes) - overlap - (overlap * (overlap - 1) // 2)


def _enumerate_candidate_pairs(left_nodes, right_nodes, blocked_set):
    pairs = []
    for u in sorted(left_nodes):
        for v in sorted(right_nodes):
            if u == v:
                continue
            edge = canonical_edge(u, v)
            if edge in blocked_set:
                continue
            pairs.append(edge)
    return np.asarray(sorted(set(pairs)), dtype=np.int64)


def _sample_random_negative_edges(
    num_nodes,
    num_samples,
    positive_set,
    rng,
    forbidden_set,
    candidate_nodes=None,
    left_candidate_nodes=None,
    right_candidate_nodes=None,
):
    if num_samples == 0:
        return np.zeros((0, 2), dtype=np.int64)

    asymmetric_input = left_candidate_nodes is not None or right_candidate_nodes is not None
    left_candidate_nodes, right_candidate_nodes, left_nodes, right_nodes = _resolve_candidate_sets(
        num_nodes,
        candidate_nodes=candidate_nodes,
        left_candidate_nodes=left_candidate_nodes,
        right_candidate_nodes=right_candidate_nodes,
    )
    if not left_nodes or not right_nodes:
        raise ValueError("Negative sampling requires non-empty candidate node sets.")
    if len(left_nodes.union(right_nodes)) < 2:
        raise ValueError("Negative sampling requires at least 2 distinct candidate nodes.")

    blocked_set = _candidate_pair_edge_set(positive_set, left_nodes, right_nodes).union(
        _candidate_pair_edge_set(forbidden_set, left_nodes, right_nodes)
    )
    max_edges = _candidate_pair_capacity(left_nodes, right_nodes)
    if len(blocked_set) + num_samples > max_edges:
        raise ValueError("Requested more negative edges than the candidate space can provide.")

    negatives = []
    seen = set()
    attempts = 0
    max_attempts = max(1000, num_samples * 100)

    while len(negatives) < num_samples:
        attempts += 1
        if attempts > max_attempts:
            available = _enumerate_candidate_pairs(left_nodes, right_nodes, blocked_set.union(seen))
            if len(available) < num_samples - len(negatives):
                raise ValueError("Unable to sample enough negative edges from the candidate space.")
            selected = rng.choice(len(available), size=num_samples - len(negatives), replace=False)
            sampled = available[selected]
            if negatives:
                return np.concatenate([np.asarray(negatives, dtype=np.int64), sampled], axis=0)
            return sampled.astype(np.int64)

        if not asymmetric_input and candidate_nodes is None:
            u = int(rng.integers(num_nodes))
            v = int(rng.integers(num_nodes - 1))
            if v >= u:
                v += 1
        elif not asymmetric_input:
            selected = rng.choice(left_candidate_nodes, size=2, replace=False)
            u = int(selected[0])
            v = int(selected[1])
        else:
            u = int(rng.choice(left_candidate_nodes))
            v = int(rng.choice(right_candidate_nodes))
            if u == v:
                continue
        edge = canonical_edge(u, v)
        if edge in positive_set or edge in forbidden_set or edge in seen:
            continue
        negatives.append(edge)
        seen.add(edge)

    return np.asarray(negatives, dtype=np.int64)


def build_two_hop_negative_candidates(
    num_nodes,
    positive_edges,
    reference_edges,
    candidate_nodes=None,
    left_candidate_nodes=None,
    right_candidate_nodes=None,
):
    candidate_nodes = _candidate_node_array(num_nodes, candidate_nodes)
    left_candidate_nodes, right_candidate_nodes, left_nodes, right_nodes = _resolve_candidate_sets(
        num_nodes,
        candidate_nodes=candidate_nodes,
        left_candidate_nodes=left_candidate_nodes,
        right_candidate_nodes=right_candidate_nodes,
    )
    active_nodes = left_nodes.union(right_nodes)
    positive_set = edge_array_to_set(positive_edges)
    adjacency = [set() for _ in range(num_nodes)]

    for u, v in np.asarray(reference_edges):
        u = int(u)
        v = int(v)
        if u == v or u not in active_nodes or v not in active_nodes:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    candidates = set()
    if candidate_nodes is None and left_candidate_nodes is None and right_candidate_nodes is None:
        node_order = range(num_nodes)
    else:
        node_order = sorted(active_nodes)

    for u in node_order:
        two_hop = set()
        for middle in adjacency[u]:
            two_hop.update(adjacency[middle])
        two_hop.discard(u)

        for v in two_hop:
            if v not in active_nodes:
                continue
            edge = canonical_edge(u, v)
            if not ((edge[0] in left_nodes and edge[1] in right_nodes) or (edge[1] in left_nodes and edge[0] in right_nodes)):
                continue
            if edge in positive_set:
                continue
            candidates.add(edge)

    if not candidates:
        return np.zeros((0, 2), dtype=np.int64)

    return np.asarray(sorted(candidates), dtype=np.int64)


def sample_negative_edges(
    num_nodes,
    num_samples,
    positive_edges,
    rng=None,
    seed=None,
    forbidden_edges=None,
    mode="random",
    reference_edges=None,
    hard_negative_candidates=None,
    candidate_nodes=None,
    left_candidate_nodes=None,
    right_candidate_nodes=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)
    if mode not in {"random", "two_hop_hard"}:
        raise ValueError(f"Unsupported negative sampling mode: {mode}")
    if num_samples == 0:
        return np.zeros((0, 2), dtype=np.int64)

    positive_set = edge_array_to_set(positive_edges)
    forbidden_set = edge_array_to_set(forbidden_edges) if forbidden_edges is not None else set()
    candidate_nodes = _candidate_node_array(num_nodes, candidate_nodes)
    left_candidate_nodes, right_candidate_nodes, left_nodes, right_nodes = _resolve_candidate_sets(
        num_nodes,
        candidate_nodes=candidate_nodes,
        left_candidate_nodes=left_candidate_nodes,
        right_candidate_nodes=right_candidate_nodes,
    )

    if mode == "random":
        return _sample_random_negative_edges(
            num_nodes,
            num_samples,
            positive_set,
            rng,
            forbidden_set,
            candidate_nodes,
            left_candidate_nodes,
            right_candidate_nodes,
        )

    if reference_edges is None:
        reference_edges = positive_edges

    if hard_negative_candidates is None:
        hard_negative_candidates = build_two_hop_negative_candidates(
            num_nodes,
            positive_edges,
            reference_edges,
            candidate_nodes=candidate_nodes,
            left_candidate_nodes=left_candidate_nodes,
            right_candidate_nodes=right_candidate_nodes,
        )

    negatives = []
    seen = set()
    candidate_edges = np.asarray(hard_negative_candidates, dtype=np.int64)

    if len(candidate_edges) > 0:
        for candidate_idx in rng.permutation(len(candidate_edges)):
            u, v = candidate_edges[candidate_idx]
            edge = canonical_edge(int(u), int(v))
            if not ((edge[0] in left_nodes and edge[1] in right_nodes) or (edge[1] in left_nodes and edge[0] in right_nodes)):
                continue
            if edge in positive_set or edge in forbidden_set or edge in seen:
                continue
            negatives.append(edge)
            seen.add(edge)
            if len(negatives) >= num_samples:
                return np.asarray(negatives, dtype=np.int64)

    random_negatives = _sample_random_negative_edges(
        num_nodes,
        num_samples - len(negatives),
        positive_set,
        rng,
        forbidden_set.union(seen),
        candidate_nodes,
        left_candidate_nodes,
        right_candidate_nodes,
    )

    if negatives:
        return np.concatenate([np.asarray(negatives, dtype=np.int64), random_negatives], axis=0)

    return random_negatives
