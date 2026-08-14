import numpy as np

from src.data_utils import filter_embeddings, load_embedding_npz, read_fasta_ids


def canonical_edge(u, v):
    return (u, v) if u < v else (v, u)


def read_string_edges(path, min_score=700):
    edges = set()

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if not header:
            raise ValueError()

        try:
            protein1_idx = header.index("protein1")
            protein2_idx = header.index("protein2")
            score_idx = header.index("combined_score")
        except ValueError as exc:
            raise ValueError("Corrupted columns") from exc

        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts:
                continue
            u = parts[protein1_idx]
            v = parts[protein2_idx]
            if u == v:
                continue
            if int(parts[score_idx]) < min_score:
                continue
            edges.add(canonical_edge(u, v))

    return sorted(edges)


def to_undirected_edge_index(edges):
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    reverse_edges = edges[:, [1, 0]]
    return np.concatenate([edges, reverse_edges], axis=0).T


def build_homo_dataset(fasta_path, edge_path, embedding_path, min_score=700):
    fasta_ids = set(read_fasta_ids(fasta_path))
    embedding_ids, x, _ = load_embedding_npz(embedding_path)
    common_ids = fasta_ids.intersection(embedding_ids)
    protein_ids, x, id_to_idx = filter_embeddings(embedding_ids, x, common_ids)

    filtered_edges = []
    for protein1, protein2 in read_string_edges(edge_path, min_score=min_score):
        if protein1 in id_to_idx and protein2 in id_to_idx:
            filtered_edges.append((id_to_idx[protein1], id_to_idx[protein2]))

    filtered_edges.sort()
    edges = np.asarray(filtered_edges, dtype=np.int64)

    return {
        "protein_ids": protein_ids,
        "x": x.astype(np.float32),
        "id_to_idx": id_to_idx,
        "edges": edges,
        "num_nodes": len(protein_ids),
    }
