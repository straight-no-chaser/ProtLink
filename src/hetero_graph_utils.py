import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.homo_graph_utils import to_undirected_edge_index


def read_typed_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected 2 col: {path} at line {line_number}")
            pairs.append((parts[0], parts[1]))
    return pairs


def build_bipartite_edges(pairs, protein_to_idx):
    other_to_idx = {}
    edges = []

    for protein_id, other_id in pairs:
        if protein_id not in protein_to_idx:
            continue
        if other_id not in other_to_idx:
            other_to_idx[other_id] = len(other_to_idx)
        edges.append((protein_to_idx[protein_id], other_to_idx[other_id]))
    
    return np.asarray(edges, dtype=np.int64), other_to_idx


def edge_type_to_list(edge_types):
    return [[src, rel, dst] for src, rel, dst in edge_types]


def build_pathway_domain_hetero_data(dataset, train_pos, pathway_edges_path, domain_edges_path):
    pathway_pairs = read_typed_pairs(pathway_edges_path)
    domain_pairs = read_typed_pairs(domain_edges_path)

    protein_to_idx = dataset["id_to_idx"]
    pathway_edges, pathway_to_idx = build_bipartite_edges(pathway_pairs, protein_to_idx)
    domain_edges, domain_to_idx = build_bipartite_edges(domain_pairs, protein_to_idx)

    data = HeteroData()
    data["protein"].x = torch.from_numpy(dataset["x"]).float()
    data["protein"].num_nodes = int(dataset["num_nodes"])

    data["pathway"].node_id = torch.arange(len(pathway_to_idx), dtype=torch.long)
    data["pathway"].num_nodes = len(pathway_to_idx)

    data["domain"].node_id = torch.arange(len(domain_to_idx), dtype=torch.long)
    data["domain"].num_nodes = len(domain_to_idx)

    data["protein", "interacts", "protein"].edge_index = torch.from_numpy(to_undirected_edge_index(train_pos)).long()
    data["protein", "in_pathway", "pathway"].edge_index = torch.from_numpy(pathway_edges.T).long()
    data["pathway", "rev_in_pathway", "protein"].edge_index = torch.from_numpy(pathway_edges[:, [1, 0]].T).long()
    data["protein", "has_domain", "domain"].edge_index = torch.from_numpy(domain_edges.T).long()
    data["domain", "rev_has_domain", "protein"].edge_index = torch.from_numpy(domain_edges[:, [1, 0]].T).long()

    metadata = {
        "graph_type": "hetero",
        "node_types": list(data.node_types),
        "edge_types": edge_type_to_list(data.edge_types),
        "num_pathways": len(pathway_to_idx),
        "num_domains": len(domain_to_idx),
    }
    return data, metadata
