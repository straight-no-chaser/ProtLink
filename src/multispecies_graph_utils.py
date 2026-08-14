import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data_utils import filter_embeddings, load_embedding_npz, read_fasta_ids, read_tsv_rows
from src.homo_graph_utils import canonical_edge, to_undirected_edge_index


def _normalize_taxid(value):
    return str(value).strip()


def _target_species_value(value):
    value = _normalize_taxid(value)
    return int(value) if value.isdigit() else value


def _build_target_species_nodes(protein_ids, protein_species, target_species):
    target_nodes = [idx for idx, protein_id in enumerate(protein_ids) if protein_species[protein_id] == target_species]
    if not target_nodes:
        raise ValueError(f"Corrupted species: {target_species}")
    return np.asarray(target_nodes, dtype=np.int64)


def build_multispecies_dataset(
    fasta_path,
    protein_metadata_path,
    ppi_edges_path,
    protein_to_orthogroup_path,
    embedding_path,
    min_score=700,
    target_species="9606",  # Homo Sapiens
):
    target_species = _normalize_taxid(target_species)

    fasta_ids = set(read_fasta_ids(fasta_path))
    embedding_ids, x, _ = load_embedding_npz(embedding_path)
    metadata_rows = read_tsv_rows(
        protein_metadata_path,
        ["protein_id", "species_taxid", "species_name"],
    )

    protein_species = {}
    protein_species_name = {}
    for row in metadata_rows:
        protein_id = row["protein_id"].strip()
        protein_species[protein_id] = _normalize_taxid(row["species_taxid"])
        protein_species_name[protein_id] = row["species_name"].strip()

    common_ids = fasta_ids.intersection(embedding_ids).intersection(protein_species)

    ppi_rows = read_tsv_rows(
        ppi_edges_path,
        ["protein1", "protein2", "species_taxid", "combined_score", "source"],
    )
    orthogroup_rows = read_tsv_rows(
        protein_to_orthogroup_path,
        ["protein_id", "orthogroup_id"],
    )

    proteins_with_context = set()
    raw_ppi_edges = []
    for row in ppi_rows:
        protein1 = row["protein1"].strip()
        protein2 = row["protein2"].strip()
        species_taxid = _normalize_taxid(row["species_taxid"])
        if protein1 == protein2:
            continue
        if protein1 not in common_ids or protein2 not in common_ids:
            continue
        if protein_species[protein1] != species_taxid or protein_species[protein2] != species_taxid:
            continue
        if int(row["combined_score"]) < min_score:
            continue
        edge = canonical_edge(protein1, protein2)
        raw_ppi_edges.append((edge[0], edge[1], species_taxid))
        proteins_with_context.add(edge[0])
        proteins_with_context.add(edge[1])

    orthogroup_pairs = []
    for row in orthogroup_rows:
        protein_id = row["protein_id"].strip()
        orthogroup_id = row["orthogroup_id"].strip()
        if protein_id not in common_ids:
            continue
        orthogroup_pairs.append((protein_id, orthogroup_id))
        proteins_with_context.add(protein_id)

    if not raw_ppi_edges or not orthogroup_pairs:
        raise ValueError("Corrupted filters")

    protein_ids, x, id_to_idx = filter_embeddings(embedding_ids, x, proteins_with_context)
    protein_ids = list(protein_ids)

    all_ppi_edges = set()
    target_edges = set()
    context_edges = set()
    for protein1, protein2, species_taxid in raw_ppi_edges:
        if protein1 not in id_to_idx or protein2 not in id_to_idx:
            continue
        edge = canonical_edge(id_to_idx[protein1], id_to_idx[protein2])
        all_ppi_edges.add(edge)
        if species_taxid == target_species:
            target_edges.add(edge)
        else:
            context_edges.add(edge)

    if not target_edges:
        raise ValueError(f"No target-species edges: {target_species}")

    orthogroup_to_idx = {}
    protein_orthogroup_edges = []
    for protein_id, orthogroup_id in orthogroup_pairs:
        if protein_id not in id_to_idx:
            continue
        if orthogroup_id not in orthogroup_to_idx:
            orthogroup_to_idx[orthogroup_id] = len(orthogroup_to_idx)
        protein_orthogroup_edges.append((id_to_idx[protein_id], orthogroup_to_idx[orthogroup_id]))

    retained_species = sorted({protein_species[protein_id] for protein_id in protein_ids})
    target_node_indices = _build_target_species_nodes(protein_ids, protein_species, target_species)

    return {
        "protein_ids": protein_ids,
        "x": x.astype(np.float32),
        "id_to_idx": id_to_idx,
        "edges": np.asarray(sorted(target_edges), dtype=np.int64),
        "num_nodes": len(protein_ids),
        "target_species": target_species,
        "target_node_indices": target_node_indices,
        "context_edges": np.asarray(sorted(context_edges), dtype=np.int64),
        "all_ppi_edges": np.asarray(sorted(all_ppi_edges), dtype=np.int64),
        "protein_to_orthogroup_edges": np.asarray(protein_orthogroup_edges, dtype=np.int64),
        "orthogroup_ids": [orthogroup_id for orthogroup_id, _ in sorted(orthogroup_to_idx.items(), key=lambda item: item[1])],
        "num_orthogroups": len(orthogroup_to_idx),
        "num_species": len(retained_species),
        "species_taxids": retained_species,
        "protein_species": {protein_id: protein_species[protein_id] for protein_id in protein_ids},
        "protein_species_name": {protein_id: protein_species_name[protein_id] for protein_id in protein_ids},
    }


def build_multispecies_hetero_data(dataset, train_target_edges):
    data = HeteroData()
    data["protein"].x = torch.from_numpy(dataset["x"]).float()
    data["protein"].num_nodes = int(dataset["num_nodes"])

    data["orthogroup"].node_id = torch.arange(dataset["num_orthogroups"], dtype=torch.long)
    data["orthogroup"].num_nodes = int(dataset["num_orthogroups"])

    protein_edges = []
    if len(dataset["context_edges"]) > 0:
        protein_edges.append(dataset["context_edges"])
    if len(train_target_edges) > 0:
        protein_edges.append(np.asarray(train_target_edges, dtype=np.int64))
    if protein_edges:
        protein_edges = np.concatenate(protein_edges, axis=0)
        data["protein", "interacts", "protein"].edge_index = torch.from_numpy(to_undirected_edge_index(protein_edges)).long()
    else:
        data["protein", "interacts", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    protein_orthogroup_edges = np.asarray(dataset["protein_to_orthogroup_edges"], dtype=np.int64)
    data["protein", "in_orthogroup", "orthogroup"].edge_index = torch.from_numpy(protein_orthogroup_edges.T).long()
    data["orthogroup", "rev_in_orthogroup", "protein"].edge_index = torch.from_numpy(protein_orthogroup_edges[:, [1, 0]].T).long()

    metadata = {
        "experiment": "multispecies_hetero",
        "graph_type": "hetero",
        "encoder": "hgt",
        "target_species": _target_species_value(dataset["target_species"]),
        "num_species": int(dataset["num_species"]),
        "num_orthogroups": int(dataset["num_orthogroups"]),
        "node_types": list(data.node_types),
        "edge_types": [[src, rel, dst] for src, rel, dst in data.edge_types],
        "num_context_ppi_edges": int(len(dataset["context_edges"])),
    }
    return data, metadata
