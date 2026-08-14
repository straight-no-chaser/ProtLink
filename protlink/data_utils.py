import csv
import numpy as np
from collections import OrderedDict

import torch


def read_fasta(path):
    sequences = OrderedDict()
    current_id = None
    chunks = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def read_fasta_ids(path):
    return list(read_fasta(path).keys())


def read_tsv_rows(path, required_columns):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Corrupted columns in {path}: {', '.join(missing)}")
        rows = [row for row in reader if any(value not in (None, "") for value in row.values())]
    return rows


def _normalize_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_embedding_npz(path):
    with np.load(path, allow_pickle=True) as data:
        ids = [_normalize_id(value) for value in data["ids"]]
        x = np.asarray(data["embeddings"], dtype=np.float32)

    id_to_idx = {protein_id: idx for idx, protein_id in enumerate(ids)}
    return ids, x, id_to_idx


def filter_embeddings(ids, x, keep_ids):
    keep_ids = set(keep_ids)
    keep_idx = [idx for idx, protein_id in enumerate(ids) if protein_id in keep_ids]
    filtered_ids = [ids[idx] for idx in keep_idx]
    filtered_x = x[keep_idx]
    id_to_idx = {protein_id: idx for idx, protein_id in enumerate(filtered_ids)}
    return filtered_ids, filtered_x, id_to_idx


def build_pair_features(x, edges):
    x_u = x[edges[:, 0]]
    x_v = x[edges[:, 1]]
    return torch.cat([x_u, x_v, torch.abs(x_u - x_v), x_u * x_v], dim=-1)
