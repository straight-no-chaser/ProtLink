import numpy as np


def _normalize_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_embedding_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if "ids" in data:
        ids = [_normalize_id(value) for value in data["ids"]]
        if "embeddings" in data:
            x = np.asarray(data["embeddings"], dtype=np.float32)
        elif "emb" in data:
            x = np.asarray(data["emb"], dtype=np.float32)
        else:
            raise ValueError("Expected 'embeddings' or 'emb' key in embedding npz file.")
    else:
        ids = sorted(keys)
        rows = [np.asarray(data[key], dtype=np.float32).reshape(-1) for key in ids]
        x = np.vstack(rows)

    if x.ndim != 2:
        raise ValueError("Embedding matrix must have shape (N, d).")
    if len(ids) != x.shape[0]:
        raise ValueError("Embedding ids and rows do not match.")

    id_to_idx = {protein_id: idx for idx, protein_id in enumerate(ids)}
    return ids, x, id_to_idx


def filter_embeddings(ids, x, keep_ids):
    keep_ids = set(keep_ids)
    keep_idx = [idx for idx, protein_id in enumerate(ids) if protein_id in keep_ids]
    filtered_ids = [ids[idx] for idx in keep_idx]
    filtered_x = x[keep_idx]
    id_to_idx = {protein_id: idx for idx, protein_id in enumerate(filtered_ids)}
    return filtered_ids, filtered_x, id_to_idx
