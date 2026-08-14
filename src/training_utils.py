import csv
import random

import numpy as np
import torch


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_eval_arrays(pos_edges, neg_edges):
    edges = np.concatenate([pos_edges, neg_edges], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(pos_edges), dtype=np.int64),
            np.zeros(len(neg_edges), dtype=np.int64),
        ]
    )
    return edges, labels


def write_predictions(path, protein_ids, edges, labels, scores, threshold):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein1", "protein2", "label", "score", "pred"])
        for (u, v), label, score in zip(edges, labels, scores):
            writer.writerow(
                [
                    protein_ids[int(u)],
                    protein_ids[int(v)],
                    int(label),
                    float(score),
                    int(score >= threshold),
                ]
            )
