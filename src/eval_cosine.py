import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.graph_data import build_dataset
from src.metrics import compute_metrics, select_threshold, summarize_metric_dicts
from src.negative_sampling import edge_array_to_set, sample_negative_edges
from src.split import split_positive_edges


def parse_args():
    parser = argparse.ArgumentParser(description="Cosine similarity baseline for PPI link prediction.")
    parser.add_argument("--fasta", default="seqs.fasta")
    parser.add_argument("--edges", default="9606.protein.physical.links.detailed.v12.0.txt")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def default_output_dir(embedding_path):
    return os.path.join("outputs", "cosine", Path(embedding_path).stem)


def build_eval_arrays(pos_edges, neg_edges):
    edges = np.concatenate([pos_edges, neg_edges], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(pos_edges), dtype=np.int64),
            np.zeros(len(neg_edges), dtype=np.int64),
        ]
    )
    return edges, labels


def cosine_scores(x_norm, edges):
    left = x_norm[edges[:, 0]]
    right = x_norm[edges[:, 1]]
    return np.sum(left * right, axis=1)


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


def run_seed(dataset, output_dir, seed, save_preds):
    train_pos, val_pos, test_pos = split_positive_edges(dataset["edges"], dataset["num_nodes"], seed=seed)
    all_pos_set = edge_array_to_set(dataset["edges"])
    rng = np.random.default_rng(seed)

    val_neg = sample_negative_edges(dataset["num_nodes"], len(val_pos), all_pos_set, rng=rng)
    test_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(test_pos),
        all_pos_set,
        rng=rng,
        forbidden_edges=val_neg,
    )

    x = dataset["x"]
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

    val_edges, val_labels = build_eval_arrays(val_pos, val_neg)
    test_edges, test_labels = build_eval_arrays(test_pos, test_neg)

    val_scores = cosine_scores(x_norm, val_edges)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)

    test_scores = cosine_scores(x_norm, test_edges)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)

    record = {
        "model": "cosine",
        "seed": int(seed),
        "embedding_path": dataset["embedding_path"],
        "embedding_name": Path(dataset["embedding_path"]).stem,
        "num_nodes": int(dataset["num_nodes"]),
        "num_edges": int(len(dataset["edges"])),
        "train_pos": int(len(train_pos)),
        "val_pos": int(len(val_pos)),
        "test_pos": int(len(test_pos)),
        "threshold": float(threshold),
        "val": val_metrics,
        "test": test_metrics,
    }

    metrics_path = os.path.join(output_dir, f"seed_{seed}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    if save_preds:
        write_predictions(
            os.path.join(output_dir, f"seed_{seed}_val_predictions.csv"),
            dataset["protein_ids"],
            val_edges,
            val_labels,
            val_scores,
            threshold,
        )
        write_predictions(
            os.path.join(output_dir, f"seed_{seed}_test_predictions.csv"),
            dataset["protein_ids"],
            test_edges,
            test_labels,
            test_scores,
            threshold,
        )

    return record


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.embeddings)
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_dataset(args.fasta, args.edges, args.embeddings, min_score=args.min_score)
    dataset["embedding_path"] = args.embeddings

    records = [run_seed(dataset, output_dir, seed, args.save_preds) for seed in args.seeds]
    summary = {
        "model": "cosine",
        "embedding_path": args.embeddings,
        "embedding_name": Path(args.embeddings).stem,
        "seeds": [int(seed) for seed in args.seeds],
        "val": summarize_metric_dicts([record["val"] for record in records]),
        "test": summarize_metric_dicts([record["test"] for record in records]),
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
