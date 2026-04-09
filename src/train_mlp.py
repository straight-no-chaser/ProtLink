import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features import build_pair_features
from src.graph_data import build_dataset
from src.metrics import compute_metrics, select_threshold, summarize_metric_dicts
from src.models import PairMLP
from src.negative_sampling import edge_array_to_set, sample_negative_edges
from src.split import split_positive_edges


def parse_args():
    parser = argparse.ArgumentParser(description="MLP baseline for PPI link prediction.")
    parser.add_argument("--fasta", default="seqs.fasta")
    parser.add_argument("--edges", default="9606.protein.physical.links.detailed.v12.0.txt")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def default_output_dir(embedding_path):
    return os.path.join("outputs", "mlp", Path(embedding_path).stem)


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


def score_pairs(model, x, edges, batch_size, device):
    model.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(edges), batch_size):
            batch_edges = torch.as_tensor(edges[start : start + batch_size], dtype=torch.long, device=device)
            features = build_pair_features(x, batch_edges)
            logits = model(features)
            scores.append(torch.sigmoid(logits).cpu().numpy())

    if scores:
        return np.concatenate(scores)
    return np.empty(0, dtype=np.float32)


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


def train_one_seed(args, dataset, output_dir, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)

    train_pos, val_pos, test_pos = split_positive_edges(dataset["edges"], dataset["num_nodes"], seed=seed)
    all_pos_set = edge_array_to_set(dataset["edges"])
    val_neg = sample_negative_edges(dataset["num_nodes"], len(val_pos), all_pos_set, rng=rng)
    test_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(test_pos),
        all_pos_set,
        rng=rng,
        forbidden_edges=val_neg,
    )

    val_edges, val_labels = build_eval_arrays(val_pos, val_neg)
    test_edges, test_labels = build_eval_arrays(test_pos, test_neg)

    x = torch.from_numpy(dataset["x"]).float().to(device)
    model = PairMLP(x.size(1) * 4, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(output_dir, f"seed_{seed}_best_model.pt")
    best_ap = -1.0
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_neg = sample_negative_edges(dataset["num_nodes"], len(train_pos), all_pos_set, rng=rng)
        train_edges, train_labels = build_eval_arrays(train_pos, train_neg)
        order = rng.permutation(len(train_edges))
        train_edges = train_edges[order]
        train_labels = train_labels[order]

        model.train()
        for start in range(0, len(train_edges), args.batch_size):
            batch_edges = torch.as_tensor(train_edges[start : start + args.batch_size], dtype=torch.long, device=device)
            batch_labels = torch.as_tensor(train_labels[start : start + args.batch_size], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            features = build_pair_features(x, batch_edges)
            logits = model(features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        val_scores = score_pairs(model, x, val_edges, args.eval_batch_size, device)
        val_threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, val_threshold)

        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap = val_metrics["ap"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "threshold": float(val_threshold),
                },
                checkpoint_path,
            )
        else:
            wait += 1

        if wait >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    val_scores = score_pairs(model, x, val_edges, args.eval_batch_size, device)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)

    test_scores = score_pairs(model, x, test_edges, args.eval_batch_size, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)

    record = {
        "model": "mlp",
        "seed": int(seed),
        "embedding_path": dataset["embedding_path"],
        "embedding_name": Path(dataset["embedding_path"]).stem,
        "device": device,
        "num_nodes": int(dataset["num_nodes"]),
        "num_edges": int(len(dataset["edges"])),
        "train_pos": int(len(train_pos)),
        "val_pos": int(len(val_pos)),
        "test_pos": int(len(test_pos)),
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(os.path.join(output_dir, f"seed_{seed}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    if args.save_preds:
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

    records = [train_one_seed(args, dataset, output_dir, seed) for seed in args.seeds]
    summary = {
        "model": "mlp",
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
