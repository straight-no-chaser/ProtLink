import argparse
import json
from pathlib import Path

import numpy as np

from protlink.homo_graph_utils import build_homo_dataset
from protlink.metrics import compute_metrics, select_threshold, summarize_metric_dicts
from protlink.negative_sampling import build_two_hop_negative_candidates, edge_array_to_set, sample_negative_edges
from protlink.split import split_positive_edges
from protlink.training_utils import build_eval_arrays, write_predictions


SPLIT_MODES = ["edge_random", "node_disjoint", "node_inductive"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[299, 792, 458],
    )
    parser.add_argument(
        "--negative-mode",
        choices=["random", "two_hop_hard"],
        default="random",
    )
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default="edge_random",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cosine",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
    )
    return parser.parse_args()


def normalize_embeddings(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def cosine_scores(x_norm, edges):
    left = x_norm[edges[:, 0]]
    right = x_norm[edges[:, 1]]
    return np.sum(left * right, axis=1)


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def unpack_split_result(split_result):
    if isinstance(split_result, dict):
        return (
            split_result["train_pos"],
            split_result["val_pos"],
            split_result["test_pos"],
            {
                "train_nodes": split_result["train_nodes"],
                "val_nodes": split_result["val_nodes"],
                "test_nodes": split_result["test_nodes"],
            },
        )
    train_pos, val_pos, test_pos = split_result
    return train_pos, val_pos, test_pos, None


def negative_candidate_kwargs(split_info, bucket):
    if split_info is None:
        return {}
    if bucket == "val":
        return {"left_candidate_nodes": split_info["val_nodes"], "right_candidate_nodes": split_info["train_nodes"]}
    if bucket == "test":
        return {"left_candidate_nodes": split_info["test_nodes"], "right_candidate_nodes": split_info["train_nodes"]}
    raise ValueError(f"Unsupported negative candidate bucket: {bucket}")


def add_split_metadata(record, split_info):
    if split_info is None:
        return
    record["num_train_nodes"] = int(len(split_info["train_nodes"]))
    record["num_val_nodes"] = int(len(split_info["val_nodes"]))
    record["num_test_nodes"] = int(len(split_info["test_nodes"]))


def run_seed(
    args,
    dataset,
    x_norm,
    all_pos_set,
    output_dir,
    seed,
):
    train_pos, val_pos, test_pos, split_info = unpack_split_result(
        split_positive_edges(
            dataset["edges"],
            dataset["num_nodes"],
            seed=seed,
            mode=args.split_mode,
        )
    )

    rng = np.random.default_rng(seed)

    val_hard_negative_candidates = None
    test_hard_negative_candidates = None
    if args.negative_mode == "two_hop_hard":
        val_hard_negative_candidates = build_two_hop_negative_candidates(
            dataset["num_nodes"],
            dataset["edges"],
            train_pos,
            **negative_candidate_kwargs(split_info, "val"),
        )
        test_hard_negative_candidates = build_two_hop_negative_candidates(
            dataset["num_nodes"],
            dataset["edges"],
            train_pos,
            **negative_candidate_kwargs(split_info, "test"),
        )

    val_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(val_pos),
        all_pos_set,
        rng=rng,
        mode=args.negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=val_hard_negative_candidates,
        **negative_candidate_kwargs(split_info, "val"),
    )

    test_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(test_pos),
        all_pos_set,
        rng=rng,
        forbidden_edges=val_neg,
        mode=args.negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=test_hard_negative_candidates,
        **negative_candidate_kwargs(split_info, "test"),
    )

    val_edges, val_labels = build_eval_arrays(
        val_pos,
        val_neg,
    )
    test_edges, test_labels = build_eval_arrays(
        test_pos,
        test_neg,
    )

    # select decision threshold using validation data only
    val_scores = cosine_scores(x_norm, val_edges)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(
        val_labels,
        val_scores,
        threshold,
    )

    test_scores = cosine_scores(x_norm, test_edges)
    test_metrics = compute_metrics(
        test_labels,
        test_scores,
        threshold,
    )

    record = {
        "model": "cosine",
        "seed": int(seed),
        "embedding_path": args.embeddings,
        "embedding_name": Path(args.embeddings).stem,
        "negative_mode": args.negative_mode,
        "split_mode": args.split_mode,
        "num_nodes": int(dataset["num_nodes"]),
        "num_edges": int(len(dataset["edges"])),
        "train_pos": int(len(train_pos)),
        "val_pos": int(len(val_pos)),
        "test_pos": int(len(test_pos)),
        "threshold": float(threshold),
        "val": val_metrics,
        "test": test_metrics,
    }
    add_split_metadata(record, split_info)

    write_json(
        output_dir / f"seed_{seed}_metrics.json",
        record,
    )

    if args.save_preds:
        write_predictions(
            output_dir / f"seed_{seed}_val_predictions.csv",
            dataset["protein_ids"],
            val_edges,
            val_labels,
            val_scores,
            threshold,
        )

        write_predictions(
            output_dir / f"seed_{seed}_test_predictions.csv",
            dataset["protein_ids"],
            test_edges,
            test_labels,
            test_scores,
            threshold,
        )

    return record


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_homo_dataset(
        args.fasta,
        args.edges,
        args.embeddings,
        min_score=args.min_score,
    )

    x_norm = normalize_embeddings(dataset["x"])
    all_pos_set = edge_array_to_set(dataset["edges"])

    records = [
        run_seed(
            args=args,
            dataset=dataset,
            x_norm=x_norm,
            all_pos_set=all_pos_set,
            output_dir=output_dir,
            seed=seed,
        )
        for seed in args.seeds
    ]

    summary = {
        "model": "cosine",
        "embedding_path": args.embeddings,
        "embedding_name": Path(args.embeddings).stem,
        "negative_mode": args.negative_mode,
        "split_mode": args.split_mode,
        "seeds": [int(seed) for seed in args.seeds],
        "val": summarize_metric_dicts(
            [record["val"] for record in records]
        ),
        "test": summarize_metric_dicts(
            [record["test"] for record in records]
        ),
    }
    if records and "num_train_nodes" in records[0]:
        summary["num_train_nodes"] = [record["num_train_nodes"] for record in records]
        summary["num_val_nodes"] = [record["num_val_nodes"] for record in records]
        summary["num_test_nodes"] = [record["num_test_nodes"] for record in records]

    write_json(
        output_dir / "summary.json",
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
