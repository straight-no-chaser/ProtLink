from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn

from protlink.data_utils import build_pair_features
from protlink.metrics import compute_metrics, select_threshold
from protlink.models import HGTEncoder, PairMLP, PairMLPDecoder
from protlink.multispecies_graph_utils import (
    build_multispecies_dataset,
    build_multispecies_hetero_data,
)
from protlink.negative_sampling import (
    build_two_hop_negative_candidates,
    edge_array_to_set,
    sample_negative_edges,
)
from protlink.split import split_positive_edges
from protlink.training_utils import (
    build_eval_arrays,
    resolve_device,
    set_seed,
    write_predictions,
)


VARIANT_LABELS = {
    "human_only": "A_human_only_hgt",
    "orthogroup_only": "B_multispecies_orthogroups_no_nonhuman_ppi",
    "full": "C_full_multispecies_hgt",
}


def canonical_pairs(edges):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return edges
    return np.sort(edges, axis=1)


def array_sha256(array):
    array = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_kwargs(split_info, bucket, target_nodes):
    if split_info is None:
        return {"candidate_nodes": target_nodes}
    if bucket == "train":
        return {
            "left_candidate_nodes": split_info["train_nodes"],
            "right_candidate_nodes": split_info["train_nodes"],
        }
    if bucket == "val":
        return {
            "left_candidate_nodes": split_info["val_nodes"],
            "right_candidate_nodes": split_info["train_nodes"],
        }
    if bucket == "test":
        return {
            "left_candidate_nodes": split_info["test_nodes"],
            "right_candidate_nodes": split_info["train_nodes"],
        }
    raise ValueError(f"Unknown split bucket: {bucket}")


def _sample_eval_negatives(
    dataset, split_info, train_pos, val_pos, test_pos, seed, negative_mode
):
    all_pos = edge_array_to_set(dataset["edges"])
    target_nodes = np.asarray(dataset["target_node_indices"], dtype=np.int64)
    rng = np.random.default_rng(10_000 + int(seed))
    hard_counts = {}

    def hard_candidates(bucket):
        if negative_mode != "two_hop_hard":
            hard_counts[bucket] = 0
            return None
        candidates = build_two_hop_negative_candidates(
            dataset["num_nodes"],
            dataset["edges"],
            train_pos,
            **candidate_kwargs(split_info, bucket, target_nodes),
        )
        hard_counts[bucket] = int(len(candidates))
        return candidates

    val_hard = hard_candidates("val")
    test_hard = hard_candidates("test")
    val_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(val_pos),
        all_pos,
        rng=rng,
        mode=negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=val_hard,
        **candidate_kwargs(split_info, "val", target_nodes),
    )
    test_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(test_pos),
        all_pos,
        rng=rng,
        forbidden_edges=val_neg,
        mode=negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=test_hard,
        **candidate_kwargs(split_info, "test", target_nodes),
    )
    fallback = {
        "val": 0.0
        if negative_mode == "random" or len(val_neg) == 0
        else max(0.0, 1.0 - min(hard_counts["val"], len(val_neg)) / len(val_neg)),
        "test": 0.0
        if negative_mode == "random" or len(test_neg) == 0
        else max(0.0, 1.0 - min(hard_counts["test"], len(test_neg)) / len(test_neg)),
    }
    return val_neg, test_neg, hard_counts, fallback


def validate_split_bundle(dataset, bundle, split_mode):
    all_pos = edge_array_to_set(dataset["edges"])
    target_nodes = set(int(x) for x in dataset["target_node_indices"])
    arrays = {
        key: canonical_pairs(bundle[key])
        for key in ["train_pos", "val_pos", "test_pos", "val_neg", "test_neg"]
    }
    for key in ["train_pos", "val_pos", "test_pos", "val_neg", "test_neg"]:
        if arrays[key].ndim != 2 or arrays[key].shape[1] != 2:
            raise ValueError(f"{key} is not an E x 2 pair matrix")
        if any(int(node) not in target_nodes for node in arrays[key].reshape(-1)):
            raise ValueError(f"{key} contains a non-target-species node")
    positive_sets = [
        edge_array_to_set(arrays[key]) for key in ["train_pos", "val_pos", "test_pos"]
    ]
    if (
        positive_sets[0] & positive_sets[1]
        or positive_sets[0] & positive_sets[2]
        or positive_sets[1] & positive_sets[2]
    ):
        raise ValueError("positive pair splits overlap")
    for key in ["val_neg", "test_neg"]:
        if edge_array_to_set(arrays[key]) & all_pos:
            raise ValueError(f"{key} contains a known positive")
    if edge_array_to_set(arrays["val_neg"]) & edge_array_to_set(arrays["test_neg"]):
        raise ValueError("validation and test negatives overlap")

    if split_mode == "node_inductive":
        train_nodes = set(int(x) for x in bundle["train_nodes"])
        val_nodes = set(int(x) for x in bundle["val_nodes"])
        test_nodes = set(int(x) for x in bundle["test_nodes"])
        if (
            train_nodes & val_nodes
            or train_nodes & test_nodes
            or val_nodes & test_nodes
        ):
            raise ValueError("node partitions overlap")
        for u, v in arrays["train_pos"]:
            if int(u) not in train_nodes or int(v) not in train_nodes:
                raise ValueError("training PPI is incident to a held-out protein")
        for key, cold_nodes in [
            ("val_pos", val_nodes),
            ("val_neg", val_nodes),
            ("test_pos", test_nodes),
            ("test_neg", test_nodes),
        ]:
            for u, v in arrays[key]:
                if not (
                    (int(u) in cold_nodes and int(v) in train_nodes)
                    or (int(v) in cold_nodes and int(u) in train_nodes)
                ):
                    raise ValueError(
                        f"{key} violates one-endpoint cold-protein semantics"
                    )
    return True


def prepare_fixed_split(dataset, seed, split_mode, negative_mode, split_path):
    split_path = Path(split_path)
    if split_path.exists():
        loaded = np.load(split_path, allow_pickle=False)
        bundle = {key: loaded[key] for key in loaded.files}
        validate_split_bundle(dataset, bundle, split_mode)
        return bundle

    result = split_positive_edges(
        dataset["edges"],
        dataset["num_nodes"],
        val_ratio=0.1,
        test_ratio=0.1,
        seed=int(seed),
        mode=split_mode,
        candidate_nodes=dataset["target_node_indices"],
    )
    if isinstance(result, dict):
        train_pos, val_pos, test_pos = (
            result["train_pos"],
            result["val_pos"],
            result["test_pos"],
        )
        split_info = {
            key: result[key] for key in ["train_nodes", "val_nodes", "test_nodes"]
        }
    else:
        train_pos, val_pos, test_pos = result
        split_info = None
    val_neg, test_neg, hard_counts, fallback = _sample_eval_negatives(
        dataset, split_info, train_pos, val_pos, test_pos, seed, negative_mode
    )
    bundle = {
        "train_pos": canonical_pairs(train_pos),
        "val_pos": canonical_pairs(val_pos),
        "test_pos": canonical_pairs(test_pos),
        "val_neg": canonical_pairs(val_neg),
        "test_neg": canonical_pairs(test_neg),
        "val_hard_candidate_count": np.asarray([hard_counts["val"]], dtype=np.int64),
        "test_hard_candidate_count": np.asarray([hard_counts["test"]], dtype=np.int64),
        "val_random_fallback_fraction": np.asarray([fallback["val"]], dtype=np.float64),
        "test_random_fallback_fraction": np.asarray(
            [fallback["test"]], dtype=np.float64
        ),
    }
    if split_info is not None:
        bundle.update(split_info)
    validate_split_bundle(dataset, bundle, split_mode)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, **bundle)
    return bundle


def peak_memory_mb():
    info = psutil.Process().memory_info()
    value = getattr(info, "peak_wset", info.rss)
    return float(value / (1024 * 1024))


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _score_hgt(encoder, decoder, data, edges, batch_size, device):
    encoder.eval()
    decoder.eval()
    scores = []
    with torch.no_grad():
        z = encoder(data)
        for start in range(0, len(edges), batch_size):
            pair = torch.as_tensor(
                edges[start : start + batch_size], dtype=torch.long, device=device
            )
            scores.append(torch.sigmoid(decoder(z, pair)).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def _score_mlp(model, x, edges, batch_size, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(edges), batch_size):
            pair = torch.as_tensor(
                edges[start : start + batch_size], dtype=torch.long, device=device
            )
            scores.append(
                torch.sigmoid(model(build_pair_features(x, pair))).cpu().numpy()
            )
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def _training_negatives(dataset, bundle, seed, epoch, negative_mode):
    split_info = None
    if "train_nodes" in bundle:
        split_info = {
            key: bundle[key] for key in ["train_nodes", "val_nodes", "test_nodes"]
        }
    target_nodes = dataset["target_node_indices"]
    kwargs = candidate_kwargs(split_info, "train", target_nodes)
    hard = None
    if negative_mode == "two_hop_hard":
        hard = build_two_hop_negative_candidates(
            dataset["num_nodes"], dataset["edges"], bundle["train_pos"], **kwargs
        )
    reserved = np.concatenate([bundle["val_neg"], bundle["test_neg"]], axis=0)
    return sample_negative_edges(
        dataset["num_nodes"],
        len(bundle["train_pos"]),
        dataset["edges"],
        seed=1_000_000 + int(seed) * 10_000 + int(epoch),
        forbidden_edges=reserved,
        mode=negative_mode,
        reference_edges=bundle["train_pos"],
        hard_negative_candidates=hard,
        **kwargs,
    )


def _base_record(
    model,
    seed,
    dataset,
    bundle,
    elapsed,
    best_epoch,
    threshold,
    val_metrics,
    test_metrics,
):
    return {
        "model": model,
        "seed": int(seed),
        "num_protein_nodes": int(dataset["num_nodes"]),
        "num_orthogroup_nodes": int(dataset["num_orthogroups"]),
        "train_positive_pairs": int(len(bundle["train_pos"])),
        "validation_positive_pairs": int(len(bundle["val_pos"])),
        "test_positive_pairs": int(len(bundle["test_pos"])),
        "validation_negative_pairs": int(len(bundle["val_neg"])),
        "test_negative_pairs": int(len(bundle["test_neg"])),
        "best_epoch": int(best_epoch),
        "training_time_sec": float(elapsed),
        "peak_gpu_memory_mb": None,
        "peak_process_memory_mb": peak_memory_mb(),
        "threshold": float(threshold),
        "validation": val_metrics,
        "test": test_metrics,
    }


def train_hgt(args, dataset, bundle, seed, graph_variant, layers, run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "training.log")
    set_seed(int(seed))
    device = resolve_device(args.device)
    data, graph_meta = build_multispecies_hetero_data(
        dataset, bundle["train_pos"], graph_variant=graph_variant
    )
    data = data.to(device)
    encoder = HGTEncoder(
        protein_input_dim=dataset["x"].shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.hgt_heads,
        layers=layers,
        dropout=args.dropout,
        metadata=data.metadata(),
        node_type_counts={"orthogroup": dataset["num_orthogroups"]},
    ).to(device)
    decoder = PairMLPDecoder(
        args.hidden_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    val_edges, val_labels = build_eval_arrays(bundle["val_pos"], bundle["val_neg"])
    test_edges, test_labels = build_eval_arrays(bundle["test_pos"], bundle["test_neg"])
    train_pos_tensor = torch.as_tensor(
        bundle["train_pos"], dtype=torch.long, device=device
    )
    checkpoint_path = run_dir / "best_model.pt"
    best_ap, best_epoch, wait = -1.0, 0, 0
    start_time = time.perf_counter()
    logger(
        f"start seed={seed} variant={graph_variant} layers={layers} device={device} "
        f"proteins={dataset['num_nodes']} train_pos={len(bundle['train_pos'])}"
    )
    for epoch in range(1, args.epochs + 1):
        train_neg = _training_negatives(
            dataset, bundle, seed, epoch, args.negative_mode
        )
        train_neg_tensor = torch.as_tensor(train_neg, dtype=torch.long, device=device)
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        z = encoder(data)
        logits = torch.cat(
            [decoder(z, train_pos_tensor), decoder(z, train_neg_tensor)], dim=0
        )
        labels = torch.cat(
            [
                torch.ones(len(train_pos_tensor), device=device),
                torch.zeros(len(train_neg_tensor), device=device),
            ]
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        val_scores = _score_hgt(
            encoder, decoder, data, val_edges, args.eval_batch_size, device
        )
        threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, threshold)
        logger(
            f"epoch={epoch} loss={loss.item():.6f} val_ap={val_metrics['ap']:.6f} val_auc={val_metrics['roc_auc']:.6f}"
        )
        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap, best_epoch, wait = val_metrics["ap"], epoch, 0
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            wait += 1
        if wait >= args.patience:
            logger(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break
    elapsed = time.perf_counter() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])
    val_scores = _score_hgt(
        encoder, decoder, data, val_edges, args.eval_batch_size, device
    )
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)
    test_scores = _score_hgt(
        encoder, decoder, data, test_edges, args.eval_batch_size, device
    )
    test_metrics = compute_metrics(test_labels, test_scores, threshold)
    record = _base_record(
        "hgt",
        seed,
        dataset,
        bundle,
        elapsed,
        best_epoch,
        threshold,
        val_metrics,
        test_metrics,
    )
    record.update(
        {
            "graph_variant": graph_variant,
            "condition": VARIANT_LABELS[graph_variant],
            "hgt_layers": int(layers),
            "hgt_heads": int(args.hgt_heads),
            "hidden_dim": int(args.hidden_dim),
            "graph_metadata": graph_meta,
        }
    )
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_predictions(
        run_dir / "validation_predictions.csv",
        dataset["protein_ids"],
        val_edges,
        val_labels,
        val_scores,
        threshold,
    )
    write_predictions(
        run_dir / "test_predictions.csv",
        dataset["protein_ids"],
        test_edges,
        test_labels,
        test_scores,
        threshold,
    )
    logger(
        f"complete best_epoch={best_epoch} test_ap={test_metrics['ap']:.6f} test_auc={test_metrics['roc_auc']:.6f}"
    )
    return record


def train_pair_mlp(args, dataset, bundle, seed, run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "training.log")
    set_seed(int(seed))
    device = resolve_device(args.device)
    x = torch.from_numpy(dataset["x"]).float().to(device)
    model = PairMLP(
        x.size(1) * 4, hidden_dim=args.mlp_hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    val_edges, val_labels = build_eval_arrays(bundle["val_pos"], bundle["val_neg"])
    test_edges, test_labels = build_eval_arrays(bundle["test_pos"], bundle["test_neg"])
    checkpoint_path = run_dir / "best_model.pt"
    best_ap, best_epoch, wait = -1.0, 0, 0
    start_time = time.perf_counter()
    logger(
        f"start seed={seed} model=pair_mlp device={device} train_pos={len(bundle['train_pos'])}"
    )
    for epoch in range(1, args.epochs + 1):
        train_neg = _training_negatives(
            dataset, bundle, seed, epoch, args.negative_mode
        )
        train_edges, train_labels = build_eval_arrays(bundle["train_pos"], train_neg)
        order = np.random.default_rng(2_000_000 + seed * 10_000 + epoch).permutation(
            len(train_edges)
        )
        model.train()
        losses = []
        for start in range(0, len(order), args.mlp_batch_size):
            idx = order[start : start + args.mlp_batch_size]
            pair = torch.as_tensor(train_edges[idx], dtype=torch.long, device=device)
            labels = torch.as_tensor(
                train_labels[idx], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(build_pair_features(x, pair)), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_scores = _score_mlp(model, x, val_edges, args.eval_batch_size, device)
        threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, threshold)
        logger(
            f"epoch={epoch} loss={np.mean(losses):.6f} val_ap={val_metrics['ap']:.6f} val_auc={val_metrics['roc_auc']:.6f}"
        )
        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap, best_epoch, wait = val_metrics["ap"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path
            )
        else:
            wait += 1
        if wait >= args.patience:
            logger(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            break
    elapsed = time.perf_counter() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_scores = _score_mlp(model, x, val_edges, args.eval_batch_size, device)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)
    test_scores = _score_mlp(model, x, test_edges, args.eval_batch_size, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)
    record = _base_record(
        "pair_mlp",
        seed,
        dataset,
        bundle,
        elapsed,
        best_epoch,
        threshold,
        val_metrics,
        test_metrics,
    )
    record["mlp_hidden_dim"] = int(args.mlp_hidden_dim)
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    write_predictions(
        run_dir / "validation_predictions.csv",
        dataset["protein_ids"],
        val_edges,
        val_labels,
        val_scores,
        threshold,
    )
    write_predictions(
        run_dir / "test_predictions.csv",
        dataset["protein_ids"],
        test_edges,
        test_labels,
        test_scores,
        threshold,
    )
    logger(
        f"complete best_epoch={best_epoch} test_ap={test_metrics['ap']:.6f} test_auc={test_metrics['roc_auc']:.6f}"
    )
    return record


def smoke_test(args, dataset, output_dir):
    output_dir = Path(output_dir)
    bundle = prepare_fixed_split(
        dataset, 0, "edge_random", "random", output_dir / "splits" / "smoke_seed_0.npz"
    )
    data, meta = build_multispecies_hetero_data(
        dataset, bundle["train_pos"], graph_variant="full"
    )
    device = resolve_device(args.device)
    data = data.to(device)
    encoder = HGTEncoder(
        protein_input_dim=dataset["x"].shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.hgt_heads,
        layers=2,
        dropout=args.dropout,
        metadata=data.metadata(),
        node_type_counts={"orthogroup": dataset["num_orthogroups"]},
    ).to(device)
    decoder = PairMLPDecoder(
        args.hidden_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    pairs = torch.as_tensor(bundle["train_pos"][:128], dtype=torch.long, device=device)
    start = time.perf_counter()
    z = encoder(data)
    loss = nn.BCEWithLogitsLoss()(
        decoder(z, pairs), torch.ones(len(pairs), device=device)
    )
    loss.backward()
    result = {
        "status": "passed",
        "real_data": True,
        "loss": float(loss.item()),
        "elapsed_sec": float(time.perf_counter() - start),
        "peak_process_memory_mb": peak_memory_mb(),
        "protein_output_shape": list(z.shape),
        "graph_metadata": meta,
    }
    (output_dir / "smoke_test.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def dataset_manifest(dataset, args):
    membership_nodes = set(int(u) for u, _ in dataset["protein_to_orthogroup_edges"])
    target = set(int(x) for x in dataset["target_node_indices"])
    return {
        "input_files": {
            key: {"path": str(Path(value).resolve()), "sha256": file_sha256(value)}
            for key, value in {
                "fasta": args.fasta,
                "protein_metadata": args.protein_metadata,
                "ppi_edges": args.ppi_edges,
                "protein_to_orthogroup": args.protein_to_orthogroup,
                "embeddings": args.embeddings,
            }.items()
        },
        "embedding_shape": list(dataset["x"].shape),
        "embedding_dtype": str(dataset["x"].dtype),
        "num_proteins": int(dataset["num_nodes"]),
        "num_target_proteins": int(len(target)),
        "num_target_ppi_edges": int(len(dataset["edges"])),
        "num_nonhuman_context_ppi_edges": int(len(dataset["context_edges"])),
        "num_orthogroups": int(dataset["num_orthogroups"]),
        "num_orthogroup_memberships": int(len(dataset["protein_to_orthogroup_edges"])),
        "target_orthogroup_assignment_fraction": float(
            len(target & membership_nodes) / len(target)
        ),
        "species_taxids": dataset["species_taxids"],
        "protein_id_order_sha256": hashlib.sha256(
            "\n".join(dataset["protein_ids"]).encode("utf-8")
        ).hexdigest(),
        "target_edges_sha256": array_sha256(canonical_pairs(dataset["edges"])),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=["all", "ablation", "node_inductive"], default="all"
    )
    parser.add_argument(
        "--graph-ablation", choices=["all", *VARIANT_LABELS], default="all"
    )
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--protein-metadata", required=True)
    parser.add_argument("--ppi-edges", required=True)
    parser.add_argument("--protein-to-orthogroup", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--target-species", default="9606")
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--negative-mode", choices=["random", "two_hop_hard"], default="random"
    )
    parser.add_argument("--output-dir", default="outputs/protlink_validation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--hgt-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-three-layer", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_multispecies_dataset(
        args.fasta,
        args.protein_metadata,
        args.ppi_edges,
        args.protein_to_orthogroup,
        args.embeddings,
        min_score=args.min_score,
        target_species=args.target_species,
    )
    dataset["embedding_path"] = args.embeddings
    config = vars(args).copy()
    config["python"] = sys.version
    config["platform"] = platform.platform()
    config["torch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest(dataset, args), indent=2), encoding="utf-8"
    )
    if not args.skip_smoke:
        print(json.dumps(smoke_test(args, dataset, output_dir), indent=2), flush=True)
    if args.smoke_only:
        return

    if args.experiment in {"all", "ablation"}:
        variants = (
            list(VARIANT_LABELS)
            if args.graph_ablation == "all"
            else [args.graph_ablation]
        )
        for seed in args.seeds:
            bundle = prepare_fixed_split(
                dataset,
                seed,
                "edge_random",
                args.negative_mode,
                output_dir / "splits" / f"ablation_seed_{seed}.npz",
            )
            for variant in variants:
                train_hgt(
                    args,
                    dataset,
                    bundle,
                    seed,
                    variant,
                    2,
                    output_dir / "ablation" / "layers_2" / variant / f"seed_{seed}",
                )
            if args.run_three_layer:
                for variant in ["orthogroup_only", "full"]:
                    train_hgt(
                        args,
                        dataset,
                        bundle,
                        seed,
                        variant,
                        3,
                        output_dir / "ablation" / "layers_3" / variant / f"seed_{seed}",
                    )

    if args.experiment in {"all", "node_inductive"}:
        for seed in args.seeds:
            bundle = prepare_fixed_split(
                dataset,
                seed,
                "node_inductive",
                args.negative_mode,
                output_dir / "splits" / f"node_inductive_seed_{seed}.npz",
            )
            train_pair_mlp(
                args,
                dataset,
                bundle,
                seed,
                output_dir / "node_inductive" / "pair_mlp" / f"seed_{seed}",
            )
            train_hgt(
                args,
                dataset,
                bundle,
                seed,
                "full",
                2,
                output_dir / "node_inductive" / "full_hgt" / f"seed_{seed}",
            )


if __name__ == "__main__":
    main()
