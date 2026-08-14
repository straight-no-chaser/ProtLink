import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from protlink.data_utils import build_pair_features
from protlink.homo_graph_utils import build_homo_dataset, to_undirected_edge_index
from protlink.hetero_graph_utils import build_pathway_domain_hetero_data
from protlink.metrics import compute_metrics, select_threshold, summarize_metric_dicts
from protlink.models import DotDecoder, GraphSAGEEncoder, GraphTransformerEncoder, HGTEncoder, PairMLP, PairMLPDecoder
from protlink.negative_sampling import build_two_hop_negative_candidates, edge_array_to_set, sample_negative_edges
from protlink.split import split_positive_edges
from protlink.training_utils import build_eval_arrays, resolve_device, set_seed, write_predictions


SPLIT_MODES = ["edge_random", "node_disjoint", "node_inductive"]


def parse_mlp_args():
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
    parser.add_argument("--negative-mode", choices=["random", "two_hop_hard"], default="random")
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="edge_random")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def default_mlp_output_dir(embedding_path):
    return os.path.join("outputs", "mlp", Path(embedding_path).stem)


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


def mlp_metadata(record, args):
    if args.negative_mode != "random":
        record["negative_mode"] = args.negative_mode
    if args.split_mode != "edge_random":
        record["split_mode"] = args.split_mode


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


def split_node_counts(split_info):
    if split_info is None:
        return {}
    return {
        "num_train_nodes": int(len(split_info["train_nodes"])),
        "num_val_nodes": int(len(split_info["val_nodes"])),
        "num_test_nodes": int(len(split_info["test_nodes"])),
    }


def add_split_metadata(record, split_info):
    record.update(split_node_counts(split_info))


def add_summary_split_metadata(summary, records):
    if not records or "num_train_nodes" not in records[0]:
        return
    summary["num_train_nodes"] = [int(record["num_train_nodes"]) for record in records]
    summary["num_val_nodes"] = [int(record["num_val_nodes"]) for record in records]
    summary["num_test_nodes"] = [int(record["num_test_nodes"]) for record in records]


def negative_candidate_kwargs(split_info, bucket, candidate_nodes=None):
    if split_info is None:
        if candidate_nodes is None:
            return {}
        return {"candidate_nodes": candidate_nodes}
    if bucket == "train":
        left_nodes = split_info["train_nodes"]
        right_nodes = split_info["train_nodes"]
    elif bucket == "val":
        left_nodes = split_info["val_nodes"]
        right_nodes = split_info["train_nodes"]
    elif bucket == "test":
        left_nodes = split_info["test_nodes"]
        right_nodes = split_info["train_nodes"]
    else:
        raise ValueError(f"Unsupported negative candidate bucket: {bucket}")
    return {"left_candidate_nodes": left_nodes, "right_candidate_nodes": right_nodes}


def build_hard_negative_candidates(args, num_nodes, positive_edges, reference_edges, split_info, bucket, candidate_nodes=None):
    if args.negative_mode != "two_hop_hard":
        return None
    return build_two_hop_negative_candidates(
        num_nodes,
        positive_edges,
        reference_edges,
        **negative_candidate_kwargs(split_info, bucket, candidate_nodes=candidate_nodes),
    )


def validate_negative_edges(negative_edges, all_pos_set, split_info=None, bucket=None):
    left_nodes = None
    right_nodes = None
    if split_info is not None and bucket is not None:
        kwargs = negative_candidate_kwargs(split_info, bucket)
        left_nodes = set(int(node) for node in kwargs["left_candidate_nodes"])
        right_nodes = set(int(node) for node in kwargs["right_candidate_nodes"])
    for u, v in negative_edges:
        edge = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        if edge in all_pos_set:
            raise ValueError("Negative sampling produced an observed positive edge.")
        if left_nodes is not None:
            if not ((edge[0] in left_nodes and edge[1] in right_nodes) or (edge[1] in left_nodes and edge[0] in right_nodes)):
                raise ValueError(f"Negative sampling produced an edge outside the {bucket} inductive candidate space.")


def train_mlp_one_seed(args, dataset, output_dir, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)

    train_pos, val_pos, test_pos, split_info = unpack_split_result(
        split_positive_edges(
            dataset["edges"],
            dataset["num_nodes"],
            seed=seed,
            mode=args.split_mode,
        )
    )
    all_pos_set = edge_array_to_set(dataset["edges"])
    train_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "train"
    )
    val_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "val"
    )
    test_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "test"
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
    validate_negative_edges(val_neg, all_pos_set, split_info, "val")
    validate_negative_edges(test_neg, all_pos_set, split_info, "test")

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
        train_neg = sample_negative_edges(
            dataset["num_nodes"],
            len(train_pos),
            all_pos_set,
            rng=rng,
            mode=args.negative_mode,
            reference_edges=train_pos,
            hard_negative_candidates=train_hard_negative_candidates,
            **negative_candidate_kwargs(split_info, "train"),
        )
        validate_negative_edges(train_neg, all_pos_set, split_info, "train")
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
    mlp_metadata(record, args)
    add_split_metadata(record, split_info)

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


def main_mlp():
    args = parse_mlp_args()
    output_dir = args.output_dir or default_mlp_output_dir(args.embeddings)
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_homo_dataset(args.fasta, args.edges, args.embeddings, min_score=args.min_score)
    dataset["embedding_path"] = args.embeddings

    records = [train_mlp_one_seed(args, dataset, output_dir, seed) for seed in args.seeds]
    summary = {
        "model": "mlp",
        "embedding_path": args.embeddings,
        "embedding_name": Path(args.embeddings).stem,
        "seeds": [int(seed) for seed in args.seeds],
        "val": summarize_metric_dicts([record["val"] for record in records]),
        "test": summarize_metric_dicts([record["test"] for record in records]),
    }
    mlp_metadata(summary, args)
    add_summary_split_metadata(summary, records)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))



def parse_graph_args():
    parser = argparse.ArgumentParser(description="GraphSAGE link prediction for PPI.")
    parser.add_argument("--fasta", default="seqs.fasta")
    parser.add_argument("--edges", default="9606.protein.physical.links.detailed.v12.0.txt")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--experiment", choices=["single_species", "multispecies_hetero"], default="single_species")
    parser.add_argument("--min-score", type=int, default=700)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--graph-type", choices=["homo", "hetero"], default="homo")
    parser.add_argument("--encoder", choices=["graphsage", "gt", "hgt"], default="graphsage")
    parser.add_argument("--decoder", choices=["dot", "mlp"], default="dot")
    parser.add_argument("--residual", choices=["none", "concat", "add"], default="none")
    parser.add_argument("--gt-heads", type=int, default=4)
    parser.add_argument("--gt-dropout", type=float, default=0.2)
    parser.add_argument("--gt-layers", type=int, default=2)
    parser.add_argument("--pathway-edges", default=None)
    parser.add_argument("--domain-edges", default=None)
    parser.add_argument("--protein-metadata", default=None)
    parser.add_argument("--ppi-edges", default=None)
    parser.add_argument("--protein-to-orthogroup", default=None)
    parser.add_argument("--target-species", default="9606")
    parser.add_argument("--hgt-heads", type=int, default=4)
    parser.add_argument("--hgt-dropout", type=float, default=0.2)
    parser.add_argument("--hgt-layers", type=int, default=2)
    parser.add_argument("--negative-mode", choices=["random", "two_hop_hard"], default="random")
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="edge_random")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def default_graph_output_dir(embedding_path, experiment="single_species"):
    if experiment == "multispecies_hetero":
        return os.path.join("outputs", "multispecies_hetero", Path(embedding_path).stem)
    return os.path.join("outputs", "graphsage", Path(embedding_path).stem)


def score_edges(encoder, decoder, x, edge_index, edges, batch_size, device):
    encoder.eval()
    decoder.eval()
    scores = []

    with torch.no_grad():
        z = encoder(x, edge_index)
        for start in range(0, len(edges), batch_size):
            batch_edges = torch.as_tensor(edges[start : start + batch_size], dtype=torch.long, device=device)
            logits = decoder(z, batch_edges)
            scores.append(torch.sigmoid(logits).cpu().numpy())

    if scores:
        return np.concatenate(scores)
    return np.empty(0, dtype=np.float32)


def graph_metadata(record, args, extra_metadata=None):
    if args.experiment != "single_species":
        record["experiment"] = args.experiment
    if args.graph_type != "homo":
        record["graph_type"] = args.graph_type
    if args.encoder != "graphsage":
        record["encoder"] = args.encoder
        if args.encoder == "gt":
            record["gt_heads"] = args.gt_heads
            record["gt_dropout"] = args.gt_dropout
            record["gt_layers"] = args.gt_layers
        if args.encoder == "hgt":
            record["hgt_heads"] = args.hgt_heads
            record["hgt_dropout"] = args.hgt_dropout
            record["hgt_layers"] = args.hgt_layers
    if args.decoder != "dot":
        record["decoder"] = args.decoder
    if args.residual != "none":
        record["residual"] = args.residual
    if args.negative_mode != "random":
        record["negative_mode"] = args.negative_mode
    if args.split_mode != "edge_random":
        record["split_mode"] = args.split_mode
    if extra_metadata is not None:
        record.update(extra_metadata)


def format_metric(value):
    if value is None or np.isnan(value):
        return "NA"
    return f"{float(value):.6f}"


def build_decoder(args, latent_dim, device):
    if args.decoder == "mlp":
        return PairMLPDecoder(latent_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    return DotDecoder().to(device)


def build_hyperparams(args, device):
    hyperparams = {
        "hidden_dim": args.hidden_dim,
        "num_layers": 2,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "eval_batch_size": args.eval_batch_size,
        "device": device,
        "decoder": args.decoder,
        "residual": args.residual,
        "negative_mode": args.negative_mode,
        "split_mode": args.split_mode,
    }
    if args.experiment != "single_species":
        hyperparams["experiment"] = args.experiment
    if args.graph_type != "homo":
        hyperparams["graph_type"] = args.graph_type
    if args.encoder != "graphsage":
        hyperparams["encoder"] = args.encoder
    if args.encoder == "gt":
        hyperparams["num_layers"] = args.gt_layers
        hyperparams["gt_heads"] = args.gt_heads
        hyperparams["gt_dropout"] = args.gt_dropout
        hyperparams["gt_layers"] = args.gt_layers
    if args.encoder == "hgt":
        hyperparams["num_layers"] = args.hgt_layers
        hyperparams["hgt_heads"] = args.hgt_heads
        hyperparams["hgt_dropout"] = args.hgt_dropout
        hyperparams["hgt_layers"] = args.hgt_layers
        if args.pathway_edges is not None:
            hyperparams["pathway_edges"] = args.pathway_edges
        if args.domain_edges is not None:
            hyperparams["domain_edges"] = args.domain_edges
    if args.experiment == "multispecies_hetero":
        hyperparams["protein_metadata"] = args.protein_metadata
        hyperparams["ppi_edges"] = args.ppi_edges
        hyperparams["protein_to_orthogroup"] = args.protein_to_orthogroup
        hyperparams["target_species"] = args.target_species
    return hyperparams


def score_edges_hetero(encoder, decoder, data, edges, batch_size, device):
    encoder.eval()
    decoder.eval()
    scores = []

    with torch.no_grad():
        z = encoder(data)
        for start in range(0, len(edges), batch_size):
            batch_edges = torch.as_tensor(edges[start : start + batch_size], dtype=torch.long, device=device)
            logits = decoder(z, batch_edges)
            scores.append(torch.sigmoid(logits).cpu().numpy())

    if scores:
        return np.concatenate(scores)
    return np.empty(0, dtype=np.float32)


def train_graph_one_seed(args, dataset, output_dir, seed):
    seed_start_time = time.perf_counter()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)

    train_pos, val_pos, test_pos, split_info = unpack_split_result(
        split_positive_edges(
            dataset["edges"],
            dataset["num_nodes"],
            seed=seed,
            mode=args.split_mode,
        )
    )
    all_pos_set = edge_array_to_set(dataset["edges"])
    train_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "train"
    )
    val_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "val"
    )
    test_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "test"
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
    validate_negative_edges(val_neg, all_pos_set, split_info, "val")
    validate_negative_edges(test_neg, all_pos_set, split_info, "test")

    val_edges, val_labels = build_eval_arrays(val_pos, val_neg)
    test_edges, test_labels = build_eval_arrays(test_pos, test_neg)

    hyperparams = build_hyperparams(args, device)
    print(f"[seed={seed}] num_nodes={dataset['num_nodes']}", flush=True)
    print(
        f"[seed={seed}] num_edges={len(dataset['edges'])} "
        f"train_edges={len(train_pos)} val_edges={len(val_pos)} test_edges={len(test_pos)}",
        flush=True,
    )
    print(f"[seed={seed}] hyperparams: {json.dumps(hyperparams, sort_keys=True)}", flush=True)

    x = torch.from_numpy(dataset["x"]).float().to(device)
    edge_index = torch.from_numpy(to_undirected_edge_index(train_pos)).long().to(device)
    train_pos_tensor = torch.as_tensor(train_pos, dtype=torch.long, device=device)

    if args.encoder == "gt":
        encoder = GraphTransformerEncoder(
            x.size(1),
            hidden_dim=args.hidden_dim,
            heads=args.gt_heads,
            layers=args.gt_layers,
            dropout=args.gt_dropout,
            residual=args.residual,
        ).to(device)
    else:
        if args.encoder != "graphsage":
            raise ValueError("Encoder 'hgt' requires --graph-type hetero.")
        encoder = GraphSAGEEncoder(
            x.size(1),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            residual=args.residual,
        ).to(device)
    decoder = build_decoder(args, encoder.output_dim, device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(output_dir, f"seed_{seed}_best_model.pt")
    best_ap = -1.0
    best_val_auc = float("nan")
    best_epoch = 0
    wait = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_neg = sample_negative_edges(
            dataset["num_nodes"],
            len(train_pos),
            all_pos_set,
            rng=rng,
            mode=args.negative_mode,
            reference_edges=train_pos,
            hard_negative_candidates=train_hard_negative_candidates,
            **negative_candidate_kwargs(split_info, "train"),
        )
        validate_negative_edges(train_neg, all_pos_set, split_info, "train")
        train_neg_tensor = torch.as_tensor(train_neg, dtype=torch.long, device=device)

        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        z = encoder(x, edge_index)
        pos_logits = decoder(z, train_pos_tensor)
        neg_logits = decoder(z, train_neg_tensor)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(train_pos_tensor), device=device),
                torch.zeros(len(train_neg_tensor), device=device),
            ],
            dim=0,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        val_scores = score_edges(encoder, decoder, x, edge_index, val_edges, args.eval_batch_size, device)
        val_threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, val_threshold)
        val_auc = val_metrics["roc_auc"]
        print(
            f"[seed={seed}][epoch={epoch}/{args.epochs}] loss={train_loss:.6f} val_auc={format_metric(val_auc)}",
            flush=True,
        )

        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap = val_metrics["ap"]
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 0
            checkpoint = {
                "encoder_state": encoder.state_dict(),
                "epoch": epoch,
                "threshold": float(val_threshold),
            }
            if args.decoder != "dot":
                checkpoint["decoder_state"] = decoder.state_dict()
            torch.save(checkpoint, checkpoint_path)
        else:
            wait += 1

        if wait >= args.patience:
            stopped_early = True
            print(
                f"[seed={seed}] early_stopping_triggered epoch={epoch} "
                f"reason=patience_exhausted best_val_auc={format_metric(best_val_auc)} best_epoch={best_epoch}",
                flush=True,
            )
            break

    if not stopped_early:
        print(f"[seed={seed}] training_finished_without_early_stopping final_epoch={args.epochs}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    if "decoder_state" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state"])

    val_scores = score_edges(encoder, decoder, x, edge_index, val_edges, args.eval_batch_size, device)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)

    test_scores = score_edges(encoder, decoder, x, edge_index, test_edges, args.eval_batch_size, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)

    record = {
        "model": "graphsage",
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
    graph_metadata(record, args)
    add_split_metadata(record, split_info)

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

    elapsed_sec = time.perf_counter() - seed_start_time
    print(f"[seed={seed}] elapsed_sec={elapsed_sec:.2f}", flush=True)

    seed_log = {
        "seed": int(seed),
        "elapsed_sec": float(elapsed_sec),
        "best_epoch": int(best_epoch),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
    }
    return record, seed_log


def train_graph_one_seed_hetero(args, dataset, output_dir, seed):
    seed_start_time = time.perf_counter()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)

    target_candidate_nodes = dataset.get("target_node_indices")
    train_pos, val_pos, test_pos, split_info = unpack_split_result(
        split_positive_edges(
            dataset["edges"],
            dataset["num_nodes"],
            seed=seed,
            mode=args.split_mode,
            candidate_nodes=target_candidate_nodes,
        )
    )
    all_pos_set = edge_array_to_set(dataset["edges"])
    train_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "train", candidate_nodes=target_candidate_nodes
    )
    val_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "val", candidate_nodes=target_candidate_nodes
    )
    test_hard_negative_candidates = build_hard_negative_candidates(
        args, dataset["num_nodes"], dataset["edges"], train_pos, split_info, "test", candidate_nodes=target_candidate_nodes
    )

    val_neg = sample_negative_edges(
        dataset["num_nodes"],
        len(val_pos),
        all_pos_set,
        rng=rng,
        mode=args.negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=val_hard_negative_candidates,
        **negative_candidate_kwargs(split_info, "val", candidate_nodes=target_candidate_nodes),
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
        **negative_candidate_kwargs(split_info, "test", candidate_nodes=target_candidate_nodes),
    )
    validate_negative_edges(val_neg, all_pos_set, split_info, "val")
    validate_negative_edges(test_neg, all_pos_set, split_info, "test")

    val_edges, val_labels = build_eval_arrays(val_pos, val_neg)
    test_edges, test_labels = build_eval_arrays(test_pos, test_neg)

    # Inductive mode hides target PPI topology; auxiliary pathway/domain metadata remains visible.
    hetero_data, hetero_metadata = build_pathway_domain_hetero_data(
        dataset,
        train_pos,
        args.pathway_edges,
        args.domain_edges,
    )
    hetero_data = hetero_data.to(device)
    train_pos_tensor = torch.as_tensor(train_pos, dtype=torch.long, device=device)

    hyperparams = build_hyperparams(args, device)
    print(f"[seed={seed}] num_nodes={dataset['num_nodes']}", flush=True)
    print(
        f"[seed={seed}] num_edges={len(dataset['edges'])} "
        f"train_edges={len(train_pos)} val_edges={len(val_pos)} test_edges={len(test_pos)}",
        flush=True,
    )
    print(f"[seed={seed}] hyperparams: {json.dumps(hyperparams, sort_keys=True)}", flush=True)

    encoder = HGTEncoder(
        protein_input_dim=dataset["x"].shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.hgt_heads,
        layers=args.hgt_layers,
        dropout=args.hgt_dropout,
        metadata=hetero_data.metadata(),
        num_pathways=hetero_metadata["num_pathways"],
        num_domains=hetero_metadata["num_domains"],
    ).to(device)
    decoder = build_decoder(args, encoder.output_dim, device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(output_dir, f"seed_{seed}_best_model.pt")
    best_ap = -1.0
    best_val_auc = float("nan")
    best_epoch = 0
    wait = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_neg = sample_negative_edges(
            dataset["num_nodes"],
            len(train_pos),
            all_pos_set,
            rng=rng,
            mode=args.negative_mode,
            reference_edges=train_pos,
            hard_negative_candidates=train_hard_negative_candidates,
            **negative_candidate_kwargs(split_info, "train", candidate_nodes=target_candidate_nodes),
        )
        validate_negative_edges(train_neg, all_pos_set, split_info, "train")
        train_neg_tensor = torch.as_tensor(train_neg, dtype=torch.long, device=device)

        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        z = encoder(hetero_data)
        pos_logits = decoder(z, train_pos_tensor)
        neg_logits = decoder(z, train_neg_tensor)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(train_pos_tensor), device=device),
                torch.zeros(len(train_neg_tensor), device=device),
            ],
            dim=0,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        val_scores = score_edges_hetero(encoder, decoder, hetero_data, val_edges, args.eval_batch_size, device)
        val_threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, val_threshold)
        val_auc = val_metrics["roc_auc"]
        print(
            f"[seed={seed}][epoch={epoch}/{args.epochs}] loss={train_loss:.6f} val_auc={format_metric(val_auc)}",
            flush=True,
        )

        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap = val_metrics["ap"]
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 0
            checkpoint = {
                "encoder_state": encoder.state_dict(),
                "epoch": epoch,
                "threshold": float(val_threshold),
            }
            if args.decoder != "dot":
                checkpoint["decoder_state"] = decoder.state_dict()
            torch.save(checkpoint, checkpoint_path)
        else:
            wait += 1

        if wait >= args.patience:
            stopped_early = True
            print(
                f"[seed={seed}] early_stopping_triggered epoch={epoch} "
                f"reason=patience_exhausted best_val_auc={format_metric(best_val_auc)} best_epoch={best_epoch}",
                flush=True,
            )
            break

    if not stopped_early:
        print(f"[seed={seed}] training_finished_without_early_stopping final_epoch={args.epochs}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    if "decoder_state" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state"])

    val_scores = score_edges_hetero(encoder, decoder, hetero_data, val_edges, args.eval_batch_size, device)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)

    test_scores = score_edges_hetero(encoder, decoder, hetero_data, test_edges, args.eval_batch_size, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)

    record = {
        "model": "graphsage",
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
    graph_metadata(record, args, hetero_metadata)
    add_split_metadata(record, split_info)

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

    elapsed_sec = time.perf_counter() - seed_start_time
    print(f"[seed={seed}] elapsed_sec={elapsed_sec:.2f}", flush=True)

    seed_log = {
        "seed": int(seed),
        "elapsed_sec": float(elapsed_sec),
        "best_epoch": int(best_epoch),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
    }
    return record, seed_log, hetero_metadata


def train_graph_one_seed_multispecies(args, dataset, output_dir, seed):
    from protlink.multispecies_graph_utils import build_multispecies_hetero_data

    seed_start_time = time.perf_counter()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(args.device)

    target_nodes = np.asarray(dataset["target_node_indices"], dtype=np.int64)
    num_nodes = int(target_nodes.max()) + 1 if len(target_nodes) > 0 else 0
    train_pos, val_pos, test_pos, split_info = unpack_split_result(
        split_positive_edges(
            dataset["edges"],
            dataset["num_nodes"],
            seed=seed,
            mode=args.split_mode,
            candidate_nodes=target_nodes,
        )
    )
    all_pos_set = edge_array_to_set(dataset["edges"])
    train_hard_negative_candidates = build_hard_negative_candidates(
        args, num_nodes, dataset["edges"], train_pos, split_info, "train", candidate_nodes=target_nodes
    )
    val_hard_negative_candidates = build_hard_negative_candidates(
        args, num_nodes, dataset["edges"], train_pos, split_info, "val", candidate_nodes=target_nodes
    )
    test_hard_negative_candidates = build_hard_negative_candidates(
        args, num_nodes, dataset["edges"], train_pos, split_info, "test", candidate_nodes=target_nodes
    )
    val_neg = sample_negative_edges(
        num_nodes,
        len(val_pos),
        all_pos_set,
        rng=rng,
        seed=seed,
        forbidden_edges=None,
        mode=args.negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=val_hard_negative_candidates,
        **negative_candidate_kwargs(split_info, "val", candidate_nodes=target_nodes),
    )
    test_neg = sample_negative_edges(
        num_nodes,
        len(test_pos),
        all_pos_set,
        rng=rng,
        seed=seed,
        forbidden_edges=val_neg,
        mode=args.negative_mode,
        reference_edges=train_pos,
        hard_negative_candidates=test_hard_negative_candidates,
        **negative_candidate_kwargs(split_info, "test", candidate_nodes=target_nodes),
    )
    validate_negative_edges(val_neg, all_pos_set, split_info, "val")
    validate_negative_edges(test_neg, all_pos_set, split_info, "test")

    val_edges, val_labels = build_eval_arrays(val_pos, val_neg)
    test_edges, test_labels = build_eval_arrays(test_pos, test_neg)

    # Inductive mode hides target-species PPI topology; orthogroup context remains visible.
    hetero_data, hetero_metadata = build_multispecies_hetero_data(dataset, train_pos)
    hetero_data = hetero_data.to(device)
    train_pos_tensor = torch.as_tensor(train_pos, dtype=torch.long, device=device)

    hyperparams = build_hyperparams(args, device)
    print(f"[seed={seed}] num_nodes={dataset['num_nodes']}", flush=True)
    print(
        f"[seed={seed}] num_edges={len(dataset['edges'])} "
        f"train_edges={len(train_pos)} val_edges={len(val_pos)} test_edges={len(test_pos)}",
        flush=True,
    )
    print(f"[seed={seed}] hyperparams: {json.dumps(hyperparams, sort_keys=True)}", flush=True)

    encoder = HGTEncoder(
        protein_input_dim=dataset["x"].shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.hgt_heads,
        layers=args.hgt_layers,
        dropout=args.hgt_dropout,
        metadata=hetero_data.metadata(),
        node_type_counts={"orthogroup": dataset["num_orthogroups"]},
    ).to(device)
    decoder = build_decoder(args, encoder.output_dim, device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(output_dir, f"seed_{seed}_best_model.pt")
    best_ap = -1.0
    best_val_auc = float("nan")
    best_epoch = 0
    wait = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        train_neg = sample_negative_edges(
            num_nodes,
            len(train_pos),
            all_pos_set,
            rng=rng,
            seed=seed,
            forbidden_edges=None,
            mode=args.negative_mode,
            reference_edges=train_pos,
            hard_negative_candidates=train_hard_negative_candidates,
            **negative_candidate_kwargs(split_info, "train", candidate_nodes=target_nodes),
        )
        validate_negative_edges(train_neg, all_pos_set, split_info, "train")
        train_neg_tensor = torch.as_tensor(train_neg, dtype=torch.long, device=device)

        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        z = encoder(hetero_data)
        pos_logits = decoder(z, train_pos_tensor)
        neg_logits = decoder(z, train_neg_tensor)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(train_pos_tensor), device=device),
                torch.zeros(len(train_neg_tensor), device=device),
            ],
            dim=0,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        val_scores = score_edges_hetero(encoder, decoder, hetero_data, val_edges, args.eval_batch_size, device)
        val_threshold = select_threshold(val_labels, val_scores)
        val_metrics = compute_metrics(val_labels, val_scores, val_threshold)
        val_auc = val_metrics["roc_auc"]
        print(
            f"[seed={seed}][epoch={epoch}/{args.epochs}] loss={train_loss:.6f} val_auc={format_metric(val_auc)}",
            flush=True,
        )

        if val_metrics["ap"] > best_ap + 1e-8:
            best_ap = val_metrics["ap"]
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 0
            checkpoint = {
                "encoder_state": encoder.state_dict(),
                "epoch": epoch,
                "threshold": float(val_threshold),
            }
            if args.decoder != "dot":
                checkpoint["decoder_state"] = decoder.state_dict()
            torch.save(checkpoint, checkpoint_path)
        else:
            wait += 1

        if wait >= args.patience:
            stopped_early = True
            print(
                f"[seed={seed}] early_stopping_triggered epoch={epoch} "
                f"reason=patience_exhausted best_val_auc={format_metric(best_val_auc)} best_epoch={best_epoch}",
                flush=True,
            )
            break

    if not stopped_early:
        print(f"[seed={seed}] training_finished_without_early_stopping final_epoch={args.epochs}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    if "decoder_state" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state"])

    val_scores = score_edges_hetero(encoder, decoder, hetero_data, val_edges, args.eval_batch_size, device)
    threshold = select_threshold(val_labels, val_scores)
    val_metrics = compute_metrics(val_labels, val_scores, threshold)

    test_scores = score_edges_hetero(encoder, decoder, hetero_data, test_edges, args.eval_batch_size, device)
    test_metrics = compute_metrics(test_labels, test_scores, threshold)

    record = {
        "model": "graphsage",
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
    graph_metadata(record, args, hetero_metadata)
    add_split_metadata(record, split_info)

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

    elapsed_sec = time.perf_counter() - seed_start_time
    print(f"[seed={seed}] elapsed_sec={elapsed_sec:.2f}", flush=True)

    seed_log = {
        "seed": int(seed),
        "elapsed_sec": float(elapsed_sec),
        "best_epoch": int(best_epoch),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
    }
    return record, seed_log, hetero_metadata


def main_graph():
    workflow_start_time = time.perf_counter()
    args = parse_graph_args()

    if args.experiment == "multispecies_hetero":
        if args.graph_type != "hetero":
            raise ValueError("Multispecies heterograph experiments require --graph-type hetero.")
        if args.encoder != "hgt":
            raise ValueError("Multispecies heterograph experiments currently require --encoder hgt.")
        if not args.protein_metadata:
            raise ValueError("Multispecies heterograph experiments require --protein-metadata.")
        if not args.ppi_edges:
            raise ValueError("Multispecies heterograph experiments require --ppi-edges.")
        if not args.protein_to_orthogroup:
            raise ValueError("Multispecies heterograph experiments require --protein-to-orthogroup.")

    if args.encoder == "hgt" and args.graph_type != "hetero":
        raise ValueError("Encoder 'hgt' requires --graph-type hetero.")
    if args.graph_type == "hetero" and args.encoder != "hgt":
        raise ValueError("Heterograph training currently requires --encoder hgt.")

    output_dir = args.output_dir or default_graph_output_dir(args.embeddings, experiment=args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    if args.experiment == "multispecies_hetero":
        from protlink.multispecies_graph_utils import build_multispecies_dataset

        dataset = build_multispecies_dataset(
            args.fasta,
            args.protein_metadata,
            args.ppi_edges,
            args.protein_to_orthogroup,
            args.embeddings,
            min_score=args.min_score,
            target_species=args.target_species,
        )
    else:
        dataset = build_homo_dataset(args.fasta, args.edges, args.embeddings, min_score=args.min_score)
    dataset["embedding_path"] = args.embeddings

    records = []
    seed_logs = []
    summary_extra_metadata = None
    for seed in args.seeds:
        if args.experiment == "multispecies_hetero":
            record, seed_log, hetero_metadata = train_graph_one_seed_multispecies(args, dataset, output_dir, seed)
            summary_extra_metadata = hetero_metadata
        elif args.graph_type == "hetero":
            record, seed_log, hetero_metadata = train_graph_one_seed_hetero(args, dataset, output_dir, seed)
            summary_extra_metadata = hetero_metadata
        else:
            record, seed_log = train_graph_one_seed(args, dataset, output_dir, seed)
        records.append(record)
        seed_logs.append(seed_log)
    summary = {
        "model": "graphsage",
        "embedding_path": args.embeddings,
        "embedding_name": Path(args.embeddings).stem,
        "seeds": [int(seed) for seed in args.seeds],
        "val": summarize_metric_dicts([record["val"] for record in records]),
        "test": summarize_metric_dicts([record["test"] for record in records]),
    }
    graph_metadata(summary, args, summary_extra_metadata)
    add_summary_split_metadata(summary, records)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    for seed_log in seed_logs:
        best_val_auc = "NA" if seed_log["best_val_auc"] is None else f"{seed_log['best_val_auc']:.6f}"
        print(
            f"[workflow] seed_summary seed={seed_log['seed']} "
            f"best_epoch={seed_log['best_epoch']} best_val_auc={best_val_auc} "
            f"elapsed_sec={seed_log['elapsed_sec']:.2f}",
            flush=True,
        )
    total_elapsed_sec = time.perf_counter() - workflow_start_time
    print(f"[workflow] total_elapsed_sec={total_elapsed_sec:.2f}", flush=True)



def main():
    parser = argparse.ArgumentParser(description="Unified training entry point.")
    parser.add_argument("trainer", choices=["mlp", "graphsage", "graph"], help="Training workflow to run.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if args.trainer == "mlp":
        main_mlp()
    else:
        main_graph()


if __name__ == "__main__":
    main()
