import argparse
import csv
import glob
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize per-seed JSON result files.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def collect_files(inputs):
    files = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            files.extend(matches)
        elif os.path.isfile(item):
            files.append(item)
    return sorted(set(files))


def load_records(paths, split):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "seed" not in data or split not in data:
            continue
        label = f"{data.get('model', 'run')}:{data.get('embedding_name', 'embeddings')}"
        records.append((label, data[split]))
    return records


def summarize(records):
    grouped = {}
    for label, metrics in records:
        grouped.setdefault(label, []).append(metrics)

    rows = []
    for label in sorted(grouped):
        metric_list = grouped[label]
        row = {"label": label, "n": len(metric_list)}
        for metric_name in ["roc_auc", "ap", "precision", "recall", "f1"]:
            values = np.asarray([metrics[metric_name] for metrics in metric_list], dtype=float)
            row[metric_name] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        rows.append(row)
    return rows


def print_table(rows):
    columns = ["label", "n", "roc_auc", "ap", "precision", "recall", "f1"]
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row[column])))

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


def write_csv(path, rows):
    columns = ["label", "n", "roc_auc", "ap", "precision", "recall", "f1"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    files = collect_files(args.inputs)
    records = load_records(files, args.split)
    if not records:
        raise SystemExit("No per-seed result JSON files found.")

    rows = summarize(records)
    print_table(rows)

    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
