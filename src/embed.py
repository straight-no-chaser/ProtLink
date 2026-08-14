import argparse
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data_utils import read_fasta
from src.training_utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="seqs.fasta")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def mean_pool_without_special_tokens(hidden_states, attention_mask):
    # Mean pooling of residue-level emb to protein-level per-sequence emb
    # Use mean pooling for stable inference and low cost, can switch to attention pooling or domain-aware pooling later
    pooled = []
    lengths = attention_mask.sum(dim=1)

    for row_idx in range(hidden_states.size(0)):
        valid = attention_mask[row_idx].bool().clone()
        seq_len = int(lengths[row_idx].item())
        if valid.sum() > 0:
            valid[0] = False
        if seq_len > 1:
            valid[seq_len - 1] = False
        if valid.any():
            pooled.append(hidden_states[row_idx][valid].mean(dim=0))
        else:
            pooled.append(hidden_states[row_idx][attention_mask[row_idx].bool()].mean(dim=0))

    return torch.stack(pooled, dim=0)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    sequences = list(read_fasta(args.fasta).items())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    model_max_length = getattr(model.config, "max_position_embeddings", None)
    max_length = args.max_length if args.max_length is not None else model_max_length

    all_ids = []
    all_embeddings = []

    for start in tqdm(range(0, len(sequences), args.batch_size), desc="Embedding"):
        batch = sequences[start : start + args.batch_size]
        batch_ids = [protein_id for protein_id, _ in batch]
        batch_sequences = [sequence for _, sequence in batch]

        tokenizer_kwargs = {
            "padding": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_length

        inputs = tokenizer(batch_sequences, **tokenizer_kwargs)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pool_without_special_tokens(outputs.last_hidden_state, inputs["attention_mask"])
        all_ids.extend(batch_ids)
        all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.savez_compressed(args.output, ids=np.asarray(all_ids), embeddings=embeddings)


if __name__ == "__main__":
    main()
