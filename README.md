# Protein-Protein Interaction Link Prediction

Minimal PyTorch and PyTorch Geometric project for human protein-protein interaction link prediction with:

- cosine similarity on pretrained ESM2 sequence embeddings
- an MLP on protein pair features
- a GraphSAGE encoder with dot-product link decoder

The project uses:

- `seqs.fasta` for STRING protein IDs and amino acid sequences
- `9606.protein.physical.links.detailed.v12.0.txt` for STRING physical interaction edges
- existing `.npz` ESM2 embeddings, or fresh embeddings generated from FASTA

## Inputs

Expected files at the repository root:

- `seqs.fasta`
- `9606.protein.physical.links.detailed.v12.0.txt`
- existing embeddings such as `esm2_35m.npz` or `esm2_t33_650m.npz`

The embedding loader supports:

- format A: one `.npz` key per protein ID
- format B: `ids` plus `embeddings`
- the provided files also use `ids` plus `emb`

## Install

```bash
pip install -r requirements.txt
```

`torch` and `torch-geometric` may need platform-specific wheels depending on your environment.

## Generate ESM2 Embeddings

```bash
python scripts/embed_esm2.py --fasta seqs.fasta --model facebook/esm2_t12_35M_UR50D --output esm2_35m_regenerated.npz
```

For the 650M model:

```bash
python scripts/embed_esm2.py --fasta seqs.fasta --model facebook/esm2_t33_650M_UR50D --output esm2_t33_650m_regenerated.npz
```

The script mean-pools final-layer residue embeddings while excluding special tokens and saves:

- `ids`
- `embeddings`

## Run Experiments

Cosine baseline:

```bash
python scripts/run_cosine.py --embeddings esm2_35m.npz
```

MLP baseline:

```bash
python scripts/run_mlp.py --embeddings esm2_35m.npz --device auto
```

GraphSAGE:

```bash
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto
```

You can swap in `esm2_t33_650m.npz` or a regenerated embedding file.

Each script runs seeds `0 1 2` by default and writes per-seed metrics plus an aggregate summary under `outputs/`.

## Data Processing

- parse `seqs.fasta`
- parse physical links from `9606.protein.physical.links.detailed.v12.0.txt`
- keep only proteins present in both FASTA and embeddings
- keep undirected edges only
- remove duplicates and self-loops
- filter to `combined_score >= 700`

Positive edges are split into train/validation/test as 80/10/10 with a simple safeguard that tries to keep at least one train edge per node when possible.

Negative edges are sampled from unlabeled non-edges with:

- no self-loops
- no overlap with positive edges
- undirected edge convention
- fixed validation/test negatives
- dynamic train negatives re-sampled every epoch for trainable models

Important: sampled negatives are unlabeled non-edges, not confirmed biological negatives.

## Outputs

Each run saves:

- metrics as JSON
- optional predictions as CSV with `--save-preds`
- best model checkpoint for MLP and GraphSAGE

Examples:

- `outputs/cosine/<embedding_name>/seed_0_metrics.json`
- `outputs/mlp/<embedding_name>/seed_0_best_model.pt`
- `outputs/graphsage/<embedding_name>/summary.json`

## Summarize Results

```bash
python scripts/summarize_results.py "outputs/mlp/esm2_35m/seed_*_metrics.json"
```

You can also pass files from multiple models to get a compact comparison table.
