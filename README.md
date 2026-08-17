# ProtLink: Multispecies HGT for interactome prediction

<p align="center">
  <img src="./asset/header.png" alt="ProtLink multispecies graph and HGT pipeline" width="100%">
</p>

ProtLink predicts protein–protein interactions (PPIs) by combining pretrained ESM2 sequence embeddings with graph context. The repository compares sequence-only and graph-based baselines, then extends the graph across species using orthogroups + HGT while keeping monospecies PPI prediction as the target task, and yielded **0.980 test AP** with ESM2-650M and **0.968 AP** under node-disjoint evaluation with multispecies HGT for *Homo sapiens*.

🤗 **Models / artifacts:** [ProtLink-Multispecies-HGT](https://huggingface.co/straight-no-chaser/ProtLink-Multispecies-HGT)

<p align="center">
  <img src="./asset/performance_heatmap.png" alt="ProtLink test AP across models and evaluation settings" width="95%">
</p>

## Install

```bash
pip install -r requirements.txt
```

> `torch` and `torch-geometric` may require platform-specific wheels.

## Data

### Single-species

- `seqs.fasta` — STRING protein IDs and sequences
- `[SPECIES].protein.physical.links.detailed.v12.0.txt` — STRING physical interactions
- `esm2_35m.npz` or `esm2_t33_650m.npz` — ESM2 protein embeddings

### Multispecies

- `proteins.fasta`
- `protein_metadata.tsv` — `protein_id`, `species_taxid`, `species_name`
- `ppi_edges.tsv` — intra-species PPIs
- `protein_to_orthogroup.tsv` — protein-to-orthogroup assignments
- `esm2_t33_650m.npz`

## Quick Start

Generate ESM2 embeddings:

```bash
python scripts/embed_esm2.py \
  --fasta seqs.fasta \
  --model facebook/esm2_t33_650M_UR50D \
  --output esm2_t33_650m_regenerated.npz
```

Run representative baselines:

```bash
python scripts/run_mlp.py --embeddings esm2_35m.npz --device auto
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto
python scripts/run_graphsage.py --embeddings esm2_35m.npz --encoder gt --device auto
```

Run the multispecies HGT model:

```bash
python scripts/run_graphsage.py \
  --experiment multispecies_hetero \
  --graph-type hetero \
  --encoder hgt \
  --decoder mlp \
  --fasta proteins.fasta \
  --protein-metadata protein_metadata.tsv \
  --ppi-edges ppi_edges.tsv \
  --protein-to-orthogroup protein_to_orthogroup.tsv \
  --embeddings esm2_t33_650m.npz \
  --target-species 9606 \
  --device auto
```

## Evaluation Options

```bash
# harder negatives
--negative-mode two_hop_hard

# held-out-node evaluation
--split-mode node_disjoint

# strict unseen-node -> seen-node evaluation
--split-mode node_inductive
```

## Disclaimer

This repository is a part of ongoing research efforts in molecular modeling in the [Cui Lab](https://cuilab.stanford.edu/research),
Stanford University.