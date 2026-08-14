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

## Experiment Layers

The repository keeps one codebase with two additive experiment layers:

- single-species baselines: cosine, MLP pair baseline, GraphSAGE, GT, and the existing heterograph HGT options
- multispecies heterograph experiments: cross-species protein + orthogroup graphs with human PPI as the prediction target

## Phase-2 Options

The existing defaults stay unchanged. The options below are strictly opt-in.

- GraphSAGE decoder: `--decoder mlp` switches from the default dot-product decoder to an MLP decoder over `z_u`, `z_v`, `abs(z_u - z_v)`, and `z_u * z_v`
- Residual fusion: `--residual concat` or `--residual add` augments GraphSAGE node representations; default is `none`
- Hard negatives: `--negative-mode two_hop_hard` prefers non-edges that are within two hops in the training graph, with fallback to random negatives when needed
- Harder split: `--split-mode node_disjoint` preferentially places edges involving held-out nodes into validation and test
- Strict inductive split: `--split-mode node_inductive` holds out validation/test protein nodes from target PPI training edges and evaluates unseen-node to seen-node interactions

Example fairer GraphSAGE comparison:

```bash
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --decoder mlp --residual concat
```

Example harder evaluation:

```bash
python scripts/run_mlp.py --embeddings esm2_35m.npz --device auto --negative-mode two_hop_hard
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --decoder mlp --residual concat --split-mode node_disjoint --negative-mode two_hop_hard
```

Strict cold-start node split:

```bash
python -m src.train mlp --embeddings esm2_35m.npz --fasta seqs.fasta --edges 9606.protein.physical.links.detailed.v12.0.txt --split-mode node_inductive
python -m src.train graph --embeddings esm2_35m.npz --fasta seqs.fasta --edges 9606.protein.physical.links.detailed.v12.0.txt --split-mode node_inductive
```

In `node_inductive`, training nodes and held-out validation/test nodes are disjoint. The training graph contains only train-node to train-node target PPI edges, while validation/test positives connect unseen validation/test proteins to seen training proteins. Held-out protein sequence embeddings remain available at evaluation time; their held-out target PPI labels and target PPI topology are not used for training message passing.

## Graph Transformer

The graph training pipeline also supports a Graph Transformer encoder without changing the existing GraphSAGE path.

- enable with `--encoder gt`
- GT-specific flags: `--gt-heads`, `--gt-dropout`, `--gt-layers`
- GT reuses the existing decoder, residual, split, and negative-sampling options

Examples:

```bash
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --encoder gt
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --encoder gt --negative-mode two_hop_hard
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --encoder gt --split-mode node_disjoint
```

## Heterograph / HGT Experiments

The graph training pipeline also supports an additive heterograph path for protein-protein prediction with pathway and domain context.

- enable with `--graph-type hetero --encoder hgt`
- required context files:
  - `--pathway-edges`: tabular `protein_id<TAB>pathway_id`
  - `--domain-edges`: tabular `protein_id<TAB>domain_id`
- prediction target remains protein-protein `interacts` links
- HGT-specific flags: `--hgt-heads`, `--hgt-dropout`, `--hgt-layers`

Examples:

```bash
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --graph-type hetero --encoder hgt --pathway-edges pathway_edges.tsv --domain-edges domain_edges.tsv --decoder mlp
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --graph-type hetero --encoder hgt --pathway-edges pathway_edges.tsv --domain-edges domain_edges.tsv --decoder mlp --negative-mode two_hop_hard
python scripts/run_graphsage.py --embeddings esm2_35m.npz --device auto --graph-type hetero --encoder hgt --pathway-edges pathway_edges.tsv --domain-edges domain_edges.tsv --decoder mlp --split-mode node_disjoint
```

## Multispecies Heterograph Experiments

The first multispecies version stays minimal and uses only:

- `protein` nodes
- `orthogroup` nodes
- intra-species PPI edges as graph context
- protein-to-orthogroup membership edges

Prediction targets are still human-human PPI edges. Non-human PPI edges and protein-orthogroup edges are context only.

Required multispecies inputs:

- `--fasta proteins.fasta`
  - FASTA headers are protein IDs, for example `>9606.ENSP00000000233`
- `--protein-metadata protein_metadata.tsv`
  - tab-separated columns: `protein_id`, `species_taxid`, `species_name`
- `--ppi-edges ppi_edges.tsv`
  - tab-separated columns: `protein1`, `protein2`, `species_taxid`, `combined_score`, `source`
  - only intra-species edges are expected
- `--protein-to-orthogroup protein_to_orthogroup.tsv`
  - tab-separated columns: `protein_id`, `orthogroup_id`
- `--embeddings esm2_t33_650m.npz`
  - uses the same embedding loader and `.npz` formats as the single-species workflows

Example metadata file:

```tsv
protein_id	species_taxid	species_name
9606.ENSP00000000233	9606	Homo_sapiens
10090.ENSMUSP00000012345	10090	Mus_musculus
```

Example PPI edge file:

```tsv
protein1	protein2	species_taxid	combined_score	source
9606.ENSP00000000233	9606.ENSP00000354587	9606	812	string
10090.ENSMUSP00000012345	10090.ENSMUSP00000054321	10090	901	string
```

Example orthogroup file:

```tsv
protein_id	orthogroup_id
9606.ENSP00000000233	OG0001234
10090.ENSMUSP00000012345	OG0001234
```

Basic multispecies HGT run:

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

Hard-negative multispecies run:

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
  --negative-mode two_hop_hard \
  --device auto
```

Node-disjoint multispecies run:

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
  --split-mode node_disjoint \
  --device auto
```

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
- `outputs/multispecies_hetero/<embedding_name>/summary.json`

## Summarize Results

```bash
python scripts/summarize_results.py "outputs/mlp/esm2_35m/seed_*_metrics.json"
```

You can also pass files from multiple models to get a compact comparison table.
