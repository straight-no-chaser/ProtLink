#!/usr/bin/env bash
#SBATCH -J sage_hgt_node
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export HF_HOME=/projects/m000226/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJECT_DIR=/projects/m000226/protlink
DATA_DIR=${DATA_DIR:-$PROJECT_DIR}
EMBEDDINGS=${EMBEDDINGS:-$PROJECT_DIR/esm2_t33_650m.npz}
PATHWAY_EDGES=${PATHWAY_EDGES:-$PROJECT_DIR/pathway_edges.tsv}
DOMAIN_EDGES=${DOMAIN_EDGES:-$PROJECT_DIR/domain_edges.tsv}
EMBED_NAME=$(basename "$EMBEDDINGS" .npz)
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/outputs/hgt_node_disjoint/$EMBED_NAME}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

python scripts/run_graphsage.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --edges "$DATA_DIR/9606.protein.physical.links.detailed.v12.0.txt" \
  --embeddings "$EMBEDDINGS" \
  --graph-type hetero \
  --encoder hgt \
  --pathway-edges "$PATHWAY_EDGES" \
  --domain-edges "$DOMAIN_EDGES" \
  --device cuda \
  --decoder mlp \
  --hgt-heads 4 \
  --hgt-dropout 0.2 \
  --hgt-layers 2 \
  --split-mode node_disjoint \
  --output-dir "$OUT_DIR"
