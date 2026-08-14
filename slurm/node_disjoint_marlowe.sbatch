#!/usr/bin/env bash
#SBATCH -J sage_p2_g3
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
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

PROJECT_DIR=/projects/m000226/sage
DATA_DIR=${DATA_DIR:-$PROJECT_DIR}
EMBEDDINGS=${EMBEDDINGS:-$PROJECT_DIR/esm2_35m.npz}
NEGATIVE_MODE=${NEGATIVE_MODE:-random}
EMBED_NAME=$(basename "$EMBEDDINGS" .npz)
OUT_ROOT=${OUT_ROOT:-$PROJECT_DIR/outputs/phase2/group3_node_disjoint/${NEGATIVE_MODE}/$EMBED_NAME}

mkdir -p "$OUT_ROOT"
cd "$PROJECT_DIR"

python scripts/run_cosine.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --edges "$DATA_DIR/9606.protein.physical.links.detailed.v12.0.txt" \
  --embeddings "$EMBEDDINGS" \
  --split-mode node_disjoint \
  --negative-mode "$NEGATIVE_MODE" \
  --output-dir "$OUT_ROOT/cosine"

python scripts/run_mlp.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --edges "$DATA_DIR/9606.protein.physical.links.detailed.v12.0.txt" \
  --embeddings "$EMBEDDINGS" \
  --device cuda \
  --split-mode node_disjoint \
  --negative-mode "$NEGATIVE_MODE" \
  --output-dir "$OUT_ROOT/mlp"

python scripts/run_graphsage.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --edges "$DATA_DIR/9606.protein.physical.links.detailed.v12.0.txt" \
  --embeddings "$EMBEDDINGS" \
  --device cuda \
  --decoder mlp \
  --residual concat \
  --split-mode node_disjoint \
  --negative-mode "$NEGATIVE_MODE" \
  --output-dir "$OUT_ROOT/graphsage_mlp"
