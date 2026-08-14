#!/usr/bin/env bash
#SBATCH -J sage_cosine
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import numpy; print('numpy', numpy.__version__)"

export HF_HOME=/projects/m000226/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJECT_DIR=/projects/m000226/protlink
DATA_DIR=${DATA_DIR:-$PROJECT_DIR}
EMBEDDINGS=${EMBEDDINGS:-$PROJECT_DIR/esm2_t33_650m.npz}
EMBED_NAME=$(basename "$EMBEDDINGS" .npz)
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/outputs/cosine/$EMBED_NAME}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

python scripts/run_cosine.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --edges "$DATA_DIR/9606.protein.physical.links.detailed.v12.0.txt" \
  --embeddings "$EMBEDDINGS" \
  --output-dir "$OUT_DIR"
