#!/usr/bin/env bash
#SBATCH -J sage_embed
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

PROJECT_DIR=/projects/m000226/protlink
DATA_DIR=${DATA_DIR:-$PROJECT_DIR}
MODEL_NAME=${MODEL_NAME:-facebook/esm2_t12_35M_UR50D}
OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_DIR/esm2_35m_regenerated.npz}

cd "$PROJECT_DIR"

python scripts/embed_esm2.py \
  --fasta "$DATA_DIR/seqs.fasta" \
  --model "$MODEL_NAME" \
  --output "$OUTPUT_PATH" \
  --device cuda
