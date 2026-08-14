#!/usr/bin/env bash
#SBATCH -J sage_ms_hgt
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
FASTA=${FASTA:-$DATA_DIR/proteins.fasta}
PROTEIN_METADATA=${PROTEIN_METADATA:-$DATA_DIR/protein_metadata.tsv}
PPI_EDGES=${PPI_EDGES:-$DATA_DIR/ppi_edges.tsv}
PROTEIN_TO_ORTHOGROUP=${PROTEIN_TO_ORTHOGROUP:-$DATA_DIR/protein_to_orthogroup.tsv}
EMBEDDINGS=${EMBEDDINGS:-$PROJECT_DIR/esm2_t33_650m.npz}
TARGET_SPECIES=${TARGET_SPECIES:-9606}
EMBED_NAME=$(basename "$EMBEDDINGS" .npz)
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/outputs/multispecies_hgt/$EMBED_NAME}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

python scripts/run_graphsage.py \
  --experiment multispecies_hetero \
  --graph-type hetero \
  --encoder hgt \
  --decoder mlp \
  --fasta "$FASTA" \
  --protein-metadata "$PROTEIN_METADATA" \
  --ppi-edges "$PPI_EDGES" \
  --protein-to-orthogroup "$PROTEIN_TO_ORTHOGROUP" \
  --embeddings "$EMBEDDINGS" \
  --target-species "$TARGET_SPECIES" \
  --device cuda \
  --hgt-heads 4 \
  --hgt-dropout 0.2 \
  --hgt-layers 2 \
  --output-dir "$OUT_DIR"
