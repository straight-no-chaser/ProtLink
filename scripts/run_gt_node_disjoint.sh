#!/usr/bin/env bash
set -euo pipefail

python scripts/run_graphsage.py \
  --encoder gt \
  --gt-heads 4 \
  --gt-dropout 0.2 \
  --gt-layers 2 \
  --split-mode node_disjoint \
  "$@"
