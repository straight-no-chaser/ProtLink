#!/usr/bin/env bash
set -euo pipefail

python scripts/run_graphsage.py \
  --graph-type hetero \
  --encoder hgt \
  --hgt-heads 4 \
  --hgt-dropout 0.2 \
  --hgt-layers 2 \
  --decoder mlp \
  --negative-mode two_hop_hard \
  "$@"
