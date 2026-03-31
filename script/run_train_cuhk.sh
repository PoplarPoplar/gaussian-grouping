#!/usr/bin/env bash
set -euo pipefail

# Run training for CUHK dataset (paths from user request)
# Usage: sh script/run_train_cuhk.sh [additional python args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"

"${PYTHON_CMD}" "${SCRIPT_DIR}/../train.py" \
  -s /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk \
  -m /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/gp \
  --custom_ply /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/outputs/Tile_1/point_cloud/layer_22/Tile_1_point_cloud_clip.ply \
  --config_file config/gaussian_dataset/train.json \
  --densify_until_iter 0 \
  -r 4 \
  --sh_degree 0 "$@"
