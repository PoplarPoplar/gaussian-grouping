#!/usr/bin/env bash
set -euo pipefail

# Editable settings (直接在脚本顶部修改这些变量)
PYTHON_CMD="python"
DATA_ROOT="/media/liu/my_pssd/program/data_milo_run/gaussian_group/gongan"
MODEL_DIR="${DATA_ROOT}/gp"
CUSTOM_PLY="${DATA_ROOT}/outputs/Tile_1/point_cloud/layer_22/Tile_1_point_cloud_clip.ply"
CONFIG_FILE="config/gaussian_dataset/train.json"
DENSIFY_UNTIL_ITER=0
R=4
SH_DEGREE=0

# 执行目录与额外参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  EXTRA_ARGS+=("$1"); shift
done

"${PYTHON_CMD}" "${SCRIPT_DIR}/../train.py" \
  -s "${DATA_ROOT}" \
  -m "${MODEL_DIR}" \
  --custom_ply "${CUSTOM_PLY}" \
  --config_file "${CONFIG_FILE}" \
  --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
  -r "${R}" \
  --sh_degree "${SH_DEGREE}" "${EXTRA_ARGS[@]}"
