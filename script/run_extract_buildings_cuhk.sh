#!/usr/bin/env bash
set -euo pipefail

# Editable settings (直接在脚本顶部修改这些变量)
PYTHON_CMD="python"
PLY_PATH="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings_grouping_run/point_cloud/iteration_30000/point_cloud.ply"
CLASSIFIER_PATH="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings_grouping_run/point_cloud/iteration_30000/classifier.pth"
OUT_PATH="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings_grouping_run/pure_buildings.ply"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  EXTRA_ARGS+=("$1"); shift
done

"${PYTHON_CMD}" "${SCRIPT_DIR}/../buildings/extract_buildings.py" \
  --ply "${PLY_PATH}" \
  --classifier "${CLASSIFIER_PATH}" \
  --out "${OUT_PATH}" "${EXTRA_ARGS[@]}"
