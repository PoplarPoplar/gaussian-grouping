#!/usr/bin/env bash
set -euo pipefail

# Editable settings (直接在脚本顶部修改这些变量)
PYTHON_CMD="python"
INPUT_PLY="/media/liu/my_pssd/program/data_milo_run/gaussian_group/gongan/pure_buildings.ply"
OUTPUT_PLY="/media/liu/my_pssd/program/data_milo_run/gaussian_group/gongan/cleaned_buildings.ply"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  EXTRA_ARGS+=("$1"); shift
done

"${PYTHON_CMD}" "${SCRIPT_DIR}/../buildings/clean_pointcloud.py" \
  --input "${INPUT_PLY}" \
  --output "${OUTPUT_PLY}" "${EXTRA_ARGS[@]}"
