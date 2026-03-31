#!/usr/bin/env bash
set -euo pipefail

# Run extract_buildings for CUHK dataset
# Usage: sh script/run_extract_buildings_cuhk.sh [additional python args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"

"${PYTHON_CMD}" "${SCRIPT_DIR}/../buildings/extract_buildings.py" \
  --ply /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/gp/point_cloud/iteration_30000/point_cloud.ply \
  --classifier /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/gp/point_cloud/iteration_30000/classifier.pth \
  --out /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/pure_buildings.ply "$@"
