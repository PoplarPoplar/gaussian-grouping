#!/usr/bin/env bash
set -euo pipefail

# Run clean_pointcloud for CUHK dataset
# Usage: sh script/run_clean_pointcloud_cuhk.sh [additional python args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"

"${PYTHON_CMD}" "${SCRIPT_DIR}/../buildings/clean_pointcloud.py" \
  --input /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/pure_buildings.ply \
  --output /media/liu/my_pssd/program/data_milo_run/gaussian_group/cuhk/cleaned_buildings.ply "$@"
