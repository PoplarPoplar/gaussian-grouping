#!/usr/bin/env bash
set -euo pipefail

# Editable settings (直接在脚本顶部修改这些变量)
PYTHON_CMD="python"
DEFAULT_INPUT="/media/liu/my_pssd/program/data_milo_run/gaussian_group/gongan/cleaned_buildings.ply"
DEFAULT_VOXEL="0.2"
DEFAULT_OUTPUT="/media/liu/my_pssd/program/data_milo_run/gaussian_group/gongan/psdr_input.ply"

# Usage examples:
#   sh script/run_prepare_psdr_cuhk.sh
#   sh script/run_prepare_psdr_cuhk.sh --voxel 0.1
#   sh script/run_prepare_psdr_cuhk.sh --input /path/to/my.ply --voxel 0.05 --output /path/out.ply

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Accept overrides via flags or edit variables above
INPUT="${DEFAULT_INPUT}"
VOXEL="${DEFAULT_VOXEL}"
OUTPUT="${DEFAULT_OUTPUT}"
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2;;
    --voxel|--voxel_size)
      VOXEL="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --)
      shift; EXTRA_ARGS+=("$@"); break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

"${PYTHON_CMD}" "${SCRIPT_DIR}/../buildings/prepare_psdr.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --voxel_size "${VOXEL}" "${EXTRA_ARGS[@]}"
