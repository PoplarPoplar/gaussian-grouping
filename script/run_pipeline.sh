#!/usr/bin/env bash
set -euo pipefail

# Pipeline: train -> extract -> clean -> prepare
# Edit only SCENE and (optionally) BASE below to run on a different scene.

# -------------------
# Editable settings
# -------------------
PYTHON_CMD="python"
BASE="/media/liu/my_pssd/program/data_milo_run/gaussian_group"
SCENE="cuhk"   # <-- 修改这里切换场景

# train params
CONFIG_FILE="config/gaussian_dataset/train.json"
DENSIFY_UNTIL_ITER=0
R=4
SH_DEGREE=0

# prepare_psdr params
VOXEL_SIZE=0.2

# -------------------
# Derived paths (通常不需要修改)
# -------------------
SCENE_ROOT="${BASE}/${SCENE}"
MODEL_DIR="${SCENE_ROOT}/gp"

# train inputs
CUSTOM_PLY="${MODEL_DIR}/outputs/Tile_1/point_cloud/layer_22/Tile_1_point_cloud_clip.ply"

# extract inputs/outputs
EXTRACT_PLY_IN="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
EXTRACT_CLASSIFIER="${MODEL_DIR}/point_cloud/iteration_30000/classifier.pth"
EXTRACT_OUT="${SCENE_ROOT}/pure_buildings.ply"

# clean inputs/outputs
CLEAN_IN="${EXTRACT_OUT}"
CLEAN_OUT="${SCENE_ROOT}/cleaned_buildings.ply"

# prepare_psdr inputs/outputs
PREPARE_IN="${CLEAN_OUT}"
PREPARE_OUT="${SCENE_ROOT}/psdr_input.ply"

# -------------------
# Run steps
# -------------------
echo "[PIPELINE] Scene: ${SCENE}"
echo "[PIPELINE] Scene root: ${SCENE_ROOT}"

# 1) Train
echo "\n[STEP 1/4] Train: running train.py..."
"${PYTHON_CMD}" "$(dirname "$0")/../train.py" \
  -s "${SCENE_ROOT}" \
  -m "${MODEL_DIR}" \
  --custom_ply "${CUSTOM_PLY}" \
  --config_file "${CONFIG_FILE}" \
  --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
  -r "${R}" \
  --sh_degree "${SH_DEGREE}"

# 2) Extract buildings
echo "\n[STEP 2/4] Extract buildings: running buildings/extract_buildings.py..."
"${PYTHON_CMD}" "$(dirname "$0")/../buildings/extract_buildings.py" \
  --ply "${EXTRACT_PLY_IN}" \
  --classifier "${EXTRACT_CLASSIFIER}" \
  --out "${EXTRACT_OUT}"

# 3) Clean pointcloud
echo "\n[STEP 3/4] Clean pointcloud: running buildings/clean_pointcloud.py..."
"${PYTHON_CMD}" "$(dirname "$0")/../buildings/clean_pointcloud.py" \
  --input "${CLEAN_IN}" \
  --output "${CLEAN_OUT}"

# 4) Prepare PSDR
echo "\n[STEP 4/4] Prepare PSDR: running buildings/prepare_psdr.py..."
"${PYTHON_CMD}" "$(dirname "$0")/../buildings/prepare_psdr.py" \
  --input "${PREPARE_IN}" \
  --output "${PREPARE_OUT}" \
  --voxel_size "${VOXEL_SIZE}"

echo "\n[PIPELINE] 完成。输出 PSDR 文件: ${PREPARE_OUT}"
