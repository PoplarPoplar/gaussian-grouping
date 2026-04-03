#!/usr/bin/env bash
set -euo pipefail

# 一键流程：
# 1) 从单建筑PLY生成虚拟COLMAP数据集
# 2) 按 run_train_cuhk.sh 风格启动冻结参数训练
# 3) 调用 prepare/export_point_labels.py 导出带语义标签的高斯点云
# 选项：--gen-only（仅生成）、--train-only（仅训练）、--export-only（仅导出）、默认（全流程）

PYTHON_CMD="python"

# -------------------
# Editable settings（和 run_train_cuhk.sh 保持同类命名）
# -------------------
DATA_ROOT="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings_virtual_dataset"
MODEL_DIR="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings_grouping_run"
CUSTOM_PLY="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings/instance_001.ply"
CONFIG_FILE="config/gaussian_dataset/train.json"
DENSIFY_UNTIL_ITER=0
R=1
SH_DEGREE=0
NUM_OBJECTS=16

# 语义标签导出参数
LABEL_OUTPUT_DIR="${MODEL_DIR}/semantic_export"
COLORIZED_SUFFIX="_color"

# 流程控制开关：修改这里可直接控制是否生成或训练
# 设为 true 表示仅执行该步骤，false 表示执行对方步骤
GEN_ONLY=false    # true=仅生成虚拟数据集，false=生成+训练
TRAIN_ONLY=false  # true=仅训练，false=生成+训练
EXPORT_ONLY=false # true=仅导出语义标签（使用已有训练结果）
SKIP_EXPORT=false # true=跳过语义标签导出
# 注意：仅允许一个 *_ONLY 为 true；如果都为 false（推荐），则执行全流程

# 虚拟数据集生成参数
INPUT_BUILDING_DIR="/media/liu/my_pssd/program/data_milo_run/paco/hj/buildings"
PLY_FILE="${CUSTOM_PLY}"
NUM_ORBIT=24
NUM_SIDE_UP=8
NUM_TOP=1
WIDTH=1024
HEIGHT=1024
FOV_DEG=60.0
ZOOM_FACTOR=1.2
UP_AXIS="Z"
ORBIT_ELEVATION_DEG=0.0
SIDE_UP_ELEVATION_DEG=35.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 处理命令行参数，可覆盖脚本顶部的设置
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gen-only)
      GEN_ONLY=true
      TRAIN_ONLY=false
      shift
      ;;
    --train-only)
      TRAIN_ONLY=true
      GEN_ONLY=false
      EXPORT_ONLY=false
      shift
      ;;
    --export-only)
      EXPORT_ONLY=true
      GEN_ONLY=false
      TRAIN_ONLY=false
      SKIP_EXPORT=false
      shift
      ;;
    --skip-export)
      SKIP_EXPORT=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ONLY_COUNT=0
[[ "${GEN_ONLY}" == "true" ]] && ONLY_COUNT=$((ONLY_COUNT + 1))
[[ "${TRAIN_ONLY}" == "true" ]] && ONLY_COUNT=$((ONLY_COUNT + 1))
[[ "${EXPORT_ONLY}" == "true" ]] && ONLY_COUNT=$((ONLY_COUNT + 1))
if [[ "${ONLY_COUNT}" -gt 1 ]]; then
  echo "[ERROR] GEN_ONLY / TRAIN_ONLY / EXPORT_ONLY 只能有一个为 true。"
  exit 1
fi

if [[ ! -f "${PLY_FILE}" ]]; then
  echo "[ERROR] 找不到建筑 PLY: ${PLY_FILE}"
  echo "        先确认目录下是否存在 instance_001.ply，或直接修改脚本顶部的 CUSTOM_PLY。"
  exit 1
fi

mkdir -p "${DATA_ROOT}" "${MODEL_DIR}"

if [[ "${EXPORT_ONLY}" == "true" ]]; then
  GEN_ONLY=false
  TRAIN_ONLY=false
fi

echo "[MODE] GEN_ONLY=${GEN_ONLY}, TRAIN_ONLY=${TRAIN_ONLY}, EXPORT_ONLY=${EXPORT_ONLY}, SKIP_EXPORT=${SKIP_EXPORT}"

if [[ "${EXPORT_ONLY}" != "true" && "${TRAIN_ONLY}" != "true" ]]; then
  echo "[STEP 1/3] 生成虚拟 COLMAP 数据集"
  "${PYTHON_CMD}" "${ROOT_DIR}/prepare/prepare_virtual_colmap.py" \
    --ply "${PLY_FILE}" \
    --input_dir "${INPUT_BUILDING_DIR}" \
    --out_dir "${DATA_ROOT}" \
    --num_orbit "${NUM_ORBIT}" \
    --num_side_up "${NUM_SIDE_UP}" \
    --num_top "${NUM_TOP}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fov_deg "${FOV_DEG}" \
    --zoom_factor "${ZOOM_FACTOR}" \
    --orbit_elevation_deg "${ORBIT_ELEVATION_DEG}" \
    --side_up_elevation_deg "${SIDE_UP_ELEVATION_DEG}" \
    --up_axis "${UP_AXIS}"
else
  echo "[SKIP] 跳过 Step 1（生成虚拟数据集）"
fi

if [[ "${EXPORT_ONLY}" != "true" && "${GEN_ONLY}" != "true" ]]; then
  echo "[STEP 2/3] 启动冻结参数训练"
  "${PYTHON_CMD}" "${ROOT_DIR}/train.py" \
    -s "${DATA_ROOT}" \
    -m "${MODEL_DIR}" \
    --custom_ply "${CUSTOM_PLY}" \
    --config_file "${CONFIG_FILE}" \
    --densify_until_iter "${DENSIFY_UNTIL_ITER}" \
    -r "${R}" \
    --sh_degree "${SH_DEGREE}" \
    "${EXTRA_ARGS[@]}"
else
  echo "[SKIP] 跳过 Step 2（训练）"
fi

if [[ "${SKIP_EXPORT}" != "true" && "${GEN_ONLY}" != "true" ]]; then
  POINT_CLOUD_DIR="${MODEL_DIR}/point_cloud"
  if [[ ! -d "${POINT_CLOUD_DIR}" ]]; then
    echo "[ERROR] 未找到训练输出目录: ${POINT_CLOUD_DIR}"
    echo "        请先完成训练，或使用 --skip-export 跳过导出。"
    exit 1
  fi

  LATEST_ITER="$(find "${POINT_CLOUD_DIR}" -maxdepth 1 -type d -name 'iteration_*' -printf '%f\n' | sed 's/iteration_//' | sort -n | tail -1)"
  if [[ -z "${LATEST_ITER}" ]]; then
    echo "[ERROR] 未找到 iteration_* 目录，无法导出语义标签。"
    echo "        请先完成训练，或使用 --skip-export 跳过导出。"
    exit 1
  fi

  ITER_DIR="${POINT_CLOUD_DIR}/iteration_${LATEST_ITER}"
  IN_PLY="${ITER_DIR}/point_cloud.ply"
  IN_CLASSIFIER="${ITER_DIR}/classifier.pth"
  mkdir -p "${LABEL_OUTPUT_DIR}"
  OUT_PLY="${LABEL_OUTPUT_DIR}/point_cloud_labeled_iter_${LATEST_ITER}.ply"
  OUT_COLOR_PLY="${LABEL_OUTPUT_DIR}/point_cloud_labeled_iter_${LATEST_ITER}${COLORIZED_SUFFIX}.ply"

  if [[ ! -f "${IN_PLY}" || ! -f "${IN_CLASSIFIER}" ]]; then
    echo "[ERROR] 导出所需文件缺失："
    echo "        PLY: ${IN_PLY}"
    echo "        Classifier: ${IN_CLASSIFIER}"
    exit 1
  fi

  echo "[STEP 3/3] 导出并按类别上色高斯点云 (iteration_${LATEST_ITER})"
  "${PYTHON_CMD}" "${ROOT_DIR}/prepare/export_point_labels.py" \
    --ply "${IN_PLY}" \
    --classifier "${IN_CLASSIFIER}" \
    --out_ply "${OUT_PLY}" \
    --num_objects "${NUM_OBJECTS}"

  "${PYTHON_CMD}" "${ROOT_DIR}/prepare/colorize_point_labels.py" \
    --in_ply "${OUT_PLY}" \
    --out_ply "${OUT_COLOR_PLY}" \
    --overwrite_rgb

  echo "[DONE] 语义标签点云: ${OUT_PLY}"
  echo "[DONE] 类别上色点云: ${OUT_COLOR_PLY}"
fi
