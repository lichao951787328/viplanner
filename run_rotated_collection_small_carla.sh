#!/bin/bash
###
 # @Author: GitHub Copilot
 # @Date: 2026-01-12
 # @Description: 小批量旋转视角数据采集（使用 Isaac Lab Python 运行）
###
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 设置环境变量
export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Carla/carla_export/new_carla_export/carla.usd
export VIPLANNER_SEMANTIC_MAP=/home/eai/VLN/viplanner/omniverse/extension/omni.viplanner/data/town01/keyword_mapping.yml
export VIPLANNER_DEBUG_SEM=1

# 默认参数
SCENE=${1:-carla}
NBR=${2:-8}
OUT=${3:-./rotated_out/carla}
SEED=${4:-1}

# 采集参数
GRID_RES=0.1
CAPTURE_SIZE=8.0
OFFSET_X=-8.0
OFFSET_Y=0.0
CAMERA_HEIGHT=15.0

# 调试目录
DEBUG_DIR="./viplanner_debug"

# ==========================================
# [关键修改] 设置 Isaac Lab 路径
# 根据你的报错路径推断，请确认此路径是否正确
ISAACLAB_PATH="/home/eai/VLN/Viplanner_isaaclab/IsaacLab"
ISAACLAB_PYTHON="$ISAACLAB_PATH/isaaclab.sh -p"
# ==========================================

echo "=========================================="
echo "Small-batch Rotated View Collection"
echo "=========================================="
echo "Using Isaac Lab at: $ISAACLAB_PATH"
echo "Scene: $SCENE"
echo "Output: $OUT"
echo "=========================================="

rm -rf "$OUT"
mkdir -p "$OUT"
rm -rf "$DEBUG_DIR"
mkdir -p "$DEBUG_DIR"

if [ -n "$SAMPLES_FILE" ] && [ -f "$SAMPLES_FILE" ]; then
	SAMPLES_CSV="$SAMPLES_FILE"
	echo "Using provided samples file: $SAMPLES_CSV"
else
	SAMPLES_CSV=""
	echo "No samples file provided. Will attempt inline sampling (Requires Isaac Sim)."
fi

# 定位 Python 脚本
PY_SCRIPT="$(dirname "$0")/omniverse/standalone/run_rotated_collection_small.py"
if [ ! -f "$PY_SCRIPT" ]; then
	PY_SCRIPT="$(pwd)/omniverse/standalone/run_rotated_collection_small.py"
fi

# 构建参数
if [ -n "$SAMPLES_CSV" ]; then
    PY_ARGS=(--samples_file "$SAMPLES_CSV" --out "$OUT" --scene "$SCENE" --num_samples "$NBR" --grid_res "$GRID_RES" --capture_size "$CAPTURE_SIZE" --camera_height "$CAMERA_HEIGHT" --headless)
else
    PY_ARGS=(--out "$OUT" --scene "$SCENE" --num_samples "$NBR" --grid_res "$GRID_RES" --capture_size "$CAPTURE_SIZE" --camera_height "$CAMERA_HEIGHT")
    # PY_ARGS=(--out "$OUT" --scene "$SCENE" --num_samples "$NBR" --grid_res "$GRID_RES" --capture_size "$CAPTURE_SIZE" --camera_height "$CAMERA_HEIGHT" --headless)
fi

echo "Calling: $ISAACLAB_PYTHON $PY_SCRIPT ${PY_ARGS[*]}"

# ==========================================
# [关键修改] 使用 Isaac Lab 的 Python 包装器运行
# ==========================================
$ISAACLAB_PYTHON "$PY_SCRIPT" "${PY_ARGS[@]}"

# echo ""
# echo "采集完成，输出目录预览： $OUT"
# ls -lh "$OUT" || true