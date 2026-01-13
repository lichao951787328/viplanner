#!/bin/bash
###
 # @Author: GitHub Copilot
 # @Date: 2026-01-09
 # @Description: 运行旋转视角的 traversability grid 和语义数据采集
### 

# 运行旋转视角的 traversability grid 和语义数据采集，并生成旋转后的俯视 RGB 图

# 设置环境变量
export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd
export VIPLANNER_SEMANTIC_MAP=/home/eai/VLN/viplanner/env/Collected_warehouse/keyword_mapping.yml
export VIPLANNER_DEBUG_SEM=1

# 设置输出目录（与原版不同）
OUTPUT_DIR="./output_rotated_view"
DEBUG_DIR="./viplanner_debug"

# 清理旧输出
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

rm -rf "$DEBUG_DIR"
mkdir -p "$DEBUG_DIR"

echo "=========================================="
echo "开始生成 Rotated View Data"
echo "=========================================="
echo "  - Height Map (rotated view)"
echo "  - Semantic Mask (rotated view)"
echo "  - RGB Image (rotated camera)"
echo "  - Camera Pose (position + yaw)"
echo "=========================================="
echo ""

# 参数说明
SCENE="warehouse"
GRID_RES=0.1           # 网格分辨率（米）
CAPTURE_SIZE=8.0       # 采集区域大小（米）
OFFSET_X=-8.0          # X偏移（米）
OFFSET_Y=0.0           # Y偏移（米）
CAMERA_HEIGHT=8.0      # 相机高度（米）

# 可选：指定 yaw 角度（注释掉则随机生成）
YAW_DEG=45.0         # 固定 yaw 角度
# YAW_MIN=0.0          # 随机 yaw 最小值
# YAW_MAX=360.0        # 随机 yaw 最大值

echo "参数配置："
echo "  Scene: $SCENE"
echo "  Grid resolution: ${GRID_RES}m"
echo "  Capture size: ${CAPTURE_SIZE}m x ${CAPTURE_SIZE}m"
echo "  Offset: (${OFFSET_X}m, ${OFFSET_Y}m)"
echo "  Camera height: ${CAMERA_HEIGHT}m"
if [ -n "$YAW_DEG" ]; then
    echo "  Yaw angle: ${YAW_DEG}° (fixed)"
else
    echo "  Yaw angle: Random (0-360°)"
fi
echo ""

# 构建命令行参数
CMD="python omniverse/standalone/traversability_grid_rotated_view.py \
  --scene $SCENE \
  --headless \
  --enable_cameras \
  --grid_res $GRID_RES \
  --capture_size $CAPTURE_SIZE \
  --offset_x $OFFSET_X \
  --offset_y $OFFSET_Y \
  --camera_height $CAMERA_HEIGHT \
  --save_dir $OUTPUT_DIR"

# 添加可选的 yaw 参数
if [ -n "$YAW_DEG" ]; then
    CMD="$CMD --yaw_deg $YAW_DEG"
fi

if [ -n "$YAW_MIN" ]; then
    CMD="$CMD --yaw_min $YAW_MIN"
fi

if [ -n "$YAW_MAX" ]; then
    CMD="$CMD --yaw_max $YAW_MAX"
fi

# 运行数据采集
echo "执行命令："
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "生成完成！检查输出文件："
echo "=========================================="
ls -lh "$OUTPUT_DIR"/

echo ""
echo "=========================================="
echo "数据统计："
echo "=========================================="
python3 -c "
import numpy as np
import os

output_dir = '$OUTPUT_DIR'

# 检查 height map
height_path = os.path.join(output_dir, 'height.npy')
if os.path.exists(height_path):
    height = np.load(height_path)
    print(f'✓ height.npy: shape={height.shape}, range=[{np.nanmin(height):.2f}, {np.nanmax(height):.2f}]m')
else:
    print('✗ height.npy not found')

# 检查 semantic mask
sem_path = os.path.join(output_dir, 'sem_mask.npy')
if os.path.exists(sem_path):
    sem_mask = np.load(sem_path)
    traversable = sem_mask.sum()
    total = sem_mask.size
    print(f'✓ sem_mask.npy: shape={sem_mask.shape}, traversable={traversable}/{total} ({100*traversable/total:.1f}%)')
else:
    print('✗ sem_mask.npy not found')

# 检查 camera pose
pose_path = os.path.join(output_dir, 'camera_pose.npy')
if os.path.exists(pose_path):
    pose = np.load(pose_path, allow_pickle=True).item()
    pos = pose['position']
    yaw = pose['yaw_deg']
    print(f'✓ camera_pose.npy: position=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), yaw={yaw:.1f}°')
else:
    print('✗ camera_pose.npy not found')

# 检查 RGB image
rgb_path = os.path.join(output_dir, 'top_down_rgb.png')
if os.path.exists(rgb_path):
    from PIL import Image
    rgb = Image.open(rgb_path)
    print(f'✓ top_down_rgb.png: {rgb.size[0]}x{rgb.size[1]} pixels')
else:
    print('✗ top_down_rgb.png not found')

# 检查可视化图片
if os.path.exists(os.path.join(output_dir, 'height.png')):
    print('✓ height.png (visualization)')
if os.path.exists(os.path.join(output_dir, 'sem_mask.png')):
    print('✓ sem_mask.png (visualization)')
"

echo ""
echo "=========================================="
echo "障碍物 mesh 调试文件："
echo "=========================================="
if [ -f "$DEBUG_DIR/obstacle_mesh.obj" ]; then
    echo "✓ $DEBUG_DIR/obstacle_mesh.obj"
    ls -lh "$DEBUG_DIR/obstacle_mesh.obj"
fi
if [ -f "$DEBUG_DIR/obstacle_mesh.ply" ]; then
    echo "✓ $DEBUG_DIR/obstacle_mesh.ply"
    ls -lh "$DEBUG_DIR/obstacle_mesh.ply"
fi

echo ""
echo "=========================================="
echo "下一步："
echo "=========================================="
echo "1. 查看 RGB 图像:"
echo "   xdg-open $OUTPUT_DIR/top_down_rgb.png"
echo ""
echo "2. 生成 3D 可视化:"
echo "   python3 visualize_3d_map.py --input_dir $OUTPUT_DIR --output_dir ${OUTPUT_DIR}_3d"
echo ""
echo "3. 查看相机姿态:"
echo "   python3 -c \"import numpy as np; pose=np.load('$OUTPUT_DIR/camera_pose.npy', allow_pickle=True).item(); print(pose)\""
echo ""
echo "4. 批量采集不同角度:"
echo "   for yaw in 0 45 90 135 180 225 270 315; do"
echo "     python omniverse/standalone/traversability_grid_rotated_view.py --yaw_deg \$yaw --save_dir output_yaw_\$yaw"
echo "   done"
echo ""
echo "5. 检查障碍物 mesh (用 MeshLab):"
echo "   meshlab $DEBUG_DIR/obstacle_mesh.obj"
echo "=========================================="
