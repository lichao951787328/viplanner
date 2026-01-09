#!/bin/bash
###
 # @Author: lichao951787328 951787328@qq.com
 # @Date: 2026-01-08 19:24:08
 # @LastEditors: lichao951787328 951787328@qq.com
 # @LastEditTime: 2026-01-09 15:57:09
 # @FilePath: /viplanner/run_data_collection.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# 运行 traversability grid 和语义数据采集，并生成俯视 RGB 图

# 设置环境变量
export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd
export VIPLANNER_SEMANTIC_MAP=/home/eai/VLN/viplanner/env/Collected_warehouse/keyword_mapping.yml
export VIPLANNER_DEBUG_SEM=1

# 清理旧输出
rm -rf ./output_with_semantics
mkdir -p ./output_with_semantics

rm -rf ./viplanner_debug
mkdir -p ./viplanner_debug

echo "=========================================="
echo "开始生成 Height Map + Semantic Mask + RGB"
echo "=========================================="

# 运行 demo (使用 0.1m 分辨率，采集 5m x 5m 区域)
python omniverse/standalone/traversability_grid_with_semantics_demo.py \
  --scene warehouse \
  --headless \
  --enable_cameras \
  --grid_res 0.1 \
  --save_dir ./output_with_semantics

echo ""
echo "=========================================="
echo "生成完成！检查输出文件："
echo "=========================================="
ls -lh ./output_with_semantics/

echo ""
echo "=========================================="
echo "数据统计："
echo "=========================================="
python3 -c "
import numpy as np
import os

os.chdir('./output_with_semantics')

if os.path.exists('height.npy'):
    height = np.load('height.npy')
    print(f'✓ height.npy: {height.shape}, range=[{np.nanmin(height):.2f}, {np.nanmax(height):.2f}]m')
else:
    print('✗ height.npy not found')

if os.path.exists('sem_mask.npy'):
    sem_mask = np.load('sem_mask.npy')
    print(f'✓ sem_mask.npy: {sem_mask.shape}, floor={sem_mask.sum()}/{sem_mask.size} ({100*sem_mask.sum()/sem_mask.size:.1f}%)')
else:
    print('✗ sem_mask.npy not found')

if os.path.exists('top_down_rgb.png'):
    from PIL import Image
    rgb = Image.open('top_down_rgb.png')
    print(f'✓ top_down_rgb.png: {rgb.size[0]}x{rgb.size[1]}')
else:
    print('✗ top_down_rgb.png not found')

if os.path.exists('height.png'):
    print('✓ height.png (visualization)')
if os.path.exists('sem_mask.png'):
    print('✓ sem_mask.png (visualization)')
"

echo ""
echo "=========================================="
echo "下一步："
echo "1. 查看可视化: python output_with_semantics/VIEW.py"
echo "2. 生成 RGB 叠加: python output_with_semantics/overlay_semantic_on_rgb.py"
echo "3. 生成 3D 点云: python output_with_semantics/visualize_3d.py"
echo "=========================================="
