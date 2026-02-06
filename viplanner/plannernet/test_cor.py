'''
Author: lichao951787328 951787328@qq.com
Date: 2026-02-05 10:54:03
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-02-05 10:54:05
FilePath: /viplanner/viplanner/plannernet/test_cor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import graph

def verify_geodesic_logic():
    # 1. 模拟参数
    map_size = 80
    map_res = 0.1   # 8m x 8m 地图
    center = (40, 40)
    
    # 2. 创建一张空地图 (0=Free, 1=Obs)
    grid_map = np.zeros((map_size, map_size), dtype=np.float32)
    
    # 3. 添加一个障碍物 (一堵墙挡在前方)
    # 假设 X是前方(Row减), 墙在前方 2m 处 (20 pixel)
    # Row: 40 - 20 = 20
    # 墙宽一些
    grid_map[18:22, 20:60] = 1.0 
    
    # 4. 设定目标点 (在墙后面)
    # 前方 3m (30 pixel) -> Row = 10
    goal_meter = np.array([3.0, 0.0]) # x=3m, y=0m
    
    # --- 模拟你的 _compute_geodesic_distance 内部逻辑 ---
    
    # 坐标转换
    row_idx = int(round(center[0] - (goal_meter[0] / map_res)))
    col_idx = int(round(center[1] - (goal_meter[1] / map_res)))
    target_node = (row_idx, col_idx)
    
    # 欧氏距离
    euclidean = np.linalg.norm(goal_meter)
    
    # Cost Grid
    cost_grid = np.ones_like(grid_map)
    cost_grid[grid_map > 0.5] = 1000.0
    
    print(f"Start: {center}")
    print(f"Goal Meter: {goal_meter}")
    print(f"Goal Pixel: {target_node}")
    print(f"Euclidean Dist: {euclidean:.2f} m")
    
    path_indices = None
    geodesic_dist = 0.0
    
    try:
        # 运行 A*
        indices, weight = graph.route_through_array(
            cost_grid, start=center, end=target_node, 
            fully_connected=True, geometric=True
        )
        path_indices = np.array(indices).T # (2, N)
        geodesic_dist = weight * map_res
        print(f"✅ Geodesic Dist: {geodesic_dist:.2f} m")
    except Exception as e:
        print(f"❌ Path Finding Failed: {e}")

    # --- 可视化验证 ---
    plt.figure(figsize=(6, 6))
    # 显示地图
    plt.imshow(grid_map, cmap='gray_r', origin='upper') # White=Obs(1), Black=Free(0) if gray_r? 
    # 调整 cmap: 0=White, 1=Black
    plt.imshow(1-grid_map, cmap='gray', origin='upper') 
    
    # 画起点
    plt.plot(center[1], center[0], 'go', label='Start')
    # 画终点
    plt.plot(target_node[1], target_node[0], 'r*', markersize=15, label='Goal')
    
    # 画路径
    if path_indices is not None:
        plt.plot(path_indices[1], path_indices[0], 'b-', linewidth=2, label='A* Path')
    
    plt.title(f"A* Check\nEuclidean: {euclidean:.2f}m | Geodesic: {geodesic_dist:.2f}m")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    verify_geodesic_logic()