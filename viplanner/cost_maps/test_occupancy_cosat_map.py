'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-22 21:27:11
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-27 20:04:38
FilePath: /viplanner/viplanner/cost_maps/test_occupancy_cosat_map.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os
from viplanner.cost_maps.occupancy_cost_map import OccupancyCostMap, MockGeneralConfig, MockOgmConfig
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter



        
def main():
    npy_path = "/home/eai/VLN/viplanner/rotated_out/carla/sample_01713/sem_mask.npy"
    
    # 2. 读取 NPY 文件
    # 这里的 raw_data: 1=Free, 0=Obstacle
    raw_data = np.load(npy_path)
    print(f"[TEST] Loaded map. Value range: [{raw_data.min()}, {raw_data.max()}]")
    print(f"[TEST] Map shape: {raw_data.shape}")

    # 3. 数据预处理 (关键步骤!)
    # 你的类要求: 0=Free, 1=Obstacle
    # 你的文件是: 1=Free, 0=Obstacle
    # 所以需要取反: input = 1 - raw_data
    processed_grid = 1 - raw_data
    
    # 确保类型是 int 或 float，并且只包含 0/1
    processed_grid = processed_grid.astype(int)
    
    print(f"[TEST] Inverted data for class input. New range: 0(Free) -> 1(Obs)")

    # 4. 初始化配置
    cfg_gen = MockGeneralConfig()
    cfg_ogm = MockOgmConfig()

    # 5. 计算 Origin (假设地图中心对应物理原点 (0,0))
    # start_x = - (rows * resolution) / 2
    # start_y = - (cols * resolution) / 2
    rows, cols = processed_grid.shape
    start_x = - (rows * cfg_gen.resolution) / 2.0
    start_y = - (cols * cfg_gen.resolution) / 2.0
    
    origin = (start_x, start_y)
    print(f"[TEST] Calculated Origin: {origin} based on center alignment.")

    # 6. 实例化并运行 CostMap
    cost_map_obj = OccupancyCostMap(cfg_gen, cfg_ogm)
    
    # 设置地图
    cost_map_obj.SetOccupancyGrid(
        ogm_array=processed_grid, 
        origin=origin, 
        resolution=cfg_gen.resolution
    )
    
    # 生成代价并显示
    print("[TEST] Running CreateCostMap (Check the Plot Window)...")
    cost_map_obj.CreateCostMap()
    
    print("[TEST] Done.")
    
    # 保存 costmap 为 npy 文件
    costmap_save_path = os.path.join(os.path.dirname(npy_path), "costmap.npy")
    np.save(costmap_save_path, cost_map_obj.cost_map)
    print(f"[TEST] Costmap saved to: {costmap_save_path}")
    # 清理测试文件
    # if os.path.exists(npy_path):
    #     os.remove(npy_path)

if __name__ == "__main__":
    main()