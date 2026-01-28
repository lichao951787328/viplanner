'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-22 09:37:54
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-22 15:15:17
FilePath: /viplanner/viplanner/utils/test_PlannerDataGenerator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

'''
import os
import torch
import numpy as np
import pypose as pp
from pathlib import Path
import cv2
from tqdm import tqdm
from PIL import Image
import shutil

from dataset_myself_ import PlannerDataGenerator


class DataCfg:
    def __init__(self):
        self.fov_angle_deg = (np.pi) / 2  # 90度
        self.fov_scale = 1.0
        self.obs_inflation_radius = 2
        self.goal_erosion_radius = 1
        self.map_resolution = 0.1  # 1像素 = 0.1米
        self.distance_scheme = {1.5: 0.3, 3.5: 0.35, 6.0: 0.2, 8.0: 0.15}  # 距离阈值: 采样占比
        self.ratio_fov_samples = 0.6
        self.ratio_front_samples = 0.3
        self.ratio_back_samples = 0.1
        self.pairs_per_image = 5
        self.ratio = 0.8
        self.max_train_pairs = 100
        
def run_test():
    test_root = "/home/eai/VLN/viplanner/rotated_out/carla"
    
    # 准备配置
    cfg = DataCfg()

    # 这里我们需要动态地给类绑定一些缺失的方法，或者在测试前修改类定义
    # 为了演示，我们直接实例化
    try:
        # 手动注入 mock 的辅助函数到类中
        import __main__
        # PlannerDataGenerator._local_to_pixel = _local_to_pixel_mock
        # 假设 map_imgs 是在 init 里生成的，或者在 get_pairs 里读取
        # 这里需要稍微修改 get_pairs 里的逻辑或者确保 map_imgs 存在
        
        print("--- Initializing PlannerDataGenerator ---")
        generator = PlannerDataGenerator(cfg, test_root)

        # 模拟 map_imgs 的加载（原代码中 init 没写这部分，但 get_pairs 用到了）
        generator.map_imgs = []
        for m_file in generator.map_filename_list:
            generator.map_imgs.append(np.load(m_file))

        print("\n--- Running get_pairs ---")
        # 此时会调用 get_path_distance_map 和 update_buffers_bulk
        # generator.get_pairs()

        for dist in cfg.distance_scheme.keys():
            scheme = generator.category_scheme_pairs[dist]
            if scheme.has_data:
                # 测试 get_data 是否能正常工作（模拟训练时的调用）
                # 假设我们需要 10 个数据
                o, g, img_d, img_s = scheme.get_data(nb_fov=5, nb_front=3, nb_back=2)
                assert o.shape[0] == 10, "Output odom count mismatch"
                assert g.shape[0] == 10, "Output goal count mismatch"
                print(f"Distance {dist} test passed!")

        print("\n--- Summary of DistanceSchemeIdx ---")
        for dist, container in generator.category_scheme_pairs.items():
            if container.has_data:
                # 统计这个距离桶里一共有多少个起点(Odom)
                num_odoms = len(container.odom_list)
                # 统计三个池子里的总目标数
                num_fov = sum(len(p) for p in container.fov_goals_pool)
                num_front = sum(len(p) for p in container.front_goals_pool)
                num_back = sum(len(p) for p in container.back_goals_pool)
                
                print(f"Distance Bin [{dist}m]:")
                print(f"  - Unique Odoms: {num_odoms}")
                print(f"  - Goals: FOV({num_fov}), Front({num_front}), Back({num_back})")
                
                # --- 测试采样功能 (get_data) ---
                try:
                    # 尝试每个类别采样 2 个点
                    o, g, img1, img2 = container.get_data(nb_fov=2, nb_front=2, nb_back=2)
                    print(f"  - Sample successful: Odom shape {o.shape}, Goal shape {g.shape}")
                except Exception as e:
                    print(f"  - Sample failed: {e}")
            else:
                print(f"Distance Bin [{dist}m]: No data collected.")
    except Exception as e:
        print(f"Test failed with exception: {e}")
    # finally:
    #     # 清理测试数据
    #     if os.path.exists(test_root):
    #         shutil.rmtree(test_root)

if __name__ == "__main__":
    run_test()