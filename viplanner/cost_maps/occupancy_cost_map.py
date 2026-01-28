import math
import os
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# 引入原有的配置类
# from omni.viplanner.config import GeneralCostMapConfig, TsdfCostMapConfig

# ==========================================
# 1. Mock Configuration Classes (模拟配置类)
# ==========================================
class MockGeneralConfig:
    def __init__(self):
        self.resolution = 0.1       # 0.1m 分辨率
        self.sigma_smooth = 2.0     # 平滑系数
        

class MockOgmConfig:
    def __init__(self):
        # 这里的参数在你的代码中部分被硬编码了，或者是未使用的
        self.obstacle_threshold = 0.5 
        self.sigma_expand = 1.0


class OccupancyCostMap:
    """
    Cost Map based on 2D Occupancy Grid Map (OGM).
    Generates a differentiable (smooth) cost surface suitable for gradient-based planning.
    """

    def __init__(self, cfg_general: MockGeneralConfig, cfg_ogm: MockOgmConfig):
        self._cfg_general = cfg_general
        self._cfg_ogm = cfg_ogm  # 复用 TSDF 的配置，主要用到 resolution, sigma_smooth 等
        
        # 地图元数据
        self.num_x = 0
        self.num_y = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self.resolution = cfg_general.resolution
        
        self.is_map_ready = False
        self.occupancy_grid = None  # 原始栅格数据
        self.cost_map = None      # 生成的代价地图

    def SetOccupancyGrid(self, ogm_array: np.ndarray, origin: tuple, resolution: float):
        """
        设置外部传入的占用栅格地图
        :param ogm_array: 2D numpy array, 值范围通常是 [0, 1] 或 [0, 100]. 
                          约定: 值越大表示越可能是障碍物。
        :param origin: (x, y) 地图左下角/中心在世界坐标系的位置
        :param resolution: 地图分辨率 (meters/cell)
        """
        self.occupancy_grid = ogm_array
        self.num_x, self.num_y = ogm_array.shape
        self.start_x, self.start_y = origin
        
        # 如果传入的分辨率和配置的不一样，可能需要在此处缩放 array，
        # 或者仅仅更新 self.resolution。为了简单，这里假设是一致的。
        if resolution != self._cfg_general.resolution:
            print(f"[WARN] Input resolution {resolution} differs from config {self._cfg_general.resolution}")
        
        self.is_map_ready = True
        print(f"Occupancy Map set with size: {self.num_x} x {self.num_y}")

    def CreateCostMap(self):
        """
        核心函数：将离散的占用栅格转换为平滑的 Cost Surface
        """
        if not self.is_map_ready or self.occupancy_grid is None:
            raise ValueError("Occupancy Grid not set!")

        # 1. 预处理：归一化并二值化
        # 假设输入是 [0, 100] (ROS标准) 或 [0, 1] (概率)
        # 我们定义障碍物阈值，例如 > 50 (0.5) 为障碍
        grid_normalized = self.occupancy_grid.copy()
        # 检查输入的栅格值是否只包含0或1
        unique_vals = np.unique(grid_normalized)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"Occupancy grid must contain only 0 and 1, but got values: {unique_vals}")

        # 检查地图中点是否为 free (0)
        mid_x = self.num_x // 2
        mid_y = self.num_y // 2
        if grid_normalized[mid_x, mid_y] != 0:
            raise ValueError("The center of the input occupancy grid must be free (0).")

        # 二值化障碍物和自由空间
        binary_obs = (grid_normalized == 1).astype(np.uint8)
        binary_free = (grid_normalized == 0).astype(np.uint8)
        
        # 方法 B (推荐): 基于距离变换的指数衰减 (Potential Field)
        # 计算到最近障碍物的距离 (像素单位)
        # inside (on obstacle): distance = 0
        dist_from_obs = ndimage.distance_transform_edt(binary_free)
        
        # 将像素距离转换为物理距离 (米)
        dist_from_obs_metric = dist_from_obs * self.resolution
        
        # 定义 Cost 函数: Cost = exp(-k * distance)
        # 距离越远，Cost 越小；距离为0，Cost 为 1
        # decay_factor 控制 Cost 衰减的快慢，类似膨胀半径
        # 假设我们希望在 0.5米处 Cost 衰减到很小
        decay_factor = 2.0  # 可调节参数，或者放到 config 中
        cost_map = np.exp(-decay_factor * dist_from_obs_metric)
        
        # 3. 强制障碍物内部为最大 Cost
        cost_map[binary_obs == 1] = 1.0
        
        # 4. 高斯平滑 (关键步骤)
        # 这一步是为了消除二值化带来的锯齿，并确保梯度不仅存在，而且平滑
        # sigma_smooth 通常取 1.0 到 3.0 之间
        self.cost_map = gaussian_filter(cost_map, sigma=self._cfg_general.sigma_smooth)
        
        # 5. 归一化与限幅 (可选)
        # 确保 Cost 在 [0, 1] 之间，这是 ViPlanner Loss 所期望的
        self.cost_map = np.clip(self.cost_map, 0.0, 1.0)
       
    def ReadPointFromFile(self):
        # 这是一个空接口，用于兼容 Trainer 代码的调用逻辑
        # 如果你的数据来源是文件（如 .png 或 .npy 的栅格图），可以在这里实现加载
        print("[INFO] OccupancyCostMap: Skipping PCD load, assuming Grid is set manually or via simpler loader.")
        pass

    def UpdateMapParams(self):
        # 兼容接口，实际参数在 SetOccupancyGrid 中已设置
        pass