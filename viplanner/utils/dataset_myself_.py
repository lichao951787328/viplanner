'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-20 15:26:31
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-22 16:17:53
FilePath: /viplanner/viplanner/utils/dataset_myself_.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
from pathlib import Path
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple
import pypose as pp
from tqdm import tqdm
from viplanner.config import DataCfg
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
# --- 噪声增强函数定义 ---

# 增加终点随机角度
class PlannerData(Dataset):
    def __init__(
        self,
        cfg: DataCfg,
        transform,
    ) -> None:
        self._cfg = cfg
        self.transform = transform
        
        self.map_filename: List[str] = []
        
        # 内存缓存 (存 numpy 数组)
        self.map_imgs: List[np.ndarray] = []
        
        self.odom: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.fov_angle: float = 0.0
        self.load_ram: bool = False
        
        # 打印类参数
        for attr in dir(self._cfg):
            if not attr.startswith("__") and not callable(getattr(self._cfg, attr)):
                print(f"  cfg.{attr}: {getattr(self._cfg, attr)}")
        print(f"  cfg: {self._cfg}")
        print(f"  transform: {self.transform}")

    def update_buffers(self, map_filename, odom, goal):
        # 接收数据
        self.map_filename = map_filename
        self.odom = odom
        self.goal = goal

    def set_fov(self, fov_angle):
        self.fov_angle = fov_angle

    # --- 1. 加载到内存 (Clean Data) ---
    def load_data_in_memory(self) -> None:
        for idx in tqdm(range(len(self.map_filename)), desc="Load Maps into RAM"):
            img_clean = self._load_map_from_disk(idx)
            self.map_imgs.append(img_clean)
        self.load_ram = True

    def _load_map_from_disk(self, idx) -> np.ndarray:
        filename = self.map_filename[idx]
        
        # --- 1. 读取原始数据 ---
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # 读取图片
            image = np.array(Image.open(filename).convert('L'))
            if self._cfg.real_world_data:
                image = np.array(Image.fromarray(image).transpose(PIL.Image.ROTATE_180))
        else:  # .npy 文件
            image = np.load(filename)

        # --- 2. 归一化 (变成 0.0 ~ 1.0) ---  如果是npy格式的，就已经归一化过了
        # 只要是整数类型，就除以 255
        
        if filename.endswith(".png") or filename.endswith(".jpg"):
            if np.issubdtype(image.dtype, np.integer):
                image = image.astype("float32") / 255.0
            else:
                image = image.astype("float32")
                # 兼容 float 格式的 0-255 数据
                if image.max() > 1.001:
                    image = image / 255.0
        
        # 此时：
        # PNG: 255(白) -> 1.0
        # NPY: 1(True) -> 1.0
        # 它们现在的含义都是 "可通行"

        # --- 3. [核心修复] 语义对齐 ---
        # 你的数据源定义是: 1.0 / 255 = 可通行 (is_traversable)
        # 训练目标定义是:   0.0 = 可通行 (Cost 0)
        # 所以：必须对所有文件进行反转！
        
        image = 1.0 - image

        # --- 4. 截断与清理 ---
        image = np.clip(image, 0.0, 1.0)
        
        return image

    # def _apply_sensor_dropout(self, image, robot_pos, dropout_prob):
    #     h, w = image.shape
    #     # 创建遮挡掩码：1代表可见，0代表不可见
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     # A. 限制最大射程 (随机射程模拟)
    #     max_r = random.randint(h // 3, h // 1)
    #     cv2.circle(mask, (int(robot_pos[0]), int(robot_pos[1])), max_r, 1, -1)
    #     # B. 随机扇区丢包 (模拟Lidar受干扰)
    #     if random.random() < dropout_prob:
    #         for _ in range(random.randint(1, 2)):
    #             start_angle = random.randint(0, 360)
    #             end_angle = start_angle + random.randint(20, 90)
    #             # 在掩码中挖掉这个扇区
    #             cv2.ellipse(mask, (int(robot_pos[0]), int(robot_pos[1])), 
    #                         (h, w), 0, start_angle, end_angle, 0, -1)
    #     # 应用掩码：不可见区域设为 0 (或其他代表未知的数值)
    #     image[mask == 0] = 0.0
    #     return image

    def _apply_roughness_and_morph(self, image, roughness_prob, morph_prob):
        res = image.copy()
        # 边缘粗糙化
        if random.random() < roughness_prob:
            k = random.choice([3, 5])
            res = cv2.GaussianBlur(res, (k, k), 0)
            noise = np.random.randn(*res.shape).astype(np.float32) * 0.1
            res = (res + noise > 0.5).astype(np.float32)
        
        # 随机胖瘦变换
        if random.random() < morph_prob:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if random.random() > 0.5:
                res = cv2.dilate(res, kernel)
            else:
                res = cv2.erode(res, kernel)
        print(f"Roughness/Morph applied.")
        return res

    def _apply_ghosting(self, image, ghost_prob):
        if random.random() > ghost_prob:
            return image
        h, w = image.shape
        res = image.copy()
        # 随机撒 1-3 个小斑点
        for _ in range(random.randint(1, 3)):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            # 只有在空地上才加噪点
            if image[cy, cx] < 0.1:
                cv2.circle(res, (cx, cy), random.randint(1, 3), 1.0, -1)
        print(f"Ghosting applied.")
        return res

    # 避免分界线花的像刀一样整齐
    def _apply_blur(self, image, blur_prob):
        if random.random() > blur_prob:
            return image
        k = random.choice([3, 5])
        print(f"Blur applied with kernel size {k}.")
        return cv2.GaussianBlur(image, (k, k), random.uniform(0.5, 1.5))

    # def _add_random_polygons(self, image, nb_polygons, max_size):
    #     for i in range(nb_polygons):
    #         num_corners = random.randint(10, 20)
    #         polygon_points = np.random.randint(0, max_size, size=(num_corners, 2))
    #         x_offset = np.random.randint(0, image.shape[0])
    #         y_offset = np.random.randint(0, image.shape[1])
    #         polygon_points[:, 0] += x_offset
    #         polygon_points[:, 1] += y_offset
    #         # Create a convex hull from the points  # ... 生成随机顶点坐标 ...
    #         hull = cv2.convexHull(polygon_points)
    #         # Draw the hull on the image
    #         cv2.fillPoly(image, [hull], 0)
    #     return image

    # --- 3. 噪声增强逻辑 (集中管理) ---
    def _augment_map(self, grid_map: np.ndarray) -> np.ndarray:
        augmented_map = grid_map.copy()
        if getattr(self._cfg, "enable_map_noise", False):
            print("Applying map noise augmentation...")
            augmented_map = self._apply_roughness_and_morph(
                augmented_map,
                roughness_prob=0.5,
                morph_prob=0.5
            )
        # --- 3. 模拟动态物体残影 (Dynamic Ghosting) ---
        # 核心：在空地上随机出现一些小的障碍物点
        if getattr(self._cfg, "enable_ghosting", False):
            print("Applying ghosting augmentation...")
            augmented_map = self._apply_ghosting(
                augmented_map, 
                ghost_prob=getattr(self._cfg, "ghost_prob", 0.3)
            )
        # --- 4. 模拟概率模糊 (Probabilistic Blur) ---
        # 核心：让 0/1 边界变得模糊，模拟占用概率图
        if getattr(self._cfg, "enable_blur", False):
            print("Applying blur augmentation...")
            augmented_map = self._apply_blur(
                augmented_map, 
                blur_prob=getattr(self._cfg, "blur_prob", 0.3)
            )
        # --- 5. 随机黑色多边形遮挡 (原有逻辑) ---
        # if self._cfg.depth_random_polygons_nb > 0:
        #     augmented_map = self._add_random_polygons(
        #         augmented_map,
        #         self._cfg.depth_random_polygons_nb,
        #         self._cfg.depth_random_polygon_size,
        #     )
        return np.clip(augmented_map, 0.0, 1.0)
    
    # 这个貌似是在全局坐标系下进行抖动，但是实际上我规划都是在局部坐标系下进行的，局部坐标系下抖动的初始点为局部地图的中心，方向依旧是朝向正北
    def _apply_odom_jitter(self, idx: int) -> torch.Tensor:
        """对输入的 odom 进行抖动增强"""
        # odom_base = self.odom[idx]
        delta_x = random.uniform(-0.1, 0.1)
        delta_y = random.uniform(-0.1, 0.1)
        delta_yaw = random.uniform(-np.radians(5), np.radians(5))
        # odom_jittered = odom_base * pp.se3([delta_x, delta_y, 0, 0, 0, delta_yaw]).Exp()
        odom_jittered = pp.se3([delta_x, delta_y, 0, 0, 0, delta_yaw])
        return odom_jittered
    
    
    def generate_reachable_goal(self, grid_map=None, start_px=None):
        # 1. 获取原始 goal
        goal = self.goal
        if isinstance(goal, torch.Tensor) and goal.ndim > 1:
            # 如果是 batch，取第一个
            goal = goal[0]
        # 2. 增加位置噪声
        pos_noise = torch.tensor([
            random.uniform(-0.1, 0.1),  # x 方向噪声
            random.uniform(-0.1, 0.1),  # y 方向噪声
            0.0, 0.0, 0.0,
            random.uniform(-np.pi, np.pi)  # 随机方向角
        ], dtype=goal.dtype, device=goal.device)
        # 3. 构造新的 SE3
        noisy_goal = goal.clone()
        noisy_goal[:2] += pos_noise[:2]
        noisy_goal[5] = pos_noise[5]
        return noisy_goal
            
    def __len__(self):
        return len(self.map_filename)

    # --- 4. 获取数据 ---
    def __getitem__(self, idx):
        # A. 获取纯净数据
        if self.load_ram:
            # 必须 copy!
            grid_map = self.map_imgs[idx].copy()
        else:
            grid_map = self._load_map_from_disk(idx)

        odom_local_jittered = self._apply_odom_jitter(idx) 
        start_px = self._local_to_pixel(odom_local_jittered)
        target_px = self.generate_reachable_goal(grid_map, start_px)
          
        grid_map = self._augment_map(grid_map)
        map_tensor = torch.from_numpy(grid_map).unsqueeze(0)
        if self.transform:
            map_tensor = self.transform(map_tensor)
        
        return (
            map_tensor,
            0,  # 这里的第二个返回值原本是 sem_rgb，现在没用了，返回 0 或者 None 占位即可
            odom_local_jittered,
            target_px,
            self.pair_augment[idx],
        )
        
    def _local_to_pixel(self, local_odom: torch.Tensor) -> Tuple[int, int]:
        x = local_odom.translation()[0].item()
        y = local_odom.translation()[1].item()
        
        # 2. 获取地图参数
        h, w = self._cfg.local_map_size_pixels
        resolution = self._cfg.map_resolution
        
        # 3. 确定地图中心的像素坐标
        # 假设机器人理想位置在地图正中心
        center_y, center_x = h // 2, w // 2
        
        # 4. 坐标映射 (Physical -> Pixel)
        # 物理 X (前) 对应 图像 Up (Row 减小)
        # 物理 Y (左) 对应 图像 Left (Col 减小)
        py = int(center_y - (x / resolution))
        px = int(center_x - (y / resolution))
        
        # 5. 边界保护 (防止抖动过大跑出图片)
        py = np.clip(py, 0, h - 1)
        px = np.clip(px, 0, w - 1)
        
        return (py, px)  # 返回 (row, col)
    
    def _pixel_to_local_se3(self, pixel, current_odom=None):
        """像素 -> 局部物理坐标 (x, y, 0)"""
        py, px = pixel
        h, w = self._cfg.local_map_size_pixels
        res = self._cfg.map_resolution
        
        center_y, center_x = h // 2, w // 2
        
        local_x = (center_y - py) * res
        local_y = (center_x - px) * res
        
        # 随机化 yaw，范围为 [-pi, pi]
        random_yaw = random.uniform(-np.pi, np.pi)
        # 返回 SE3 格式的目标点（只设置 yaw，其余为 0）
        return pp.se3([local_x, local_y, 0.0, 0.0, 0.0, random_yaw])
        
        # return torch.tensor([local_x, local_y, 0.0])
  

def get_path_distance_map(grid_map, start_pixel):
    """
    计算从 start_pixel 到地图上所有像素的 Dijkstra 路径距离
    grid_map: 0 为障碍, 255 为空地
    """
    # print("Calculating path distance map...")
    # 检查 grid_map 必须为 0 或 1
    unique_vals = np.unique(grid_map)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"grid_map 必须只包含 0 或 1, 当前包含: {unique_vals}")

    h, w = grid_map.shape
    # 建立邻接矩阵 (只连接空地像素)
    # for row in range(grid_map.shape[0]):
    #     for col in range(grid_map.shape[1]):
    #         print(grid_map[row, col], end=' ')
    #     print()
    # 这里简化处理：只连上下左右 4 连通，也可以改为 8 连通
    def get_idx(r, c):
        return r * w + c

    center_value = grid_map[start_pixel[0], start_pixel[1]]
    # print(f"Center value at start pixel: {center_value}")
    nodes = []
    neighbors = []
    weights = []
    directions = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),  # 直行
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414) # 对角线
    ]
    for r in range(h):
        for c in range(w):
            if grid_map[r, c] != center_value:
                continue  # 障碍物跳过
            curr_idx = get_idx(r, c)
            # 检查邻居
            for dr, dc, weight in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid_map[nr, nc] == center_value:
                    nodes.append(curr_idx)
                    neighbors.append(get_idx(nr, nc))
                    weights.append(weight)  # 单位距离

    # 创建稀疏矩阵
    graph = csr_matrix((weights, (nodes, neighbors)), shape=(h * w, h * w))
    
    # 执行 Dijkstra
    start_idx = get_idx(start_pixel[0], start_pixel[1])
    dist_matrix = dijkstra(csgraph=graph, directed=False, indices=start_idx)
    
    # 返回重构为地图形状的距离图 (inf 表示不可达)
    return dist_matrix.reshape((h, w))


class DistanceSchemeIdx:
    def __init__(self, distance: float) -> None:
        self.distance: float = distance
        
        # 基础信息存储（每个起点 Odom 只存一次）
        # 存储的是odom的值
        self.odom_list: List[pp.LieTensor] = []
        # 存储的是对应的地图索引，通过上层类PlannerDataGenerator，在结合索引即可获得图片
        self.map_idx_list: List[str] = []
        
        # 终点池：List[List] 结构，外层索引对应 odom_list 的索引
        self.fov_goals_pool: List[List[pp.LieTensor]] = []
        self.front_goals_pool: List[List[pp.LieTensor]] = []
        self.back_goals_pool: List[List[pp.LieTensor]] = []
        
        # 辅助标记：用于快速找到哪些 odom 拥有特定类型的 goal
        self.has_fov: List[bool] = []
        self.has_front: List[bool] = []
        self.has_back: List[bool] = []

        self.has_data: bool = False

    def update_buffers_bulk(
        self,
        odom: pp.LieTensor,
        goals: List[pp.LieTensor],
        categories: List[str],  # 对应 goals 的类别：['fov', 'front', 'back']
        map_index: int,
    ) -> None:
        """
        添加一个起点及其对应的多个终点
        """
        # 1. 存入起点信息
        self.odom_list.append(odom)
        self.map_idx_list.append(map_index)
        
        # 2. 分类终点
        fovs, fronts, backs = [], [], []
        for goal, cat in zip(goals, categories):
            if cat == "fov":
                fovs.append(goal)
            elif cat == "front":
                fronts.append(goal)
            elif cat == "back":
                backs.append(goal)
            
        self.fov_goals_pool.append(fovs)
        self.front_goals_pool.append(fronts)
        self.back_goals_pool.append(backs)
        
        # 3. 更新辅助标记
        self.has_fov.append(len(fovs) > 0)
        self.has_front.append(len(fronts) > 0)
        self.has_back.append(len(backs) > 0)
        
        self.has_data = True

    def _sample_and_extract(self, category_mask: List[bool], pool: List[List[pp.LieTensor]], requested_nb: int):
        """
        内部辅助函数：采样 Odom 索引，并从对应的终点池中随机选一个目标
        """
        # 找到所有拥有该类别终点的 Odom 索引
        valid_odom_indices = np.where(category_mask)[0]
        
        if len(valid_odom_indices) == 0:
            return np.array([], dtype=np.int64), []

        # 采样 Odom 索引
        # 如果现有的 Odom 数量少于要求的数量，则允许重复采样 Odom
        replace = len(valid_odom_indices) < requested_nb
        selected_odom_indices = np.random.choice(valid_odom_indices, requested_nb, replace=replace)
        
        # 对于选中的每个 Odom，随机从它的池子里挑一个终点
        selected_goals = []
        for idx in selected_odom_indices:
            goal = random.choice(pool[idx])  # 你的核心思想：同类中随机选一个
            selected_goals.append(goal)
        # 这两个是一一对应的
        return selected_odom_indices, selected_goals

    def get_data(
        self,
        nb_fov: int,
        nb_front: int,
        nb_back: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], np.ndarray]:
        assert self.has_data, f"DistanceSchemeIdx for distance {self.distance} has no data"

        # 1. 分类采样 Odom 索引和 Goal
        idx_odom_fov, goals_fov = self._sample_and_extract(self.has_fov, self.fov_goals_pool, nb_fov)
        idx_odom_front, goals_front = self._sample_and_extract(self.has_front, self.front_goals_pool, nb_front)
        idx_odom_back, goals_back = self._sample_and_extract(self.has_back, self.back_goals_pool, nb_back)

        # 合并所有选中的起点索引和终点
        all_odom_idx = np.concatenate([idx_odom_fov, idx_odom_front, idx_odom_back])
        all_goals = goals_fov + goals_front + goals_back
        
        if len(all_goals) == 0:
            return torch.empty(0), torch.empty(0), [], []
            
        # 转换为 Tensor
        odom_tensor = torch.stack([self.odom_list[i] for i in all_odom_idx])
        goal_tensor = torch.stack(all_goals)
        
        # 对应图片列表（由于是一对多，一张图会被多次使用，这里根据采样索引取图）
        # 假设 depth 和 sem 图片路径相同或可从 map_filename 转换
        img_list = [self.map_idx_list[i] for i in all_odom_idx]
        # 这三者也会成为一一对应的形式
        return odom_tensor, goal_tensor, img_list


# 数据生成器
class PlannerDataGenerator(Dataset):
    # debug = False
    mesh_size = 0.5

    def __init__(
        self,
        cfg: DataCfg,
        root: str,
    ) -> None:
        # super().__init__()
        # set parameters
        self._cfg = cfg
        self.root = root
        self.fov_angle = cfg.fov_angle_deg  # in degrees 直接读取角度
        
        # init list for final odom, goal and img mapping
        self.map_filename_list = []
        self.odom: torch.Tensor = None
        self.pair_outside: np.ndarray = None
        self.pair_difficult: np.ndarray = None
        self.pair_within_fov: np.ndarray = None
        self.pair_front_of_robot: np.ndarray = None
        self.odom_array: pp.LieTensor = None
        self.map_array: np.ndarray = None
        self.category_scheme_pairs: Dict[float, DistanceSchemeIdx] = {}
        self.odom_used: int = 0
        self.odom_no_suitable_goals: int = 0

        self.debug = getattr(cfg, "debug_mode", True                       )

        # set parameters
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.load_dataset()  # 1. 加载数据
        # 2. 过滤掉离障碍物太近的轨迹点 (filter_obs_inflation)
        self.filter_obs_inflation()
        
        self.get_pairs()
        
        return

    """LOAD HELPER FUNCTIONS"""

    def load_dataset(self) -> None:
        """
        合并加载位姿和地图数据。
        扫描每一个 sample_XXXXX 文件夹，同步读取位姿和 sem_mask.npy。
        """
        print(f"[INFO] Loading dataset from {self.root}...")
        
        root_path = Path(self.root)
        # 查找所有以 'sample_' 开头的文件夹并排序，确保数据顺序一致
        sample_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])
        
        odom_list = []
        self.map_array = []
        
        for sample_dir in tqdm(sample_dirs, desc="Processing Samples"):
            pose_file = sample_dir / "camera_pose.txt"
            mask_file = sample_dir / "sem_mask.npy"
            
            # 检查必要文件是否存在
            if not pose_file.exists() or not mask_file.exists():
                print(f"[WARNING] Missing files in {sample_dir}, skipping...")
                continue
                
            # 1. 解析 camera_pose.txt
            try:
                with open(pose_file, "r") as f:
                    # 过滤掉注释行和空行，读取数值
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    # 预期的格式是 6 行：x, y, z, yaw, pitch, roll
                    vals = [float(v) for v in lines]
                
                if len(vals) < 6:
                    print(f"[ERROR] Invalid pose format in {pose_file}")
                    continue
                    
                x, y, z, yaw_deg, pitch_deg, roll_deg = vals[:6]
                
                # 2. 转换到位姿 (PyPose)
                # 转换角度为弧度
                yaw_rad = np.radians(yaw_deg)
                pitch_rad = np.radians(pitch_deg)
                roll_rad = np.radians(roll_deg)
                
                # 构建平移向量
                pos = torch.tensor([x, y, z], device=self._device)
                
                # 构建旋转 (PyPose euler2SO3 默认顺序通常是 roll, pitch, yaw)
                # 注意：请根据你的坐标系定义确认此处的顺序
                rot = pp.euler2SO3(torch.tensor([roll_rad, pitch_rad, yaw_rad], device=self._device))
                
                # 合成 SE3 位姿
                # pp.SE3 内部存储为 [x, y, z, qx, qy, qz, qw]
                se3 = pp.SE3(torch.cat([pos, rot.tensor()], dim=-1))
                
                odom_list.append(se3)
                
                # 3. 读取地图
                map = self._load_image(mask_file)  # 预加载地图到内存
                self.map_array.append(map)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {sample_dir}: {e}")
                continue

        if len(odom_list) == 0:
            raise RuntimeError(f"No valid data found in {self.root}")

        # 封装为 LieTensor 数组
        self.odom_array = torch.stack(odom_list)
        self.nb_odom_points = len(self.odom_array)

        print(f"[INFO] Successfully loaded {self.nb_odom_points} pairs of pose and mask.")

    # 过滤掉离障碍物太近的起点和终点
    def filter_obs_inflation(self) -> None:
        print("[INFO] Filtering odom points too close to obstacles...")
        inflation_radius = getattr(self._cfg, "obs_inflation_radius", 5)  # 膨胀半径（像素），可根据需要调整
        mask = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflation_radius + 1, 2 * inflation_radius + 1))
        for i in range(len(self.map_array)):
            map = self.map_array[i]
            # 膨胀障碍区域
            
            # if self.debug:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(map, cmap='gray')
            #     plt.title(f"Original Map {i}")
            #     plt.show()
            
            inflated_obs = cv2.dilate(map, kernel)
            
            # if self.debug:
            #     plt.imshow(inflated_obs, cmap='gray')
            #     plt.title(f"Inflated Map {i}")
            #     plt.show()
            
            # 获取当前起点像素坐标
            h, w = inflated_obs.shape
            py, px = h // 2, w // 2
            # 判断起点是否在膨胀障碍内
            if inflated_obs[py, px] >= 0.5:
                mask.append(False)
                # if self.debug:
                #     print(f"[DEBUG] Odom point {i} filtered out (too close to obstacle).")
            else:
                mask.append(True)
                # if self.debug:
                #     print(f"[DEBUG] Odom point {i} kept.")     
        
        # 将列表转换为 numpy 布尔数组，方便 PyTorch/PyPose 索引
        mask = np.array(mask)
        print(f"[INFO] {np.sum(mask)} odom points kept after inflation filtering.")
        print(f"[INFO] Total odom points after filtering: {len(self.odom_array)}")
        # 根据过滤结果更新 odom_array
        self.odom_array = self.odom_array[mask]
        self.map_array = [x for x, m in zip(self.map_array, mask) if m]
        print(f"[INFO] **** Total odom points after filtering: {len(self.odom_array)}")
 
    # 这个到底需要不需要啊？机器人mid360安装的较高，理论上盲区较小，应该把所有由障碍围成的封闭区域填充才对
    def generate_blind_zones(self):
        print("[INFO] Generating blind zones for each map...")
        for idx, map_file in enumerate(self.map_filename_list):
            # 读取地图
            if map_file.endswith(".npy"):
                grid_map = np.load(map_file)
            else:
                grid_map = np.array(Image.open(map_file).convert('L')).astype(np.float32) / 255.0
            # 归一化并反转（障碍为1，空地为0）
            binary_map = (1.0 - grid_map) > 0.5
            binary_map = binary_map.astype(np.uint8)
            h, w = binary_map.shape

            # 获取起点像素坐标
            odom = self.odom_array[idx]
            if hasattr(self, "_local_to_pixel"):
                py, px = self._local_to_pixel(odom)
            else:
                py, px = h // 2, w // 2

            # 发射射线，获取击中障碍物的点
            num_rays = 360
            ray_pts = []
            for angle in np.linspace(0, 2 * np.pi, num_rays, endpoint=False):
                for r in range(1, max(h, w)):
                    ry = int(py + r * np.sin(angle))
                    rx = int(px + r * np.cos(angle))
                    if 0 <= ry < h and 0 <= rx < w:
                        if binary_map[ry, rx] == 1:
                            ray_pts.append((ry, rx))
                            break

            # 连成多边形
            if len(ray_pts) < 3:
                print(f"[WARN] Not enough ray points for blind zone in map {map_file}")
                continue
            rr, cc = skimage.draw.polygon([pt[0] for pt in ray_pts], [pt[1] for pt in ray_pts], shape=binary_map.shape)
            visible_mask = np.zeros_like(binary_map)
            visible_mask[rr, cc] = 1

            # 腐蚀1个像素
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            visible_mask_eroded = cv2.erode(visible_mask, kernel)

            # 取反并填充盲区
            blind_zone = 1 - visible_mask_eroded
            # 将盲区填充到原图（比如设为障碍）
            filled_map = binary_map.copy()
            filled_map[blind_zone == 1] = 1

            # 可选：保存或更新地图
            # np.save(map_file.replace('.npy', '_blindzone.npy'), filled_map)
            # 或者 self.map_imgs[idx] = filled_map

        print("[INFO] Blind zone generation complete.")
                                 
    def _load_image(self, filename):
        if filename.suffix == ".npy":
            image = np.load(filename)
        else:
            image = np.array(Image.open(filename).convert('L')).astype("float32") / 255.0
              
        # 归一化
        binary_map = (1.0 - image) > 0.5
        binary_map = binary_map.astype(np.uint8)
        # if self.debug:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(binary_map, cmap='gray')
        #     plt.title(f"Loaded Image: {filename}")
        #     plt.show()
            
        return binary_map
        
    
    """GENERATE SAMPLES"""
    
    def get_pairs(self):
        print("[INFO] Generating goal pairs for each odom point...")
        print(f"DEBUG: odom_array length: {len(self.odom_array)}")
        print(f"DEBUG: map_array length: {len(self.map_array)}")
        print(f"DEBUG: map_filename_list length: {len(self.map_filename_list)}")
        self.category_scheme_pairs = {
            dist: DistanceSchemeIdx(distance=dist) for dist in self._cfg.distance_scheme.keys()
        }
        dist_thresholds = sorted(self._cfg.distance_scheme.keys())
        res = self._cfg.map_resolution
        cell_size_px = int(1.0 / res)  # 1米对应的像素数

        for odom_idx in range(len(self.odom_array)):
            current_odom = self.odom_array[odom_idx]
            map_img = self.map_array[odom_idx]
            h, w = map_img.shape
            center_y, center_x = h // 2, w // 2
            # print(f"[INFO] Processing Odom {odom_idx}...({center_y},{center_x})")
            free_area = (map_img < 0.5).astype(np.uint8)  # 0=障碍, 1=空地
            # if self.debug:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(free_area, cmap='gray')
            #     plt.title(f"Free Area Map for Odom {odom_idx}")
            #     plt.show()
                
            erosion_radius = getattr(self._cfg, "goal_erosion_radius", 3)
            if erosion_radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_radius + 1, 2 * erosion_radius + 1))
                free_area = cv2.erode(free_area, kernel)
            
            # if self.debug:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(free_area, cmap='gray')
            #     plt.title(f"Eroded Free Area Map for Odom {odom_idx}")
            #     plt.show()
            # if self.debug:
            #     for row in range(free_area.shape[0]):
            #         for col in range(free_area.shape[1]):
            #             print(free_area[row, col], end=' ')
            #         print()
            
            # 只保留腐蚀后仍为可行区域的像素
            grid_map = free_area.copy()
            path_dist_map = get_path_distance_map(grid_map, (center_y, center_x))
            
            # if self.debug:
            #     for row in range(path_dist_map.shape[0]):
            #         for col in range(path_dist_map.shape[1]):
            #             print(path_dist_map[row, col], end=' ')
            #         print()
            
            batch_data = {d: {'goals': [], 'cats': []} for d in dist_thresholds}
            for row in range(8):
                for col in range(8):
                    # 确定单元格像素范围
                    # print(f"[INFO] Sampling in Odom {odom_idx}, Cell ({row},{col})...")
                    y_min, y_max = row * cell_size_px, (row + 1) * cell_size_px
                    x_min, x_max = col * cell_size_px, (col + 1) * cell_size_px
                    
                    # 在当前单元格内寻找路径可达的点
                    cell_dist_slice = path_dist_map[y_min:y_max, x_min:x_max]
                    reachable_coords = np.argwhere((cell_dist_slice > 0.3) & (cell_dist_slice != np.inf))
                    
                    if len(reachable_coords) == 0:
                        # print(f"[INFO] Odom {odom_idx}, Cell ({row},{col}): No reachable coordinates found, skipping...")
                        continue
                    
                    # 3. 单元格内随机采样一个像素
                    selected_rel_coord = reachable_coords[np.random.choice(len(reachable_coords))]
                    py = y_min + selected_rel_coord[0]
                    px = x_min + selected_rel_coord[1]
                    # print(f"[INFO] Odom {odom_idx}, Cell ({row},{col}): Selected Pixel ({py},{px})")
                    
                    if self.debug:
                        import matplotlib.pyplot as plt
                        import matplotlib.patches as patches

                        fig, ax = plt.subplots(figsize=(6, 6))
                        # 1. 显示 path_dist_map
                        im = ax.imshow(path_dist_map, cmap='jet')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title(f"path_dist_map (odom {odom_idx})")

                        # 2. 画出当前cell的边框
                        rect = patches.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            linewidth=2,
                            edgecolor='lime',
                            facecolor='none'
                        )
                        ax.add_patch(rect)

                        # 3. 画出cell内所有reachable_coords
                        if len(reachable_coords) > 0:
                            # 转换为全局坐标
                            global_coords = reachable_coords + np.array([y_min, x_min])
                            ax.scatter(global_coords[:, 1], global_coords[:, 0], c='cyan', s=20, label='reachable_coords')

                        # 4. 画出采样点 selected_rel_coord
                        sel_py = y_min + selected_rel_coord[0]
                        sel_px = x_min + selected_rel_coord[1]
                        ax.scatter([sel_px], [sel_py], c='red', s=60, marker='*', label='selected_rel_coord')

                        ax.legend()
                        plt.show()
                    
                    # 4. 计算该点的真实路径距离
                    real_dist = path_dist_map[py, px] * res  # 像素单位转米
                    
                    # 寻找对应的距离桶 (Distance Bin) 采样的终点距离分类
                    # next((... if real_dist <= d), None):
                    # 这是一个生成器表达式。它会按顺序检查 dist_thresholds 里的每一个数字 d。
                    # 它的含义是： 找到第一个大于等于 real_dist 的阈值。
                    target_bin = next((d for d in dist_thresholds if real_dist <= d), None)
                    if target_bin is None:
                        # print(f"[WARN] Odom {odom_idx}, Cell ({row},{col}): Sampled goal distance {real_dist:.2f} m exceeds all defined thresholds.")
                        continue

                    # 5. 计算方位分类
                    # 转换到局部坐标系计算方位。图像坐标系转为机器人实际坐标系，默认机器人朝向为正 X 轴，左为正 Y 轴，上为正 Z 轴。此计算方式将图像坐标系转为机器人的坐标系
                    local_y = -(px - center_x) * res
                    local_x = -(py - center_y) * res
                    goal_in_robot = pp.SE3([local_x, local_y, 0, 0, 0, 0, 1])
                    # print(f"[DEBUG] Odom {odom_idx}, Cell ({row},{col}): Goal in robot frame: {goal_in_robot}")
                    is_fov, is_front, is_back = self.get_goal_categories(goal_in_robot.unsqueeze(0))
                    
                    cat = "fov" if is_fov[0] else ("front" if is_front[0] else "back")
                    
                    # 6. 转换回世界坐标系并存入临时 Batch，是不是不用存回去了，直接存像素坐标就行
                    batch_data[target_bin]['goals'].append(goal_in_robot)
                    batch_data[target_bin]['cats'].append(cat)

                    # if self.debug:
                    #     print(
                    #         f"[DEBUG] Odom {odom_idx}, Cell ({row},{col}): "
                    #         f"Selected Pixel ({py},{px}), Real Dist: {real_dist:.2f} m, "
                    #         f"Target Bin: {target_bin}, Category: {cat}"
                    #     )
            
            # print(f"[INFO] Odom {odom_idx}: Finished sampling all cells.")          
            # 7. 将此起点的所有采样终点批量更新到分类容器中
            for d_bin in dist_thresholds:
                if len(batch_data[d_bin]['goals']) > 0:
                    self.category_scheme_pairs[d_bin].update_buffers_bulk(
                        odom=current_odom,
                        goals=batch_data[d_bin]['goals'],
                        categories=batch_data[d_bin]['cats'],
                        map_index=odom_idx,
                    )
            # print(f"[INFO] Odom {odom_idx}: Updated distance scheme pairs.")
                    
    def get_goal_categories(self, goal_odom_frame: pp.LieTensor):
        """
        Decide which of the samples are within the fov, in front of the robot or behind the robot.
        """
        # get if odom-goal is within fov or outside the fov but still in front of the robot
        goal_angle = abs(torch.atan2(goal_odom_frame.data[:, 1], goal_odom_frame.data[:, 0]))
        within_fov = goal_angle < self.fov_angle / 2
        front_of_robot = goal_angle < torch.pi / 2
        front_of_robot[within_fov] = False

        behind_robot = ~front_of_robot.clone()
        behind_robot[within_fov] = False

        return within_fov, front_of_robot, behind_robot

    """SPLIT HELPER FUNCTIONS"""
    def split_samples(
        self,
        test_dataset: PlannerData,
        train_dataset: Optional[PlannerData] = None,
        generate_split: bool = False,
        ratio_fov_samples: Optional[float] = None,
        ratio_front_samples: Optional[float] = None,
        ratio_back_samples: Optional[float] = None,
        allow_augmentation: bool = True,
    ) -> None:
        # check if ratios are given or defaults are used
        ratio_fov_samples = ratio_fov_samples if ratio_fov_samples is not None else self._cfg.ratio_fov_samples
        ratio_front_samples = ratio_front_samples if ratio_front_samples is not None else self._cfg.ratio_front_samples
        ratio_back_samples = ratio_back_samples if ratio_back_samples is not None else self._cfg.ratio_back_samples
        assert round(ratio_fov_samples + ratio_front_samples + ratio_back_samples, 2) == 1.0, (
            "Sample ratios must sum up to 1.0, currently"
            f" {ratio_back_samples + ratio_front_samples + ratio_fov_samples}"
        )

        # max sample number
        if self._cfg.max_train_pairs:
            max_sample_number = min(
                int(self._cfg.max_train_pairs / self._cfg.ratio),
                int(self.odom_used * self._cfg.pairs_per_image),
            )
        else:
            max_sample_number = int(self.odom_used * self._cfg.pairs_per_image)

        # init buffers
        odom = torch.zeros((max_sample_number, 7), dtype=torch.float32)
        goal = torch.zeros((max_sample_number, 7), dtype=torch.float32)
        augment_samples = np.zeros((max_sample_number), dtype=bool)
        depth_filename = []
        sem_rgb_filename = []

        current_idx = 0
        for distance, distance_percentage in self._cfg.distance_scheme.items():
            if not self.category_scheme_pairs[distance].has_data:
                print(f"[WARN] No samples for distance {distance} in ENV" f" {os.path.split(self.root)[-1]}")
                continue

            # get number of samples
            buffer_data = self.category_scheme_pairs[distance].get_data(
                nb_fov=int(ratio_fov_samples * distance_percentage * max_sample_number),
                nb_front=int(ratio_front_samples * distance_percentage * max_sample_number),
                nb_back=int(ratio_back_samples * distance_percentage * max_sample_number),
                augment=allow_augmentation,
            )
            nb_samples = buffer_data[0].shape[0]

            # add to buffers
            odom[current_idx : current_idx + nb_samples] = buffer_data[0]
            goal[current_idx : current_idx + nb_samples] = buffer_data[1]
            depth_filename += buffer_data[2]
            sem_rgb_filename += buffer_data[3]
            augment_samples[current_idx : current_idx + nb_samples] = buffer_data[4]

            current_idx += nb_samples

        # cut off unused space
        odom = odom[:current_idx]
        goal = goal[:current_idx]
        augment_samples = augment_samples[:current_idx]

        # print data mix
        print(
            f"[INFO] datamix containing {odom.shape[0]} suitable odom-goal"
            " pairs: \n"
            "\t fov               :"
            f" \t{int(odom.shape[0] * ratio_fov_samples)  } ({round(ratio_fov_samples*100, 2)} %) \n"
            "\t front of robot    :"
            f" \t{int(odom.shape[0] * ratio_front_samples)} ({round(ratio_front_samples*100, 2)} %) \n"
            "\t back of robot     :"
            f" \t{int(odom.shape[0] * ratio_back_samples) } ({round(ratio_back_samples*100, 2)} %) \n"
            "from"
            f" {self.odom_used} ({round(self.odom_used/self.nb_odom_points*100, 2)} %)"
            " different starting points where \n"
            "\t non-suitable filter:"
            f" {self.odom_no_suitable_goals} ({round(self.odom_no_suitable_goals/self.nb_odom_points*100, 2)} %)"
        )

        # generate split
        idx = np.arange(odom.shape[0])
        if generate_split:
            train_index = sample(idx.tolist(), int(len(idx) * self._cfg.ratio))
            idx = np.delete(idx, train_index)

            train_dataset.update_buffers(
                depth_filename=[depth_filename[i] for i in train_index],
                sem_rgb_filename=([sem_rgb_filename[i] for i in train_index] if (self.semantics or self.rgb) else None),
                odom=odom[train_index],
                goal=goal[train_index],
                pair_augment=augment_samples[train_index],
            )
            train_dataset.set_fov(self.alpha_fov)

        test_dataset.update_buffers(
            depth_filename=[depth_filename[i] for i in idx],
            sem_rgb_filename=([sem_rgb_filename[i] for i in idx] if (self.semantics or self.rgb) else None),
            odom=odom[idx],
            goal=goal[idx],
            pair_augment=augment_samples[idx],
        )
        test_dataset.set_fov(self.alpha_fov)

        return

    """ Warp semantic on depth image helper functions"""


    """ Cleanup Script for files generated by this class"""

    def cleanup(self):
        print(
            ("[INFO] Cleaning up for environment" f" {os.path.split(self.root)[1]} ..."),
            end=" ",
        )
        # remove semantic_warp directory
        if os.path.isdir(os.path.join(self.root, "img_warp")):
            shutil.rmtree(os.path.join(self.root, "img_warp"))
        # remove depth_noise_edges directory
        if os.path.isdir(os.path.join(self.root, "depth_noise_edges")):
            shutil.rmtree(os.path.join(self.root, "depth_noise_edges"))
        print("Done!")
        return

