# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math

# python
import os
import random
import shutil
from pathlib import Path
from random import sample
from typing import Dict, List, Optional, Tuple
# 引入图像处理、图算法、数学库、3D可视化库
import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import PIL
import pypose as pp   # 机器人位姿处理库
import scipy.spatial.transform as tf
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.kdtree import KDTree  # 用于快速查找最近邻
from skimage.util import random_noise  # 用于添加图像噪声
from torch.utils.data import Dataset
from tqdm import tqdm  # 进度条

# implerative-planner-learning # 引入项目内的配置和地图处理类
from omni.viplanner.config import DataCfg
from omni.viplanner.cost_maps import CostMapPCD

# set default dtype to float32
torch.set_default_dtype(torch.float32)


class PlannerData(Dataset):
    def __init__(
        self,
        cfg: DataCfg,
        semantics: bool = False,
    ) -> None:
        self._cfg = cfg
        self.semantics = semantics

        # init buffers # 初始化数据缓存列表
        self.sem_imgs: List[torch.Tensor] = []
        self.odom: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_augment: np.ndarray = None
        self.fov_angle: float = 0.0
        self.load_ram: bool = False
        return
    
    # 填充数据到缓存
    def update_buffers(
        self,
        sem_rgb_filename: List[str],
        odom: torch.Tensor,
        goal: torch.Tensor,
        pair_augment: np.ndarray,
    ) -> None:
        #  这个函数通常由 Generator 调用，将生成好的数据列表注入到 Dataset 中
        self.sem_rgb_filename = sem_rgb_filename
        self.odom = odom
        self.goal = goal
        self.pair_augment = pair_augment  # 标记哪些样本需要进行水平翻转增强
        return

    def set_fov(self, fov_angle):
        self.fov_angle = fov_angle
        return

    """Augment Images with black polygons"""

    # 数据增强的辅助函数，具体过程如下：
    # 模拟视线遮挡--机器人看不到被遮挡的区域
    # 模拟建图噪声--由于使用occ得到的栅格地图，应该没有椒盐噪声，但有边缘粗糙化
    # 模拟定位误差--由于视觉里程计的误差，可能会导致部分区域信息缺失
    # 概率模糊--二值的仿真地图中，锐利的边缘变成渐变边缘
    # 动态物体残影--在真实 SLAM 中，移动物体可能会在地图上留下一串“鬼影”或者断断续续的障碍点
    # 模拟丢包和视场受限--相机视场外的区域无法观测到，部分图像区域可能因为传输问题而丢失，视场内被遮挡的部分也无法观测
    def _add_obstruction_regions(self, image, robot_pos, num_rays=720, max_range=None):
        """
        Simulate Lidar-like visibility using Ray Casting and Polygon Filling.
        
        Args:
            image: 2D numpy array (H, W), raw occupancy map (1=obstacle, 0=free)
                Note: Input image should be binary (0 or 1).
            robot_pos: (x, y) tuple, robot center in pixel coordinates
            num_rays: number of rays (increase to 720 or 1080 for better precision)
            max_range: maximum range in pixels
            
        Returns:
            vis_image: 2D numpy array with:
                    0.0 = Free (visible space)
                    0.5 = Unknown (occluded/outside FOV)
                    1.0 = Occupied (obstacles)
        """
        h, w = image.shape
        x0, y0 = robot_pos
        
        # 1. 初始化输出图像为全灰 (Unknown = 0.5)
        vis_image = np.full_like(image, 0.5, dtype=np.float32)
        
        if max_range is None:
            max_range = np.sqrt(h**2 + w**2)
        
        # 2. 生成所有射线的角度
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        # 3. 预计算射线的采样点 (向量化操作替代内层循环)
        # 生成从 0 到 max_range 的距离向量
        r_step = 1.0 # 步长，越小越精确
        radii = np.arange(0, max_range, r_step)
        
        # 击中点列表 (用于构建可视区域多边形)
        hit_points = []
        
        # 为了加速，我们可以将所有射线放在一个大矩阵里计算，
        # 但为了代码可读性和内存平衡，这里对每条射线做向量化是足够快的
        
        for theta in angles:
            # 计算该射线路径上所有点的坐标 (X, Y)
            # x = x0 + r * cos(theta)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            xs = x0 + radii * cos_t
            ys = y0 + radii * sin_t
            
            # 转换为整数索引
            xs_int = np.rint(xs).astype(np.int32)
            ys_int = np.rint(ys).astype(np.int32)
            
            # 边界检查：找到第一个跑出地图的点
            valid_mask = (xs_int >= 0) & (xs_int < w) & (ys_int >= 0) & (ys_int < h)
            
            # 如果整条射线一开始就出界了(不太可能)，直接跳过
            if not valid_mask[0]:
                hit_points.append((int(x0), int(y0)))
                continue
                
            # 截取在地图内的部分
            valid_len = np.sum(valid_mask)
            xs_int = xs_int[:valid_len]
            ys_int = ys_int[:valid_len]
            
            # 从地图中提取这些点的值
            # image[y, x] 注意行列顺序
            path_values = image[ys_int, xs_int]
            
            # 4. 寻找碰撞点 (argmax 找到第一个为 1 的索引)
            # path_values == 1 生成布尔数组，argmax 返回第一个 True 的下标
            obstacle_indices = np.where(path_values == 1)[0]
            
            if len(obstacle_indices) > 0:
                # 击中了障碍物
                hit_idx = obstacle_indices[0]
                hit_point = (xs_int[hit_idx], ys_int[hit_idx])
                
                # 顺便把这个障碍物点在输出图上标记出来
                vis_image[ys_int[hit_idx], xs_int[hit_idx]] = 1.0
            else:
                # 没击中障碍物，光线在地图边缘或 max_range 处停止
                hit_point = (xs_int[-1], ys_int[-1])
                
            hit_points.append(hit_point)
            
        # 5. 核心修复：多边形填充
        # 将击中点连成一个多边形，内部填充为 0 (Free)
        # 这解决了射线之间有空隙的问题
        poly_pts = np.array(hit_points, dtype=np.int32)
        # 0 代表 Free Space。注意 fillPoly 需要 list of arrays
        cv2.fillPoly(vis_image, [poly_pts], 0.0)
        
        # 6. 重新叠加障碍物 (可选，确保障碍物没有被 fillPoly 覆盖掉)
        # 这一步是为了锐化边界，因为 fillPoly 可能会把障碍物边缘像素吃掉
        # 但通常上面的 loop 里已经标了 obstacle，或者我们可以最后把原图的障碍物叠回来
        # 注意：只有在“可视区域内”的障碍物才应该被标出，所以比较复杂。
        # 这里建议：上面循环中标记的 obstacle 点是最准确的。
        # 如果为了防止 fillPoly 覆盖了刚才标记的障碍物点：
        # 我们再次遍历 hit_points 把它们设回 1 (如果是障碍物的话)
        
        # 实际上，更简单的做法是：
        # fillPoly 填了 0，但我们的逻辑是 0=Free, 0.5=Unknown, 1=Obstacle
        # 我们需要把刚才 fillPoly 覆盖掉的那个“击中点”恢复成 1
        
        # 提取多边形轮廓上的点（即击中点），检查它们在原图是否是障碍物
        # 如果原图是 1，则 vis_image 设为 1
        for pt in hit_points:
            if 0 <= pt[1] < h and 0 <= pt[0] < w:
                if image[pt[1], pt[0]] == 1:
                    vis_image[pt[1], pt[0]] = 1.0

        # 机器人所在位置肯定是 Free
        vis_image[int(y0), int(x0)] = 0.0
        
        return vis_image
        # 数据增强：随机遮挡
        def _add_random_polygons(self, image, nb_polygons, max_size):
            # 在图像上随机画黑色的多边形，模拟遮挡或传感器缺失，提高模型鲁棒性
            for i in range(nb_polygons):
                num_corners = random.randint(10, 20)
                polygon_points = np.random.randint(0, max_size, size=(num_corners, 2))
                x_offset = np.random.randint(0, image.shape[0])
                y_offset = np.random.randint(0, image.shape[1])
                polygon_points[:, 0] += x_offset
                polygon_points[:, 1] += y_offset

                # Create a convex hull from the points  # ... 生成随机顶点坐标 ...
                hull = cv2.convexHull(polygon_points)

                # Draw the hull on the image
                cv2.fillPoly(image, [hull], (0, 0, 0))
            return image

        """Load images"""

        # 加载深度图和语义/RGB图到内存
        def load_data_in_memory(self) -> None:
            #  读取深度图文件
            """Load data into RAM to speed up training"""
            for idx in tqdm(range(len(self.depth_filename)), desc="Load images into RAM"):
                # self.depth_imgs.append(self._load_depth_img(idx))
                # if self.semantics or self.rgb:
                #     self.sem_imgs.append(self._load_sem_rgb_img(idx))
                # ！！！关键：加载时不要加噪声，存纯净图！！！
                self.depth_imgs.append(self._load_depth_img(idx, augment=False)) 
            self.load_ram = True
            return

        # 加载单张深度图
        def _load_depth_img(self, idx, augment=True) -> torch.Tensor:
            #  读取深度图文件
            if self.depth_filename[idx].endswith(".png"):
                depth_image = Image.open(self.depth_filename[idx])
                if self._cfg.real_world_data:
                    # 真实世界数据可能需要旋转180度（取决于相机安装方式）
                    depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
                else:
                    depth_image = np.array(depth_image)
            else:
                # 支持 .npy 格式的深度图
                depth_image = np.load(self.depth_filename[idx])
            # 处理无效值和归一化（毫米转米）
            depth_image[~np.isfinite(depth_image)] = 0.0
            depth_image = (depth_image / 1000.0).astype("float32")
            depth_image[depth_image > self._cfg.max_depth] = 0.0  # 截断超过最大距离的值

            # add noise to depth image # 添加噪声 (椒盐噪声 或 高斯噪声)
            # if self._cfg.depth_salt_pepper or self._cfg.depth_gaussian:
            #     depth_norm = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
            #     if self._cfg.depth_salt_pepper:
            #         depth_norm = random_noise(
            #             depth_norm,
            #             mode="s&p",
            #             amount=self._cfg.depth_salt_pepper,
            #             clip=False,
            #         )
            #     if self._cfg.depth_gaussian:
            #         depth_norm = random_noise(
            #             depth_norm,
            #             mode="gaussian",
            #             mean=0,
            #             var=self._cfg.depth_gaussian,
            #             clip=False,
            #         )
            #     depth_image = depth_norm * (np.max(depth_image) - np.min(depth_image)) + np.min(depth_image)
            # # 添加随机遮挡
            # if self._cfg.depth_random_polygons_nb and self._cfg.depth_random_polygons_nb > 0:
            #     depth_image = self._add_random_polygons(
            #         depth_image,
            #         self._cfg.depth_random_polygons_nb,
            #         self._cfg.depth_random_polygon_size,
            #     )

            if augment:
                depth_image = self.apply_noise(depth_image)

            # transform depth image 转为 Tensor 并应用 PyTorch transform
            depth_image = self.transform(depth_image).type(torch.float32)
            # 如果标记为增强样本，进行水平翻转
            if self.pair_augment[idx]:
                depth_image = self.flip_transform.forward(depth_image)

            return depth_image

    # 我给出的结论是：在大多数深度学习与传感器融合的训练场景下，保持不一致（独立）通常效果更好，虽然直觉上觉得“由于物体遮挡，应该两个都看不见”，但从训练鲁棒性的角度来看，独立遮挡更有价值。
    # 以下是详细的分析：
    # 1. 这种遮挡模拟的是什么？
    # 我们需要区分两种“遮挡”：
    # 类型 A：环境中的物理遮挡（Physical Occlusion）
    # 例如：一根柱子挡在前面，或者一个人走过。
    # 表现：在仿真器渲染原始图像时，柱子既会挡住深度相机的视线，也会挡住RGB相机的视线。
    # 现状：这部分已经由你的仿真器（Omniverse/Gazebo）自然完成了。不需要通过代码里的 _add_random_polygons 来模拟。
    # 类型 B：传感器本身的数据丢失或污损（Sensor Artifacts / Dropout）
    # 例如：
    # 深度相机：强光照射导致红外过曝、黑色吸光物体导致没有回波、镜头上有油污。
    # RGB/语义相机：镜头上有泥点、图像传输出现丢包导致部分画面花屏、语义分割网络对某块区域分类失败（输出全黑或乱码）。
    # 现状：这正是 _add_random_polygons 想要模拟的情况。
        def _load_sem_rgb_img(self, idx) -> torch.Tensor:
            image = Image.open(self.sem_rgb_filename[idx])
            if self._cfg.real_world_data:
                image = np.array(image.transpose(PIL.Image.ROTATE_180))
            else:
                image = np.array(image)
            # normalize image
            if self.pixel_mean is not None and self.pixel_std is not None:
                image = (image - self.pixel_mean) / self.pixel_std

            # add noise to semantic image
            if self._cfg.sem_rgb_black_img:
                if random.randint(0, 99) < self._cfg.sem_rgb_black_img * 100:
                    image = np.zeros_like(image)
            if self._cfg.sem_rgb_pepper:
                image = random_noise(
                    image,
                    mode="pepper",
                    amount=self._cfg.depth_salt_pepper,
                    clip=False,
                )
            if self._cfg.sem_rgb_random_polygons_nb and self._cfg.sem_rgb_random_polygons_nb > 0:
                image = self._add_random_polygons(
                    image,
                    self._cfg.sem_rgb_random_polygons_nb,
                    self._cfg.sem_rgb_random_polygon_size,
                )

            # transform semantic image
            image = self.transform(image).type(torch.float32)
            assert image.round(decimals=1).max() <= 1.0, (
                f"Image '{self.sem_rgb_filename[idx]}' is not normalized with max" f" value {image.max().item()}"
            )

            if self.pair_augment[idx]:
                image = self.flip_transform.forward(image)

            return image

        """Get image in training"""

        def __len__(self):
            return len(self.depth_filename)

    # 获取样本
        def __getitem__(self, idx):
            """
            Get batch items

            Returns:
                - depth_image: depth image
                - sem_rgb_image: semantic image
                - odom: odometry of the start pose (point and rotation)
                - goal: goal point in the camera frame
                - pair_augment: bool if the pair is augmented (flipped at the y-axis of the image)
            """

            if self.load_ram:
                depth_image = self.depth_imgs[idx].copy()  # 注意要copy，防止修改内存中的原图
            else:
                # 如果从硬盘读，也不要在这里加噪声，读纯净的
                depth_image = self._load_depth_img(idx, augment=False)
            
            depth_image = self.apply_noise(depth_image)
            # get depth image
            if self.load_ram:
                depth_image = self.depth_imgs[idx]
                if self.semantics or self.rgb:
                    sem_rgb_image = self.sem_imgs[idx]
                else:
                    sem_rgb_image = 0
            else:
                depth_image = self._load_depth_img(idx)
                if self.semantics or self.rgb:
                    sem_rgb_image = self._load_sem_rgb_img(idx)
                else:
                    sem_rgb_image = 0

            return (
                depth_image,
                sem_rgb_image,
                self.odom[idx],     # 当前位姿
                self.goal[idx],     # 目标点
                self.pair_augment[idx],     # 是否进行水平翻转增强
            )


    def augment_map_noise(grid_map, 
                        roughness_prob=0.8, 
                        morph_prob=0.8,
                        roughness_intensity=0.1):
        """
        模拟建图噪声：边缘粗糙化 + 随机膨胀/腐蚀
        
        Args:
            grid_map: 2D numpy array (H, W), 值为 0 (Free) 或 1 (Obstacle)。
                    如果是 0-255 的图，请先归一化或修改内部阈值。
            roughness_prob: 边缘粗糙化发生的概率 (0.0 - 1.0)
            morph_prob: 膨胀/腐蚀发生的概率 (0.0 - 1.0)
            roughness_intensity: 粗糙程度，值越大边缘越锯齿 (建议 0.05 - 0.2)
            
        Returns:
            noisy_map: 处理后的 2D numpy array, 0 或 1
        """
        
        # 确保输入是 float 类型以便计算，且是二值的
        # 如果输入含有 0.5 (Unknown)，建议先分离出障碍物层单独处理
        noisy_map = grid_map.astype(np.float32).copy()
        h, w = noisy_map.shape

        # ==========================================
        # 1. 边缘粗糙化 (Edge Roughness / Jittering)
        # 原理：先模糊边缘，再叠加高斯噪声，最后重新二值化
        # ==========================================
        if random.random() < roughness_prob:
            # A. 高斯模糊：让锐利的 0/1 边缘变成 0.1, 0.5, 0.9 的渐变带
            # kernel size 决定了受影响边缘的宽度
            blur_ksize = random.choice([3, 5]) 
            blurred = cv2.GaussianBlur(noisy_map, (blur_ksize, blur_ksize), 0)
            
            # B. 生成噪声：只在边缘附近产生影响
            # 噪声让 0.5 的地方可能变成 0.4 或 0.6
            noise = np.random.randn(h, w) * roughness_intensity
            
            # C. 叠加噪声
            potential_map = blurred + noise
            
            # D. 重新阈值化：切断噪声，形成参差不齐的边缘
            # 0.5 是中间阈值
            noisy_map = (potential_map > 0.5).astype(np.float32)

        # ==========================================
        # 2. 随机膨胀/腐蚀 (Random Dilation/Erosion)
        # 原理：模拟光斑扩散(变胖)或反射率不足(变瘦) 这个是不是可以不要
        # ==========================================
        if random.random() < morph_prob:
            # 转换回 uint8 供 opencv 形态学操作使用
            morph_map = (noisy_map * 255).astype(np.uint8)
            
            # 随机选择核大小 (决定变胖变瘦的幅度)
            # 3x3 是轻微变化，5x5 是明显变化
            kernel_size = random.choice([3, 5])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 随机决策：变胖、变瘦、还是(小概率)开闭运算
            op_type = random.random()
            
            if op_type < 0.4:
                # 膨胀 (Dilate) -> 障碍物变胖 (模拟运动模糊/大光斑)
                morph_map = cv2.dilate(morph_map, kernel, iterations=1)
                
            elif op_type < 0.8:
                # 腐蚀 (Erode) -> 障碍物变瘦 (模拟点云稀疏/被剔除)
                morph_map = cv2.erode(morph_map, kernel, iterations=1)
                
            else:
                # 开运算 (Opening) -> 去除孤立噪点，断开细小连接
                morph_map = cv2.morphologyEx(morph_map, cv2.MORPH_OPEN, kernel)

            # 转回 0/1 float
            noisy_map = (morph_map > 127).astype(np.float32)

        return noisy_map


    def augment_probabilistic_blur(grid_map, blur_prob=0.5, max_sigma=2.0):
        """
        模拟概率栅格地图的模糊特性。
        
        Args:
            grid_map: (H, W) float32 array.
            blur_prob: 执行模糊的概率.
            max_sigma: 高斯模糊的最大标准差，值越大越模糊.
            
        Returns:
            blurred_map: 模糊后的地图.
        """
        if random.random() > blur_prob:
            return grid_map

        blurred_map = grid_map.copy()
        
        # 随机选择模糊核大小 (必须是奇数)
        k_size = random.choice([3, 5, 7])
        
        # 随机选择 sigma (模糊强度)
        sigma = random.uniform(0.5, max_sigma)
        
        # 应用高斯模糊
        # 这会让 0.0 和 1.0 的边界变成 0.3, 0.6 等中间值
        blurred_map = cv2.GaussianBlur(blurred_map, (k_size, k_size), sigma)
        
        # 保持数值在 [0, 1] 范围内
        blurred_map = np.clip(blurred_map, 0.0, 1.0)
        
        return blurred_map
    
    def augment_dynamic_ghosting(grid_map, ghost_prob=0.5, num_ghosts=5, max_ghost_size=5):
        """
        模拟动态物体留下的残影 (Ghosting)。
        在 Free 区域随机撒一些小的障碍物斑点。
        
        Args:
            grid_map: (H, W) float32 array.
            ghost_prob: 执行操作的概率.
            num_ghosts: 随机生成多少个残影斑点.
            max_ghost_size: 残影斑点的最大半径 (像素).
            
        Returns:
            ghost_map: 叠加了残影的地图.
        """
        if random.random() > ghost_prob:
            return grid_map
            
        h, w = grid_map.shape
        ghost_map = grid_map.copy()
        
        for _ in range(num_ghosts):
            # 1. 随机选一个中心点
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            
            # 为了真实，尽量只在 Free 区域 (0.0) 或 Unknown (0.5) 区域添加，
            # 如果本来就是墙 (1.0)，加了也没意义
            if grid_map[cy, cx] > 0.8: 
                continue
                
            # 2. 随机生成斑点的大小和形状
            # 我们用随机画圆或椭圆来模拟一团噪声
            axis_x = random.randint(1, max_ghost_size)
            axis_y = random.randint(1, max_ghost_size)
            angle = random.randint(0, 180)
            
            # 3. 在地图上画出这个残影 (设为 1.0 或一个较高的概率值如 0.8)
            # 注意：这里直接修改像素值
            cv2.ellipse(ghost_map, (cx, cy), (axis_x, axis_y), angle, 0, 360, 1.0, -1)
            
        return ghost_map
    
    def augment_sensor_dropout_fov(grid_map, robot_pos=None, max_range_px=None, dropout_prob=0.3):
        """
        模拟 Lidar 最大射程限制和随机扇区丢包。
        将看不见的地方设为 Unknown (0.5)。
        
        Args:
            grid_map: (H, W) float32 array.
            robot_pos: (x, y) 机器人在地图上的像素坐标. 如果为 None，默认在地图中心.
            max_range_px: 最大射程 (像素). 如果为 None，则随机生成一个.
            dropout_prob: 发生随机丢包的概率.
            
        Returns:
            degraded_map: 视场受限后的地图.
        """
        h, w = grid_map.shape
        if robot_pos is None:
            robot_pos = (w // 2, h // 2)
        
        rx, ry = robot_pos
        
        # 创建一个掩码 (Mask): 1 代表可见，0 代表不可见 (Unknown)
        # 初始全为 0 (全黑/全未知)
        visibility_mask = np.zeros((h, w), dtype=np.uint8)
        
        # ==========================
        # 1. 模拟最大射程 (FOV Limit)
        # ==========================
        if max_range_px is None:
            # 随机设定一个射程，模拟不同性能的雷达，或者环境导致的衰减
            # 假设图大概 200x200，射程随机在 60 到 150 之间
            max_range_px = random.randint(min(h, w)//4, min(h, w)//1)
            
        # 在掩码上画一个白色的实心圆，圆内代表探测范围内
        cv2.circle(visibility_mask, (int(rx), int(ry)), int(max_range_px), 1, -1)
        
        # ==========================
        # 2. 模拟随机丢包 (Dropout) - 扇形缺失
        # ==========================
        if random.random() < dropout_prob:
            # 随机决定丢几个扇区 (Lidar 的某几束光或某个角度的数据包丢了)
            num_drops = random.randint(1, 3)
            
            for _ in range(num_drops):
                # 随机起始角度和结束角度
                start_angle = random.randint(0, 360)
                end_angle = start_angle + random.randint(10, 60) # 丢掉 10~60 度的范围
                
                # 在掩码上把这个扇区画黑 (0)，表示变为 Unknown
                cv2.ellipse(visibility_mask, (int(rx), int(ry)), 
                            (h, w), # 轴长设得很大以覆盖全图
                            0, start_angle, end_angle, 0, -1)

        # ==========================
        # 3. 应用掩码
        # ==========================
        degraded_map = grid_map.copy()
        
        # 逻辑：
        # 如果 mask 是 1 (可见)，保留原地图的值
        # 如果 mask 是 0 (不可见)，强制设为 0.5 (Unknown)
        degraded_map[visibility_mask == 0] = 0.5
        
        return degraded_map



# 按距离对样本进行分类存储。为了保证数据平衡，代码会将“起点-终点”对按距离分组（例如 1m, 3m, 5m 组），并分别管理
class DistanceSchemeIdx:
    def __init__(self, distance: float) -> None:
        self.distance: float = distance

        self.odom_list: List[pp.LieTensor] = []
        self.goal_list: List[pp.LieTensor] = []
        self.pair_within_fov: List[bool] = []
        self.pair_front_of_robot: List[bool] = []
        self.pair_behind_robot: List[bool] = []
        self.sem_rgb_img_list: List[str] = []

        # flags
        self.has_data: bool = False
        return

    def update_buffers(
        self,
        odom: pp.LieTensor,
        goal: pp.LieTensor,
        within_fov: bool = False,
        front_of_robot: bool = False,
        behind_robot: bool = False,
        depth_filename: str = None,
        sem_rgb_filename: str = None,
    ) -> None:
        self.odom_list.append(odom)
        self.goal_list.append(goal)
        self.pair_within_fov.append(within_fov)
        self.pair_front_of_robot.append(front_of_robot)
        self.pair_behind_robot.append(behind_robot)
        self.depth_img_list.append(depth_filename)
        self.sem_rgb_img_list.append(sem_rgb_filename)

        self.has_data = len(self.odom_list) > 0
        return

    # 根据指定的数量要求，从当前距离分类的数据池中采样数据，并在数据不足时通过数据增强（镜像翻转）来补全数据，以保证数据集的平衡
    def get_data(
        self,
        nb_fov: int,  # 视野内（Goal within FOV）的样本数量
        nb_front: int,  # 机器人前方的样本数量
        nb_back: int,  # 机器人后方的样本数量
        augment: bool = True,  # 是否允许数据增强（镜像翻转）
    ) -> Tuple[List[pp.LieTensor], List[pp.LieTensor], List[str], List[str], np.ndarray,]:
        assert self.has_data, f"DistanceSchemeIdx for distance {self.distance} has no data"

        # get all pairs that are within the fov 获取索引
        idx_fov = np.where(self.pair_within_fov)[0]
        idx_front = np.where(self.pair_front_of_robot)[0]
        idx_back = np.where(self.pair_behind_robot)[0]
        idx_augment = []

        # augment pairs if not enough
        if len(idx_fov) == 0:  # 情况 A：没有数据
            print(f"[WARNING] for distance {self.distance} no 'within_fov'" " samples")
            idx_fov = np.array([], dtype=np.int64)
        elif len(idx_fov) < nb_fov:  # 数据不足（需要补全）
            print(
                f"[INFO] for distance {self.distance} not enough 'within_fov'"
                f" samples ({len(idx_fov)} instead of {nb_fov})"
            )
            if augment:
                # 计算缺多少个：nb_fov - len(idx_fov)
                # 从现有数据中随机抽样来填补空缺，放入 idx_augment 列表
                idx_augment.append(
                    np.random.choice(
                        idx_fov,
                        min(len(idx_fov), nb_fov - len(idx_fov)),
                        # 如果缺的数量比现有的还多，就需要重复采样 (replace=True)
                        replace=(nb_fov - len(idx_fov) > len(idx_fov)),
                    )
                )
            else:
                #  不增强，有多少拿多少
                idx_fov = np.random.choice(idx_fov, len(idx_fov), replace=False)
        else:
            # 随机抽取指定数量 (nb_fov)，不重复采样
            idx_fov = np.random.choice(idx_fov, nb_fov, replace=False)

        if len(idx_front) == 0:
            print(f"[WARNING] for distance {self.distance} no 'front_of_robot'" " samples")
            idx_front = np.array([], dtype=np.int64)
        elif len(idx_front) < nb_front:
            print(
                f"[INFO] for distance {self.distance} not enough"
                f" 'front_of_robot' samples ({len(idx_front)} instead of"
                f" {nb_front})"
            )
            if augment:
                idx_augment.append(
                    np.random.choice(
                        idx_front,
                        min(len(idx_front), nb_front - len(idx_front)),
                        replace=(nb_front - len(idx_front) > len(idx_front)),
                    )
                )
            else:
                idx_front = np.random.choice(idx_front, len(idx_front), replace=False)
        else:
            idx_front = np.random.choice(idx_front, nb_front, replace=False)

        if len(idx_back) == 0:
            print(f"[WARNING] for distance {self.distance} no 'behind_robot'" " samples")
            idx_back = np.array([], dtype=np.int64)
        elif len(idx_back) < nb_back:
            print(
                f"[INFO] for distance {self.distance} not enough"
                f" 'behind_robot' samples ({len(idx_back)} instead of"
                f" {nb_back})"
            )
            if augment:
                idx_augment.append(
                    np.random.choice(
                        idx_back,
                        min(len(idx_back), nb_back - len(idx_back)),
                        replace=(nb_back - len(idx_back) > len(idx_back)),
                    )
                )
            else:
                idx_back = np.random.choice(idx_back, len(idx_back), replace=False)
        else:
            idx_back = np.random.choice(idx_back, nb_back, replace=False)

        # 合并原本就有的真实数据的索引
        idx = np.hstack([idx_fov, idx_front, idx_back])

        # stack buffers
        odom = torch.stack(self.odom_list)
        goal = torch.stack(self.goal_list)

        # get pairs
        if idx_augment:
            #  如果有需要增强的数据
            idx_augment = np.hstack(idx_augment)
            # 1. 堆叠 Odom：包括原始选中的(idx) 和 需要被增强的原始样本(idx_augment)
            odom = torch.vstack([odom[idx], odom[idx_augment]])
            # 2. 堆叠 Goal 并进行镜像翻转 (核心!)
            # goal[idx_augment] 取出那些需要被翻转的样本的目标点
            # * torch.tensor([[1, -1, 1, 1, 1, 1, 1]]) 是在做坐标变换
            goal = torch.vstack(
                [
                    goal[idx],
                    goal[idx_augment].tensor() * torch.tensor([[1, -1, 1, 1, 1, 1, 1]]),
                ]
            )
            depth_img_list = [self.depth_img_list[j] for j in idx.tolist()] + [
                self.depth_img_list[i] for i in idx_augment.tolist()
            ]
            sem_rgb_img_list = [self.sem_rgb_img_list[j] for j in idx.tolist()] + [
                self.sem_rgb_img_list[i] for i in idx_augment.tolist()
            ]
            # 3. 创建标记数组
            # 0 代表原始数据，1 代表这是增强数据（后续加载图片时会根据这个标记进行图片翻转）
            augment = np.hstack([np.zeros(len(idx)), np.ones(len(idx_augment))])
            return odom, goal, depth_img_list, sem_rgb_img_list, augment
        else:
            return (
                odom[idx],
                goal[idx],
                [self.depth_img_list[j] for j in idx.tolist()],
                [self.sem_rgb_img_list[j] for j in idx.tolist()],
                np.zeros(len(idx)),
            )


# 数据生成器
class PlannerDataGenerator(Dataset):
    debug = False
    mesh_size = 0.5

    def __init__(
        self,
        cfg: DataCfg,
        root: str,
        semantics: bool = False,
        rgb: bool = False,
        cost_map: CostMapPCD = None,
    ) -> None:
        print(
            f"[INFO] PlannerDataGenerator init with semantics={semantics},"
            f" rgb={rgb} for ENV {os.path.split(root)[-1]}"
        )
        # super().__init__()
        # set parameters
        self._cfg = cfg
        self.root = root
        self.cost_map = cost_map
        self.semantics = semantics
        self.rgb = rgb
        assert not (self.semantics and self.rgb), "semantics and rgb cannot be true at the same time"

        # init list for final odom, goal and img mapping
        self.depth_filename_list = []
        self.sem_rgb_filename_list = []
        self.odom_depth: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_outside: np.ndarray = None
        self.pair_difficult: np.ndarray = None
        self.pair_augment: np.ndarray = None
        self.pair_within_fov: np.ndarray = None
        self.pair_front_of_robot: np.ndarray = None
        self.odom_array_sem_rgb: pp.LieTensor = None
        self.odom_array_depth: pp.LieTensor = None

        self.odom_used: int = 0
        self.odom_no_suitable_goals: int = 0

        # set parameters
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # get odom data and filter
        # # 1. 加载里程计数据
        self.load_odom()
        # 2. 过滤掉离障碍物太近的轨迹点 (filter_obs_inflation)
        self.filter_obs_inflation()

        # noise edges in depth image --> real world Realsense difficulties along edges
        # 3. (可选) 给深度图边缘添加噪声，模拟 RealSense 相机缺陷
        if self._cfg.noise_edges:
            self.noise_edges()

        # find odom-goal pairs # 4. 核心步骤：生成起点-目标点对
        self.get_odom_goal_pairs()
        return

    """LOAD HELPER FUNCTIONS"""

    # 读取 camera_extrinsic.txt 文件，将其转换为 pypose.SE3 对象（包含位置和旋转）。
    def load_odom(self) -> None:
        print("[INFO] Loading odom data...", end=" ")
        # load odom of every image
        odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.depth_suffix}.txt")
        odom_np = np.loadtxt(odom_path, delimiter=",")
        self.odom_array_depth = pp.SE3(odom_np)

        if self.semantics or self.rgb:
            odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.sem_suffix}.txt")
            odom_np = np.loadtxt(odom_path, delimiter=",")
            self.odom_array_sem_rgb = pp.SE3(odom_np)

        if self.debug:
            # plot odom
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points
            small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
            odom_vis_list = []

            for i in range(len(self.odom_array_depth)):
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            self.odom_array_depth[i, 0],
                            self.odom_array_depth[i, 1],
                            self.odom_array_depth[i, 2],
                        )
                    )
                )
            odom_vis_list.append(self.cost_map.pcd_tsdf)

            o3d.visualization.draw_geometries(odom_vis_list)
        print("DONE!")
        return

    def load_images(self, root_path, domain: str = "depth"):
        img_path = os.path.join(root_path, domain)
        assert os.path.isdir(img_path), f"Image directory path '{img_path}' does not exist for domain" f" {domain}"
        assert len(os.listdir(img_path)) > 0, f"Image directory '{img_path}' is empty for domain {domain}"

        # use the more precise npy files if available
        img_filename_list = [str(s) for s in Path(img_path).rglob("*.npy")]
        if len(img_filename_list) == 0:
            img_filename_list = [str(s) for s in Path(img_path).rglob("*.png")]

        if domain == "depth":
            img_filename_list.sort(key=lambda x: int(x.split("/")[-1][: -(4 + len(self._cfg.depth_suffix))]))
        else:
            img_filename_list.sort(key=lambda x: int(x.split("/")[-1][: -(4 + len(self._cfg.sem_suffix))]))
        return img_filename_list

    """FILTER HELPER FUNCTIONS"""
    # 碰撞检测过滤
    # 作用是剔除掉那些离障碍物太近的训练数据点。
    # 在机器人路径规划的训练中，如果机器人的位置紧贴着墙壁或者已经撞上了障碍物，这些数据通常被认为是“坏数据”或“危险数据”。如果拿这些数据去训练网络，网络可能会误以为“贴着墙走”甚至“撞墙”是合理的行为。
    def filter_obs_inflation(self) -> None:
        """
        Filter odom points within the inflation range of the obstacles in the cost map.

        Filtering only performed according to the position of the depth camera, due to the close position of depth and semantic camera.
        """
        print(
            ("[INFO] Filter odom points within the inflation range of the" " obstacles in the cost map..."),
            end="",
        )
        # 将机器人的位置投影到 Cost Map (代价地图) 上
        norm_inds, _ = self.cost_map.Pos2Ind(self.odom_array_depth[:, None, :3])
        cost_grid = self.cost_map.cost_array.T.expand(self.odom_array_depth.shape[0], 1, -1, -1)
        norm_inds = norm_inds.to(cost_grid.device)
        # 使用 grid_sample 采样地图中的代价值
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        oloss_M = oloss_M.to(torch.float32).to("cpu")
        if self.semantics or self.rgb:
            points_free_space = oloss_M < self._cfg.obs_cost_height + abs(
                self.cost_map.cfg.sem_cost_map.negative_reward
            )
        else:
            # 判断每个点是否在“自由空间” (free space)
            # 如果代价值 > 阈值，说明离障碍物太近，标记为不安全
            points_free_space = oloss_M < self._cfg.obs_cost_height

        if self._cfg.carla:
            # for CARLA filter large open spaces
            # Extract the x and y coordinates from the odom poses
            x_coords = self.odom_array_depth.tensor()[:, 0]
            y_coords = self.odom_array_depth.tensor()[:, 1]

            # Filter the point cloud based on the square coordinates
            mask_area_1 = (y_coords >= 100.5) & (y_coords <= 325.5) & (x_coords >= 208.9) & (x_coords <= 317.8)
            mask_area_2 = (y_coords >= 12.7) & (y_coords <= 80.6) & (x_coords >= 190.3) & (x_coords <= 315.8)
            mask_area_3 = (y_coords >= 10.0) & (y_coords <= 80.0) & (x_coords >= 123.56) & (x_coords <= 139.37)

            combined_mask = mask_area_1 | mask_area_2 | mask_area_3 | ~points_free_space.squeeze(1)
            points_free_space = (~combined_mask).unsqueeze(1)

        if self.debug:
            # plot odom
            odom_vis_list = []
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points

            for i in range(len(self.odom_array_depth)):
                if round(oloss_M[i].item(), 3) == 0.0:
                    small_sphere.paint_uniform_color([0.4, 0.1, 1.0])  # violette
                elif points_free_space[i]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green
                else:
                    small_sphere.paint_uniform_color([1.0, 0.4, 0.1])  # red
                if self.semantics or self.rgb:
                    z_height = self.odom_array_depth.tensor()[i, 2] + abs(
                        self.cost_map.cfg.sem_cost_map.negative_reward
                    )
                else:
                    z_height = self.odom_array_depth.tensor()[i, 2]

                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            self.odom_array_depth.tensor()[i, 0],
                            self.odom_array_depth.tensor()[i, 1],
                            z_height,
                        )
                    )
                )

            odom_vis_list.append(self.cost_map.pcd_tsdf)
            o3d.visualization.draw_geometries(odom_vis_list)

        nb_odom_point_prev = len(self.odom_array_depth)
        self.odom_array_depth = self.odom_array_depth[points_free_space.squeeze()]
        self.nb_odom_points = self.odom_array_depth.shape[0]

        # load depth image files as name list
        depth_filename_list = self.load_images(self.root, "depth")
        self.depth_filename_list = [
            depth_filename_list[i] for i in range(len(depth_filename_list)) if points_free_space[i]
        ]

        if self.semantics:
            self.odom_array_sem_rgb = self.odom_array_sem_rgb[points_free_space.squeeze()]
            sem_rgb_filename_list = self.load_images(self.root, "semantics")
            self.sem_rgb_filename_list = [
                sem_rgb_filename_list[i] for i in range(len(sem_rgb_filename_list)) if points_free_space[i]
            ]
        elif self.rgb:
            self.odom_array_sem_rgb = self.odom_array_sem_rgb[points_free_space.squeeze()]
            sem_rgb_filename_list = self.load_images(self.root, "rgb")
            self.sem_rgb_filename_list = [
                sem_rgb_filename_list[i] for i in range(len(sem_rgb_filename_list)) if points_free_space[i]
            ]

        assert len(self.depth_filename_list) != 0, "No depth images left after filtering"
        print("DONE!")
        print(
            "[INFO] odom points outside obs inflation :"
            f" \t{self.nb_odom_points} ({round(self.nb_odom_points/nb_odom_point_prev*100, 2)} %)"
        )

        return

    """GENERATE SAMPLES"""

    def get_odom_goal_pairs(self) -> None:
        # get fov
        self.get_intrinscs_and_fov()
        # construct graph
        self.get_graph()
        # get pairs
        self.get_pairs()

        # free up memory
        self.odom_array_depth = self.odom_array_sem_rgb = None
        return

    def compute_ratios(self) -> Tuple[float, float, float]:
        # ratio of general samples distribution
        num_within_fov = self.odom_depth[self.pair_within_fov].shape[0]
        ratio_fov = num_within_fov / self.odom_depth.shape[0]
        ratio_front = np.sum(self.pair_front_of_robot) / self.odom_depth.shape[0]
        ratio_back = 1 - ratio_front - ratio_fov

        # samples ratios within fov samples
        num_easy = (
            num_within_fov
            - self.pair_difficult[self.pair_within_fov].sum().item()
            - self.pair_outside[self.pair_within_fov].sum().item()
        )
        ratio_easy = num_easy / num_within_fov
        ratio_hard = self.pair_difficult[self.pair_within_fov].sum().item() / num_within_fov
        ratio_outside = self.pair_outside[self.pair_within_fov].sum().item() / num_within_fov
        return (
            ratio_fov,
            ratio_front,
            ratio_back,
            ratio_easy,
            ratio_hard,
            ratio_outside,
        )

    def get_intrinscs_and_fov(self) -> None:
        # load intrinsics
        intrinsic_path = os.path.join(self.root, "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        self.K_depth = P[0].reshape(3, 4)[:3, :3]
        self.K_sem_rgb = P[1].reshape(3, 4)[:3, :3]

        self.alpha_fov = 2 * math.atan(self.K_depth[0, 0] / self.K_depth[0, 2])
        return

    # 构建可达性图
    def get_graph(self) -> None:
        num_connections = 3
        num_intermediate = 3

        # get occpuancy map from tsdf map
        cost_array = self.cost_map.tsdf_array.cpu().numpy()
        if self.semantics or self.rgb:
            occupancy_map = (
                cost_array > self._cfg.obs_cost_height + abs(self.cost_map.cfg.sem_cost_map.negative_reward)
            ).astype(np.uint8)
        else:
            occupancy_map = (cost_array > self._cfg.obs_cost_height).astype(np.uint8)
        # construct kdtree to find nearest neighbors of points
        odom_points = self.odom_array_depth.data[:, :2].data.cpu().numpy()
        kdtree = KDTree(odom_points)
        _, nearest_neighbors_idx = kdtree.query(odom_points, k=num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = nearest_neighbors_idx[:, 1:]

        # define origin and neighbor points
        origin_point = np.repeat(odom_points, repeats=num_connections, axis=0)
        neighbor_points = odom_points[nearest_neighbors_idx, :].reshape(-1, 2)
        # interpolate points between origin and neighbor points
        x_interp = (
            origin_point[:, None, 0]
            + (neighbor_points[:, 0] - origin_point[:, 0])[:, None]
            * np.linspace(0, 1, num=num_intermediate + 1, endpoint=False)[1:]
        )
        y_interp = (
            origin_point[:, None, 1]
            + (neighbor_points[:, 1] - origin_point[:, 1])[:, None]
            * np.linspace(0, 1, num=num_intermediate + 1, endpoint=False)[1:]
        )
        inter_points = np.stack((x_interp.reshape(-1), y_interp.reshape(-1)), axis=1)
        # get the indices of the interpolated points in the occupancy map
        occupancy_idx = (
            inter_points - np.array([self.cost_map.cfg.x_start, self.cost_map.cfg.y_start])
        ) / self.cost_map.cfg.general.resolution

        # check occupancy for collisions at the interpolated points
        collision = occupancy_map[
            occupancy_idx[:, 0].astype(np.int64),
            occupancy_idx[:, 1].astype(np.int64),
        ]
        collision = np.any(collision.reshape(-1, num_intermediate), axis=1)

        # get edge indices
        idx_edge_start = np.repeat(np.arange(odom_points.shape[0]), repeats=num_connections, axis=0)
        idx_edge_end = nearest_neighbors_idx.reshape(-1)

        # filter collision edges idx_edge_start 和 idx_edge_end 是一一对应的：
        idx_edge_end = idx_edge_end[~collision]
        idx_edge_start = idx_edge_start[~collision]

        # init graph
        self.graph = nx.Graph()
        # add nodes with position attributes
        self.graph.add_nodes_from(list(range(odom_points.shape[0])))
        pos_attr = {i: {"pos": odom_points[i]} for i in range(odom_points.shape[0])}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
        distance_attr = {
            (i, j): {"distance": np.linalg.norm(odom_points[i] - odom_points[j])}
            for i, j in zip(idx_edge_start, idx_edge_end)
        }
        nx.set_edge_attributes(self.graph, distance_attr)

        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt

            nx.draw_networkx(
                self.graph,
                nx.get_node_attributes(self.graph, "pos"),
                node_size=10,
                with_labels=False,
                node_color=[0.0, 1.0, 0.0],
            )
            plt.show()
        return

    # 筛选配对 遍历所有机器人走过的位置（作为起点），寻找其他所有位置（作为潜在终点），计算它们之间的真实路径距离，并根据距离和相对方位（视野内/前/后）将这些“起点-终点”对分类存储起来。
    def get_pairs(self):
        # iterate over all odom points and find goal points
        self.odom_no_suitable_goals = 0
        self.odom_used = 0

        # init semantic warp parameters
        # 目的：如果使用了语义图或 RGB 图，通常因为相机安装位置不同（例如深度相机和 RGB 相机有几厘米的偏移），需要进行重投影（Reprojection）或Warp操作，把语义图的像素对齐到深度图的像素上。
        if self.semantics or self.rgb:
            # compute pixel tensor
            depth_filename = self.depth_filename_list[0]
            depth_img = self._load_depth_image(depth_filename)
            x_nums, y_nums = depth_img.shape
            self.pix_depth_cam_frame = self.compute_pixel_tensor(x_nums, y_nums, self.K_depth)
            # make dir
            os.makedirs(os.path.join(self.root, "img_warp"), exist_ok=True)

        # get distances between odom and goal points  计算所有点对之间的“真实路径距离” (核心计算)
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(
                self.graph,
                cutoff=self._cfg.max_goal_distance,
                weight="distance",
            )
        )

        # init dataclass for each entry in the distance scheme
        self.category_scheme_pairs: Dict[float, DistanceSchemeIdx] = {
            distance: DistanceSchemeIdx(distance=distance) for distance in self._cfg.distance_scheme.keys()
        }

        # iterate over all odom points 遍历每一个位置作为“起点”
        for odom_idx in tqdm(range(self.nb_odom_points), desc="Start-End Pairs Generation"):
            odom = self.odom_array_depth[odom_idx]

            # transform all odom points to current odom frame
            # self.odom_array_depth 是所有点在世界坐标系下的位置。
            # pp.Inv(odom) 是当前起点的逆变换。
            # @ 是 PyPose 库的矩阵乘法。
            # 结果：goals 变成了所有点相对于当前机器人视角的坐标。
            goals = pp.Inv(odom) @ self.odom_array_depth
            # categorize goals
            # 根据相对坐标，判断每个潜在终点是在：
            # 视野内 (within_fov)：机器人能直接看到的区域。
            # 前方但视野外 (front_of_robot)：在前面，但可能被遮挡或角度太偏。
            # 后方 (behind_robot)：在机器人屁股后面。
            (
                within_fov,
                front_of_robot,
                behind_robot,
            ) = self.get_goal_categories(
                goals
            )  # returns goals in odom frame

            # filter odom if no suitable goals within the fov are found
            if within_fov.sum() == 0:
                self.odom_no_suitable_goals += 1
                continue
            self.odom_used += 1
            # 处理图像 (Warping)
            if self.semantics or self.rgb:
                # semantic warp
                img_new_path = self._get_overlay_img(odom_idx)
            else:
                img_new_path = None

            # get pair according to distance scheme for each category
            # 拿到所有候选终点的真实距离（从 odom_goal_distances 查表）。
            # 根据距离（比如是否在 0.5m ~ 1.5m 之间）把终点分配给对应的 DistanceSchemeIdx 容器。
            # 随机采样：如果符合条件的点太多，只随机选几个（例如选 3 个），避免数据爆炸。
            self.reduce_pairs(
                odom_idx,
                goals,
                within_fov,
                odom_goal_distances[odom_idx],
                img_new_path,
                within_fov=True,
            )
            self.reduce_pairs(
                odom_idx,
                goals,
                behind_robot,
                odom_goal_distances[odom_idx],
                img_new_path,
                behind_robot=True,
            )
            self.reduce_pairs(
                odom_idx,
                goals,
                front_of_robot,
                odom_goal_distances[odom_idx],
                img_new_path,
                front_of_robot=True,
            )

            # DEBUG
            if self.debug:
                # plot odom
                small_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    self.mesh_size / 3.0
                )  # successful trajectory points
                odom_vis_list = []
                goal_odom = odom @ goals
                hit_pcd = (goal_odom).cpu().numpy()[:, :3]
                for idx, pts in enumerate(hit_pcd):
                    if within_fov[idx]:
                        small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
                    elif front_of_robot[idx]:
                        small_sphere.paint_uniform_color([0.0, 0.5, 0.5])
                    else:
                        small_sphere.paint_uniform_color([0.0, 0.1, 1.0])
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((pts[0], pts[1], pts[2])))

                # viz cost map
                odom_vis_list.append(self.cost_map.pcd_tsdf)

                # field of view visualization
                fov_vis_length = 0.75  # length of the fov visualization plane in meters
                fov_vis_pt_right = odom @ pp.SE3(
                    [
                        fov_vis_length * np.cos(self.alpha_fov / 2),
                        fov_vis_length * np.sin(self.alpha_fov / 2),
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
                fov_vis_pt_left = odom @ pp.SE3(
                    [
                        fov_vis_length * np.cos(self.alpha_fov / 2),
                        -fov_vis_length * np.sin(self.alpha_fov / 2),
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
                fov_vis_pt_right = fov_vis_pt_right.numpy()[:3]
                fov_vis_pt_left = fov_vis_pt_left.numpy()[:3]
                fov_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(
                        np.array(
                            [
                                odom.data.cpu().numpy()[:3],
                                fov_vis_pt_right,
                                fov_vis_pt_left,
                            ]
                        )
                    ),
                    triangles=o3d.utility.Vector3iVector(np.array([[2, 1, 0]])),
                )
                fov_mesh.paint_uniform_color([1.0, 0.5, 0.0])
                odom_vis_list.append(fov_mesh)

                # odom viz
                small_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            odom.data[0].item(),
                            odom.data[1].item(),
                            odom.data[2].item(),
                        )
                    )
                )

                # plot goal
                o3d.visualization.draw_geometries(odom_vis_list)

        if self.debug:
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points
            odom_vis_list = []

            for distance in self._cfg.distance_scheme.keys():
                odoms = torch.vstack(self.category_scheme_pairs[distance].odom_list)
                odoms = odoms.tensor().cpu().numpy()[:, :3]
                for idx, odom in enumerate(odoms):
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((odom[0], odom[1], odom[2])))
                    if idx > 10:
                        break
            # viz cost map
            odom_vis_list.append(self.cost_map.pcd_tsdf)

            # plot goal
            o3d.visualization.draw_geometries(odom_vis_list)

        return

    # 分层采样与数据归档（Stratified Sampling & Bucketing）。
    # 它的作用是从一堆符合特定几何条件（比如都在视野内）的候选目标点中，按照距离远近进行筛选，并随机抽取一定数量的样本，存入相应的数据容器中。
    # 如果没有这个函数，训练数据可能会充斥着大量重复的、或者距离分布极不均匀的样本（例如全是 0.5米的短路径，没有 5米的长路径）
    # 过滤：去掉太近或方位不对的点。
    # 平衡：强迫数据集在短距离、中距离、长距离上都有分布，而不是堆积在某一类距离上。
    # 采样：限制每个起点的样本数量（每个距离段最多 3 个），防止数据集爆炸。
    # 存储：将处理好的数据（起点Odom、终点Goal、图片路径）塞进最终的数据结构 category_scheme_pairs 中，等待被打包成 Dataset。
    def reduce_pairs(
        self,
        odom_idx: int,
        goals: pp.LieTensor,
        decision_tensor: torch.Tensor,
        odom_distances: dict,
        warp_img_path: Optional[str],
        within_fov: bool = False,
        behind_robot: bool = False,
        front_of_robot: bool = False,
    ):
        # remove all goals depending on the decision tensor from the odom_distances dict
        keep_distance_entries = decision_tensor[list(odom_distances.keys())]
        distances = np.array(list(odom_distances.values()))[keep_distance_entries.numpy()]
        goal_idx = np.array(list(odom_distances.keys()))[keep_distance_entries.numpy()]

        # max distance enforced odom_distances, here enforce min distance
        within_distance_idx = distances > self._cfg.min_goal_distance
        goal_idx = goal_idx[within_distance_idx]
        distances = distances[within_distance_idx]

        # check if there are any goals left
        if len(goal_idx) == 0:
            return

        # select the goal according to the distance_scheme
        for distance in self._cfg.distance_scheme.keys():
            # select nbr_samples from goals within distance
            within_curr_distance_idx = distances < distance
            if sum(within_curr_distance_idx) == 0:
                continue
            selected_idx = np.random.choice(
                goal_idx[within_curr_distance_idx],
                min(3, sum(within_curr_distance_idx)),
                replace=False,
            )
            # remove the selected goals from the list for further selection
            distances = distances[~within_curr_distance_idx]
            goal_idx = goal_idx[~within_curr_distance_idx]

            for idx in selected_idx:
                self.category_scheme_pairs[distance].update_buffers(
                    odom=self.odom_array_depth[odom_idx],
                    goal=goals[idx],
                    within_fov=within_fov,
                    front_of_robot=front_of_robot,
                    behind_robot=behind_robot,
                    depth_filename=self.depth_filename_list[odom_idx],
                    sem_rgb_filename=warp_img_path,
                )

    def get_goal_categories(self, goal_odom_frame: pp.LieTensor):
        """
        Decide which of the samples are within the fov, in front of the robot or behind the robot.
        """
        # get if odom-goal is within fov or outside the fov but still in front of the robot
        goal_angle = abs(torch.atan2(goal_odom_frame.data[:, 1], goal_odom_frame.data[:, 0]))
        within_fov = goal_angle < self.alpha_fov / 2 * self._cfg.fov_scale
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

    @staticmethod
    def compute_pixel_tensor(x_nums: int, y_nums: int, K_depth: np.ndarray) -> None:
        # get image plane mesh grid
        pix_u = np.arange(0, y_nums)
        pix_v = np.arange(0, x_nums)
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack([pixels, np.ones((len(pixels), 1))])  # add ones for 3D coordinates

        # transform to camera frame
        k_inv = np.linalg.inv(K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        return pix_cam_frame[[2, 0, 1], :].T * np.array([1, -1, -1])

    def _load_depth_image(self, depth_filename):
        if depth_filename.endswith(".png"):
            depth_image = Image.open(depth_filename)
            if self._cfg.real_world_data:
                depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
            else:
                depth_image = np.array(depth_image)
        else:
            depth_image = np.load(depth_filename)

        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / self._cfg.depth_scale).astype("float32")
        depth_image[depth_image > self._cfg.max_depth] = 0.0
        return depth_image

    @staticmethod
    def compute_overlay(
        pose_dep,
        pose_sem,
        depth_img,
        sem_rgb_image,
        pix_depth_cam_frame,
        K_sem_rgb,
    ):
        # get 3D points of depth image
        rot = tf.Rotation.from_quat(pose_dep[3:]).as_matrix()
        dep_im_reshaped = depth_img.reshape(
            -1, 1
        )  # flip s.t. start in lower left corner of image as (0,0) -> has to fit to the pixel tensor
        points = dep_im_reshaped * (rot @ pix_depth_cam_frame.T).T + pose_dep[:3]

        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_sem[3:]).as_matrix().T @ (points - pose_sem[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]] * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (K_sem_rgb @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < sem_rgb_image.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < sem_rgb_image.shape[0])
        )
        # get semantic annotation
        sem_annotation = np.zeros((pixels.shape[0], 3), dtype=np.uint8)
        sem_annotation[filter_idx] = sem_rgb_image[
            pixels[filter_idx, 1].astype(int),
            pixels[filter_idx, 0].astype(int),
        ]
        # reshape to image

        return sem_annotation.reshape(depth_img.shape[0], depth_img.shape[1], 3)

    def _get_overlay_img(self, odom_idx):
        # get corresponding filenames
        depth_filename = self.depth_filename_list[odom_idx]
        sem_rgb_filename = self.sem_rgb_filename_list[odom_idx]

        # load semantic and depth image and get their poses
        depth_img = self._load_depth_image(depth_filename)
        sem_rgb_image = Image.open(sem_rgb_filename)
        if self._cfg.real_world_data:
            sem_rgb_image = np.array(sem_rgb_image.transpose(PIL.Image.ROTATE_180))
        else:
            sem_rgb_image = np.array(sem_rgb_image)
        pose_dep = self.odom_array_depth[odom_idx].data.cpu().numpy()
        pose_sem = self.odom_array_sem_rgb[odom_idx].data.cpu().numpy()

        sem_rgb_image_warped = self.compute_overlay(
            pose_dep,
            pose_sem,
            depth_img,
            sem_rgb_image,
            self.pix_depth_cam_frame,
            self.K_sem_rgb,
        )
        assert sem_rgb_image_warped.dtype == np.uint8, "sem_rgb_image_warped has to be uint8"

        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(depth_img)
            ax2.imshow(sem_rgb_image_warped / 255)
            ax3.imshow(sem_rgb_image)
            # ax3.imshow(depth_img)
            # ax3.imshow(sem_rgb_image_warped / 255, alpha=0.5)
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            plt.show()

        # save semantic image under the new path
        sem_rgb_filename = os.path.split(sem_rgb_filename)[1]
        sem_rgb_image_path = os.path.join(self.root, "img_warp", sem_rgb_filename)
        sem_rgb_image_warped = cv2.cvtColor(sem_rgb_image_warped, cv2.COLOR_RGB2BGR)  # convert to BGR for cv2
        assert cv2.imwrite(sem_rgb_image_path, sem_rgb_image_warped)

        return sem_rgb_image_path

    """Noise Edges helper functions"""
    # 模拟真实深度相机（如 RealSense）在物体边缘处的测量缺陷。
    # 它的核心逻辑是：检测深度图像中的物体边缘，并将边缘附近的深度值强制设为 0（即标记为无效/无数据）。
    def noise_edges(self):
        """
        Along the edges in the depth image, set the values to 0.
        Mimics the real-world behavior where RealSense depth cameras have difficulties along edges.
        """
        print("[INFO] Adding noise to edges in depth images ...", end=" ")
        new_depth_filename_list = []
        # create new directory
        depth_noise_edge_dir = os.path.join(self.root, "depth_noise_edges")
        os.makedirs(depth_noise_edge_dir, exist_ok=True)

        for depth_filename in self.depth_filename_list:
            depth_img = self._load_depth_image(depth_filename)
            # Perform Canny edge detection
            image = ((depth_img / depth_img.max()) * 255).astype(np.uint8)  # convert to CV_U8 format
            edges = cv2.Canny(image, self._cfg.edge_threshold, self._cfg.edge_threshold * 3)
            # Dilate the edges to extend their space
            kernel = np.ones(self._cfg.extend_kernel_size, np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            # Erode the edges to refine their shape
            eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
            # modify depth image
            depth_img[eroded_edges == 255] = 0.0
            # save depth image
            depth_img = (depth_img * self._cfg.depth_scale).astype("uint16")
            if depth_filename.endswith(".png"):
                assert cv2.imwrite(
                    os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]),
                    depth_img,
                )
            else:
                np.save(
                    os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]),
                    depth_img,
                )
            new_depth_filename_list.append(os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]))

        self.depth_filename_list = new_depth_filename_list
        print("Done!")
        return

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


# EoF
