# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 主要用于计算机器人规划轨迹的代价（Cost/Loss）
# 训练阶段：作为损失函数（Loss Function），指导神经网络学习生成无碰撞、平滑且符合动力学的轨迹。
# 优化阶段：评估生成轨迹的质量，用于选择最优路径。

# 该类 TrajCost 接收一组预测的路径点（Waypoints）、机器人的当前状态（Odometry）、目标点（Goal）以及环境地图（Cost Map），然后计算出一个标量值（Cost）。这个 Cost 越小，说明轨迹越好（无碰撞、平滑、接近目标）。
# 坐标变换：利用李群库 pypose 将路径点从机器人局部坐标系转换到世界坐标系。
# 地图采样：利用 torch.nn.functional.grid_sample 在离散的栅格地图（Cost Map/Height Map）上进行双线性插值采样，获取路径点处的障碍物代价值和地形高度值。这是实现端到端可微分训练的关键。
# 多目标优化：总代价是多个子代价的加权和：
# 碰撞风险学习（Fear）：除了几何代价，它还计算一个“恐惧标签”（Fear Label），用于监督学习网络预测当前状态是否危险（即是否即将发生碰撞）。

from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_dtype(torch.float32)

from cost_maps import OccupancyCostMap  # 处理点云地图或栅格地图的类

# visual-imperative-planning
from traj_cost_opt.traj_opt_myself import TrajOpt

try:
    import pypose as pp  # only used for training 用于处理机器人位姿变换（SE3）
    import wandb  # only used for training 用于训练时的日志记录
except ModuleNotFoundError or ImportError:  # eval in issac sim  # TODO: check if all can be installed in Isaac Sim
    print("[Warning] pypose or wandb not found, only use for evaluation")


class SimpleCostMapWrapper:
    def __init__(self, map_array, resolution=0.1, origin_x=-10.0, origin_y=-10.0, device='cpu'):
        # 将 numpy 转为 tensor
        self.cost_array = torch.tensor(map_array, dtype=torch.float32, device=device)
        self.num_x, self.num_y = map_array.shape
        
        self.center_index = self.num_x // 2, self.num_y // 2
        
        # 模拟配置结构 self.cost_map.cfg.general.resolution
        class Config:
            def __init__(self, res, x, y):
                self.general = type('General', (), {'resolution': res})
                self.x_start = x
                self.y_start = y
        self.cfg = Config(resolution, origin_x, origin_y)

    def Pos2Ind(self, points):
        """
        只负责将世界坐标转换为图像像素坐标 (Row, Col)
        不进行归一化，方便可视化使用
        """
        device = points.device
        px = points[..., 0]
        py = points[..., 1]
        
        res = self.cfg.general.resolution
        c_row = self.center_index[0]
        c_col = self.center_index[1]
        
        # 计算像素坐标
        row_idx = c_row - (px / res)
        col_idx = c_col - (py / res)
        
        return torch.stack((row_idx, col_idx), dim=-1)
    
    def Pos2IndNormal(self, points):
        ind = self.Pos2Ind(points)
        
        # 2. 获取尺寸 (确保 self.cost_array 能被访问到)
        H, W = self.cost_array.shape
        
        # 3. 归一化公式: 2 * (x / (size-1)) - 1
        # 注意：这里 ind[..., 0] 是 row, ind[..., 1] 是 col
        norm_row = 2.0 * (ind[..., 0] / (H - 1)) - 1.0
        norm_col = 2.0 * (ind[..., 1] / (W - 1)) - 1.0
        
        # 4. 堆叠 (配合 cost_map.T 使用，顺序为 Row, Col)
        grid = torch.stack((norm_row, norm_col), dim=-1) 
        
        # 5. 截断防止越界
        grid = torch.clamp(grid, -1.0, 1.0)
        return grid

class TrajCost:
    debug = False

    def __init__(
        self,
        gpu_id: Optional[int] = 0,
        log_data: bool = False,
        w_obs: float = 0.25,
        w_motion: float = 1.5,
        w_goal: float = 2.0,
        obstalce_thread: float = 0.75,
        robot_width: float = 0.6,
        robot_max_moving_distance: float = 0.15,
    ) -> None:
        # init map and optimizer
        self.device = gpu_id
        self.cost_map: OccupancyCostMap = None
        self.opt = TrajOpt()
        self.is_map = False
        self.neg_reward: torch.Tensor = None

        # loss weights
        self.w_obs = w_obs  # 障碍物代价权重
        self.w_motion = w_motion  # 运动平滑代价权重
        self.w_goal = w_goal  # 目标距离代价权重

        # fear label threshold value
        self.obstalce_thread = obstalce_thread

        # footprint radius
        self.robot_width = robot_width
        self.robot_max_moving_distance = robot_max_moving_distance

        # logging
        self.log_data = log_data
        self.debug = False
        return

    # 这个地方需要加一个栅格地图index与实际局部地图坐标系下的转换吗？
    @staticmethod
    def TransformPoints(odom, points):
        # 1. 自动适配 odom 维度
        if odom.shape[-1] == 4:
            # 假设输入是 [x, y, yaw]，转换为 SE3 [x, y, 0, qx, qy, qz, qw]
            # 注意：如果你的输入是 [x, y, z]，这里的逻辑需要改为只补四元数
            if odom.device != points.device:
                odom = odom.to(points.device)
            # 这里以 [x, y, yaw] 为例（平面导航常见）
            batch_size = odom.shape[0]
            zeros = torch.zeros((batch_size, 1), device=odom.device)
            ones = torch.ones((batch_size, 1), device=odom.device)
            
            x = odom[:, 0:1]
            y = odom[:, 1:2]
            yaw = torch.atan2(odom[:, 3:4], odom[:, 2:3])
            
            # Euler (yaw) to Quaternion
            # cy = cos(yaw * 0.5), sy = sin(yaw * 0.5)
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            
            # 构建 SE3 (x, y, z=0, qx=0, qy=0, qz=sy, qw=cy)
            odom_se3 = torch.cat([x, y, zeros, zeros, zeros, sy, cy], dim=-1)
            
        elif odom.shape[-1] == 7:
            odom_se3 = odom
        else:
            raise ValueError(f"Unsupported odom shape: {odom.shape}")

        # 2. 变换计算
        batch_size, num_p, dims = points.shape # 获取输入点的维度
        
        # 创建 identity SE3 用于存放 points
        world_ps = pp.identity_SE3(
            batch_size,
            num_p,
            device=points.device,
            requires_grad=points.requires_grad,
        )
        
        # === 修复开始: 自动适配 2D/3D 输入 ===
        if dims == 2:
            # 如果输入只有 x,y，补一个 z=0
            zeros = torch.zeros((batch_size, num_p, 1), device=points.device, dtype=points.dtype)
            points_3d = torch.cat([points, zeros], dim=-1)
            # 赋值给 SE3 的前3位 (Translation)
            world_ps.tensor()[:, :, 0:3] = points_3d
        elif dims == 3:
            # 如果已经是 3D，直接赋值
            world_ps.tensor()[:, :, 0:3] = points
        else:
             raise ValueError(f"Unsupported points dimension: {dims}")
        # === 修复结束 ===
        
        # 执行变换: T_world = T_odom * T_local
        # 注意：使用 odom_se3[:, None, :] 进行广播
        world_ps = pp.SE3(odom_se3[:, None, :]) @ world_ps
        
        return world_ps

    # 加载 TSDF（截断符号距离场）地图或语义地图，后续用于查询障碍物信息。
    # 这个怎么会这样来读取地图？为了满足内存需要，不copy地图？
    def SetMap(self, root_path, map_name):
        if not map_name.endswith('.npy'):
            file_name = map_name + '.npy'
        else:
            file_name = map_name
        file_path = os.path.join(root_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found at: {file_path}")
        print(f"[INFO] Loading cost map from {file_path}...")
        
        # 1. 加载数据
        npy_data = np.load(file_path)
        
        # 2. 确定设备 (GPU/CPU)
        device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        
        # 3. 【关键】使用 Wrapper 包装，而不是直接赋值
        # 注意：你需要根据你的实际地图设置 resolution 和 origin
        # 如果不知道，先写死一个默认值让代码跑起来
        self.cost_map = SimpleCostMapWrapper(
            npy_data, 
            resolution=0.1,       # 假设分辨率 0.1m
            origin_x=0.0,       # 假设原点 X
            origin_y=0.0,       # 假设原点 Y
            device=device
        )
        self.is_map = True
        
        # 验证物理坐标到像素坐标的转换
        # import matplotlib.pyplot as plt

        # # 假设有一个物理坐标点
        # phys_point = torch.tensor([[2.0, 3.0, 0.0]], device=device)  # 物理坐标 (x=2.0, y=3.0)
        # phys_point = phys_point.unsqueeze(0)  # [1, 1, 3]  batch=1, num_points=1

        # # 转换为像素坐标
        # pixel_idx = self.cost_map.Pos2Ind(phys_point)  # [1, 1, 2]
        # print(f"[DEBUG] Physical Point: {phys_point[0,0,:].cpu().numpy()}, Pixel Index: {pixel_idx[0,0,:].cpu().numpy()}")
        # pixel_idx_np = pixel_idx[0, 0].cpu().numpy()
        # print(f"[DEBUG] Transformed Pixel Index: Row={pixel_idx_np[0]:.2f}, Col={pixel_idx_np[1]:.2f}")

        # # 显示地图和点
        # plt.figure()
        # plt.imshow(self.cost_map.cost_array.cpu().numpy(), cmap='viridis')
        # plt.scatter([pixel_idx_np[1]], [pixel_idx_np[0]], c='red', marker='x', s=100, label='Transformed Point')
        # plt.title(f"Physical ({phys_point[0,0,0].item():.2f},{phys_point[0,0,1].item():.2f}) -> Pixel ({pixel_idx_np[0]:.1f},{pixel_idx_np[1]:.1f})")
        # plt.legend()
        # plt.show()
    
    def _compute_motion_loss(self, waypoints, mask=None):
        """
        严谨考虑 Mask 的动力学 Loss 计算。
        
        Args:
            waypoints: [Batch, N, 2]
            mask: [Batch, N] (True 表示无效/Padding, False 表示有效)
        """
        device = waypoints.device
        batch_size, num_p, dims = waypoints.shape
        eps = 1e-6

        # 如果没有传入 mask，创建一个全 False 的 mask (所有点有效)
        if mask is None:
            mask = torch.zeros((batch_size, num_p), dtype=torch.bool, device=device)

        # ==========================================
        # 1. 计算物理量 (Derivatives)
        # ==========================================
        
        # 速度向量 V [B, N-1, 2]
        v_vec = waypoints[:, 1:] - waypoints[:, :-1]
        # 速度 Mask [B, N-1] (如果终点 i+1 无效，则段 i 无效)
        mask_v = mask[:, 1:] 

        # 加速度向量 A [B, N-2, 2]
        a_vec = v_vec[:, 1:] - v_vec[:, :-1]
        # 加速度 Mask [B, N-2]
        mask_a = mask[:, 2:]

        # 加加速度向量 Jerk [B, N-3, 2]
        j_vec = a_vec[:, 1:] - a_vec[:, :-1]
        # Jerk Mask [B, N-3]
        mask_j = mask[:, 3:]

        # ==========================================
        # 2. 计算各项 Loss (Apply Mask)
        # ==========================================
        
        # --- A. Accel Loss (平滑性 - 二阶导 L2) ---
        # 计算平方和: [B, N-2]
        acc_sq = torch.sum(a_vec ** 2, dim=-1)
        # 将无效部分的 Loss 设为 0
        acc_sq = acc_sq.masked_fill(mask_a, 0.0)
        # 计算分母：每个 Batch 有效的加速度点数
        valid_count_a = (~mask_a).sum(dim=1).clamp(min=1.0) # [B]
        # 求均值: 这里的均值是对“有效点”求均值
        acc_loss = torch.mean(torch.sum(acc_sq, dim=1) / valid_count_a)

        # --- B. Jerk Loss (舒适度 - 三阶导 L2) ---
        jerk_sq = torch.sum(j_vec ** 2, dim=-1)
        jerk_sq = jerk_sq.masked_fill(mask_j, 0.0)
        valid_count_j = (~mask_j).sum(dim=1).clamp(min=1.0)
        jerk_loss = torch.mean(torch.sum(jerk_sq, dim=1) / valid_count_j)

        # --- C. Curvature Loss (几何平滑 - 余弦相似度) ---
        # 归一化速度向量 (防止除以0)
        v_norm = torch.norm(v_vec, dim=-1, keepdim=True) + eps
        v_dir = v_vec / v_norm # [B, N-1, 2]
        
        # 计算相邻向量的点积 (Cosine Similarity) -> [B, N-2]
        cos_sim = torch.sum(v_dir[:, :-1] * v_dir[:, 1:], dim=-1)
        
        # Loss = 1 - cos_sim (1.0是直线, -1.0是掉头)
        # 只有当两个向量都有效时，夹角才有效 -> 使用 mask_a
        curv_val = 1.0 - cos_sim
        curv_val = curv_val.masked_fill(mask_a, 0.0)
        # 分母使用 mask_a 的计数
        curvature_loss = torch.mean(torch.sum(curv_val, dim=1) / valid_count_a)

        # --- D. Uniformity Loss (速度均匀性) ---
        # 我们希望在一条轨迹内部，速度大小是恒定的 (方差为0)
        # 速度大小: [B, N-1]
        v_mags = torch.norm(v_vec, dim=-1)
        
        # 1. 先把无效速度设为 0，方便求和
        v_mags_clean = v_mags.masked_fill(mask_v, 0.0)
        # 2. 计算每个 Batch 的平均速度
        valid_count_v = (~mask_v).sum(dim=1).clamp(min=1.0)
        mean_v = torch.sum(v_mags_clean, dim=1, keepdim=True) / valid_count_v.unsqueeze(1) # [B, 1]
        
        # 3. 计算方差 (v - mean_v)^2
        # 注意：这里不仅要 mask 计算结果，还要确保不要把 padding 的 0 和 mean_v 做差导致产生 loss
        var_v = (v_mags - mean_v) ** 2
        var_v = var_v.masked_fill(mask_v, 0.0)
        
        uniformity_loss = torch.mean(torch.sum(var_v, dim=1) / valid_count_v)

        # --- E. [新增] Length Loss (路径总长度最小化) ---
        # 目的：消除绕路，拉紧轨迹
        # 使用 v_mags (每段的长度)
        # 注意：这里我们求 SUM (总长度)，而不是 Mean (平均步长)
        
        # 1. 确保无效段长度为 0
        # --- E. [修改] Energy Loss (弹簧能量损失 - 长度平方和) ---
        # 目的：
        # 1. Shortest Path: 缩短总路径
        # 2. Uniformity: 强迫点与点之间间距相等 (因为 a^2+b^2 >= 2ab, 均分时最小)
        
        # 1. 计算长度平方 [B, N-1]
        dist_sq = v_mags ** 2
        
        # 2. Mask 掉无效段 (设为 0)
        dist_sq = dist_sq.masked_fill(mask_v, 0.0)
        
        # 3. 对每条轨迹求和 [B]
        # 注意：这里是 Sum，代表整条轨迹的总能量
        trajectory_energy = torch.sum(dist_sq, dim=1)
        
        # 4. Batch 平均
        length_loss = torch.mean(trajectory_energy)
        # ==========================================
        # 3. 组合权重
        # ==========================================
        # 权重建议：
        # Accel: 最核心，拉直轨迹
        # Jerk: 辅助，微调
        # Curv: 关键，防止急转弯 (数值通常在 0~2 之间，权重给高一点)
        # Uniform: 辅助，防止点堆积
        
        total_motion_loss = (
            1.0 * acc_loss + 
            0.5 * jerk_loss + 
            2.0 * curvature_loss +  # 提高权重，因为这个 Loss 很重要
            0.5 * uniformity_loss + 
            5 * length_loss
        )

        return total_motion_loss, acc_loss, curvature_loss
    # 计算轨迹总代价 CostofTraj (核心函数)
    def CostofTraj(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        dataset: str = "train",
        mask: Optional[torch.Tensor] = None, # <--- 新增 mask 参数 [B, K] (True表示无效)
    ):
        batch_size, num_p, _ = waypoints.shape
        assert self.is_map, "Map has to be set for cost calculation"

        # 1. 计算原始 Obstacle Loss 矩阵 [B, K]
        oloss_M = self._compute_oloss(waypoints, batch_size)
        
        # === 修正 A: 应用 Mask 到 oloss ===
        if mask is not None:
            # oloss_M 的长度是 N-1，而 mask 的长度是 N
            # 我们必须对 mask 进行切片以匹配 oloss_M
            # 策略: 如果点 i+1 无效，则认为从 i 到 i+1 的段（oloss_M[i]）无效
            seg_mask = mask[:, 1:]  # [B, N-1]
            
            # 使用切片后的 mask 进行填充
            oloss_M = oloss_M.masked_fill(seg_mask, 0.0)
            
            # 计算平均值时，分母应该是有效段的数量
            valid_counts = (~seg_mask).sum(dim=1) + 1e-6
            # 注意：这里的 num_p 是 N，但 oloss_M 是 N-1。
            # 为了保持量级，乘以 (num_p - 1) 可能更准确，或者保持 num_p 也行，差别不大
            oloss = torch.mean(torch.sum(oloss_M, axis=1) / valid_counts * (num_p - 1))
        else:
            oloss = torch.mean(torch.sum(oloss_M, axis=1))

        # 2. Goal Loss (通常只看最后一个点，或者看 Mask 之前的最后一个有效点)
        # 如果你已经有了变长逻辑，Gloss 应该取最后一个“有效”点，而不是数组的最后一个点
        if mask is not None:
            # 找到每个 batch 最后一个有效点的索引
            # mask: True 是无效，False 是有效
            # valid_len = (~mask).sum(dim=1)
            # last_valid_idx = valid_len - 1
            # end_points = waypoints[torch.arange(batch_size), last_valid_idx]
            
            # 简化版：通常 TrajGenerator 会把多余的点堆在最后，所以直接取 -1 也可以
            # 只要网络学会把点堆在终点，取 -1 和取 last_valid 是一样的
            # gloss_M = torch.norm(goal[:, :2] - waypoints[:, -1, :2], dim=1)
            valid_len = (~mask).sum(dim=1).long()
            last_valid_idx = (valid_len - 1).clamp(min=0)
            batch_idx = torch.arange(waypoints.shape[0], device=waypoints.device)
            end_points = waypoints[batch_idx, last_valid_idx, :2]
            gloss_M = torch.norm(goal[:, :2] - end_points, dim=1)
        else:
            gloss_M = torch.norm(goal[:, :2] - waypoints[:, -1, :2], dim=1)
            
        gloss = torch.mean(torch.log(gloss_M + 1.0))

        # 3. Moving Loss (平滑性)
        
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:2], step=1.0 / (num_p - 1), start_vel=odom[:, 2:4], goal_vel=goal[:, 2:4])
        
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0 : num_p - 1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0 : num_p - 1, :], dim=2)
              
        raw_mloss = torch.abs(desired_ds - wp_ds)
        
        # === 修正 B: 应用 Mask 到 mloss ===
        if mask is not None:
            # mask 的维度通常对应点，而 ds 对应段 (点数-1)
            # 我们取段起点的 mask 作为该段的 mask
            # mask[:, :-1]
            seg_mask = mask[:, 1:] # 如果终点无效，那连接终点的线段也无效
            
            raw_mloss = raw_mloss.masked_fill(seg_mask, 0.0)
            valid_seg_counts = (~seg_mask).sum(dim=1) + 1e-6
            mloss = torch.mean(torch.sum(raw_mloss, axis=1) / valid_seg_counts) # 已经被 mask 了，直接求和
        else:
            mloss = torch.sum(raw_mloss, axis=1) # 已经被 mask 了，直接求和
            
        mloss = torch.mean(mloss)
        # print(f"[DEBUG] mloss final value:\n{mloss.detach().cpu().numpy()}")
        
        # mloss, acc_debug, curv_debug = self._compute_motion_loss(waypoints, mask)
        v_vec = waypoints[:, 1:] - waypoints[:, :-1]
        v_mags = torch.norm(v_vec, dim=-1)
        dist_sq = v_mags ** 2
        mask_v = mask[:, 1:]
        # 2. Mask 掉无效段 (设为 0)
        dist_sq = dist_sq.masked_fill(mask_v, 0.0)
        
        # 3. 对每条轨迹求和 [B]
        # 注意：这里是 Sum，代表整条轨迹的总能量
        trajectory_energy = torch.sum(dist_sq, dim=1)
        
        
        # Complete Trajectory Loss
        trajectory_loss = self.w_obs * oloss + self.w_motion * mloss + self.w_goal * gloss + 0.5 * torch.mean(trajectory_energy)
        print(f"Obs: {self.w_obs*oloss:.4f}, Motion: {self.w_motion*mloss:.4f}, Goal: {self.w_goal*gloss:.4f}")
        
        # Fear labels # 计算路径累积距离
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        # print(f"[DEBUG] goal_dists values before vstack:\n{goal_dists.detach().cpu().numpy()}")
        goal_dists = torch.vstack([goal_dists] * 3)
        # print(f"[DEBUG] goal_dists values after vstack:\n{goal_dists.detach().cpu().numpy()}")
        # oloss_M (原始障碍物代价图)
        floss_M = torch.clone(oloss_M)
        # 忽略超过一定预瞄距离(ahead_dist)后的障碍物影响
        floss_M[goal_dists > ahead_dist] = 0.0
        # print(f"[DEBUG] floss_M values after masking:\n{floss_M.detach().cpu().numpy()}")
        # 如果路径上某点的障碍物值超过阈值，则标记为“危险” (Label=1)
        # 在当前的一条轨迹中，寻找最大的障碍物代价值。
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # print(f"[DEBUG] fear_labels values before thresholding:\n{fear_labels.detach().cpu().numpy()}")
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thread) 
        # 如果这条路径上（在 ahead_dist 范围内）遇到的最大障碍物代价超过了阈值（obstalce_thread），则认为这条路是死路或危险路径
        fear_labels = fear_labels > self.obstalce_thread
        # print(f"[DEBUG] fear_labels values after thresholding:\n{fear_labels.detach().cpu().numpy()}")
        fear_labels = torch.any(fear_labels.reshape(3, batch_size).T, dim=1, keepdim=True).to(torch.float32)
        # print(f"[DEBUG] final fear_labels values:\n{fear_labels.detach().cpu().numpy()}")
        # Fear loss # 计算二元交叉熵损失 (BCE Loss)，训练网络预测“恐惧”
        # fear：这是神经网络预测出来的碰撞概率（0~1之间）。
        # fear_labels：这是上面根据地图计算出来的实际碰撞情况（0或1）。
        # 含义：使用二元交叉熵损失（BCE Loss）来训练神经网络，使其能根据图像输入准确预测当前的危险程度。
        if fear.shape != fear_labels.shape:
            fear = fear.view_as(fear_labels)
        collision_probabilty_loss = nn.BCELoss()(fear, fear_labels.float())
        # print(f"[DEBUG] collision_probabilty_loss value:\n{collision_probabilty_loss.detach().cpu().numpy()}")
        print(f"Collision Probabilty Loss: {collision_probabilty_loss:.4f}")
        # TODO: kinodynamics cost
        return collision_probabilty_loss + trajectory_loss


    def obs_cost_eval(self, odom: torch.Tensor, waypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Obstacle Loss for eval_sim_static script!

        Args:
            odom (torch.Tensor): Current odometry
            waypoints (torch.Tensor): waypoints in camera frame

        Returns:
            tuple: mean obstacle loss for each trajectory, max obstacle loss for each trajectory
        """
        assert self.is_map, "Map has to be loaded for evaluation"

        # compute obstacle loss
        world_ps = self.TransformPoints(odom, waypoints).tensor()
        oloss_M = self._compute_oloss(world_ps, waypoints.shape[0])
        # account for negative reward
        oloss_M = oloss_M - self.neg_reward[2]
        oloss_M[oloss_M < 0] = 0.0
        oloss_M = oloss_M.reshape(-1, waypoints.shape[0], oloss_M.shape[1])
        return torch.mean(oloss_M, axis=[0, 2]), torch.amax(oloss_M, dim=[0, 2])

    def cost_of_recorded_path(
        self,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Cost of recorded path - for evaluation only

        Args:
            waypoints (torch.Tensor): Path coordinates in world frame
        """
        assert self.is_map, "Map has to be loaded for evaluation"
        oloss_M = self._compute_oloss(waypoints.unsqueeze(0), 1)
        return torch.max(oloss_M)

    def _compute_oloss(self, world_ps, batch_size):
        import numpy as np
        if world_ps.shape[1] == 1:  # special case when evaluating cost of a recorded path # 如果只有一个点，不进行膨胀
            world_ps_inflated = world_ps
        else:
            # include robot dimension as square # 1. 计算路径切向量 (Tangent)
            tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]  # get tangent vector
            tangent = tangent / torch.norm(tangent, dim=2, keepdim=True)  # normalize normals vector
            # 2. 计算法向量 (Normal) = 切向量旋转90度
            normals = tangent[:, :, [1, 0]] * torch.tensor(
                [-1, 1], dtype=torch.float32, device=world_ps.device
            )  # get normal vector
            world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)  # duplicate points
            # 3. 膨胀：生成左、中、右三个点，模拟机器人的宽度
            # front_right, center, front_left
            world_ps_inflated[:, :, 0:2] = torch.vstack(
                [
                    # movement corners
                    world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,  # front_right
                    world_ps[:, :-1, 0:2],  # center
                    world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,  # front_left
                ]
            )
        # 将膨胀后的点转换为地图索引
        cost_idx = self.cost_map.Pos2Ind(world_ps_inflated)

        # 可视化 cost_idx 上的点，两组不同颜色
        if self.debug:
            import matplotlib.pyplot as plt
            
            # 1. 专门为画图计算一次完整的中心线索引
            full_center_idx = self.cost_map.Pos2Ind(world_ps)

            # 准备底图数据
            cost_map_np = self.cost_map.cost_array.cpu().numpy()
            H, W = cost_map_np.shape  # 获取地图尺寸，用于边界检查

            plt.figure(figsize=(12, 10))
            # 显示底图 (注意 origin 设置，确保跟你的坐标系一致)
            plt.imshow(cost_map_np, cmap='viridis', origin='upper')

            colors = ['red', 'blue', 'green', 'orange', 'purple']

            # 2. 遍历每一个 Batch
            for b in range(batch_size):
                color = colors[b % len(colors)]

                # --- A. 获取坐标数据 ---
                # 中心线 [N, 2]
                c_pts = full_center_idx[b].detach().cpu().numpy()
                
                # 左右膨胀线 [N-1, 2]
                right_pts = cost_idx[b].detach().cpu().numpy()
                left_pts = cost_idx[b + 2 * batch_size].detach().cpu().numpy()

                # --- B. 绘图与标注 ---
                
                # 定义一个标注函数，避免重复代码
                def plot_with_labels(points, marker, label_prefix, alpha=1.0):
                    # points: [N, 2] (Row, Col)
                    # label_prefix: 只是为了区分打印log，或者不用
                    
                    # 画散点
                    # scatter(x=Col, y=Row)
                    plt.scatter(points[:, 1], points[:, 0], c=color, s=20, marker=marker, alpha=alpha)
                    
                    # 遍历每个点，添加 Cost 数值标签
                    for i in range(len(points)):
                        row, col = points[i, 0], points[i, 1]
                        
                        # 1. 安全转为整数索引
                        r_idx, c_idx = int(round(row)), int(round(col))
                        
                        # 2. 边界检查 (防止越界报错)
                        if 0 <= r_idx < H and 0 <= c_idx < W:
                            # 读取 Cost
                            cost_val = cost_map_np[r_idx, c_idx]
                            
                            # 3. 在点旁边添加文字
                            # text(x, y, string) -> (Col, Row)
                            plt.text(col, row, f"{cost_val:.1f}", 
                                    color='white', fontsize=7, fontweight='bold', 
                                    ha='right', va='bottom') # ha/va 调整文字相对点的位置
                
                # 1. 画中心线
                plt.plot(c_pts[:, 1], c_pts[:, 0], c=color, linewidth=2, label=f'Traj {b} Center')
                plot_with_labels(c_pts, marker='o', label_prefix='Center')

                # 2. 画右侧线
                plot_with_labels(right_pts, marker='x', label_prefix='Right', alpha=0.6)

                # 3. 画左侧线
                plot_with_labels(left_pts, marker='x', label_prefix='Left', alpha=0.6)

            plt.title(f"Visual Check: Path & Inflated Bounds with Cost Values (Batch={batch_size})")
            plt.xlabel("Map Col (X)")
            plt.ylabel("Map Row (Y)")
            plt.legend()
            plt.show()

        # Obstacle Cost
        # 采样 Cost Map
        # 任何落在障碍物区域的点都会采样到高 Cost
        sample_grid = self.cost_map.Pos2IndNormal(world_ps_inflated)
        cost_grid = self.cost_map.cost_array.T[None, None, ...].expand(
            world_ps_inflated.shape[0], 1, -1, -1
        )
        # print(f"[DEBUG] cost_grid shape: {cost_grid.shape}")
        # print(f"[DEBUG] cost_grid values:\n{cost_grid.detach().cpu().numpy()}")
        oloss_M = (
            F.grid_sample(
                cost_grid,
                sample_grid[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # print(f"[DEBUG] oloss_M before type cast:\n{oloss_M.detach().cpu().numpy()}")
        oloss_M = oloss_M.to(torch.float32)
        # print(f"[DEBUG] oloss_M values:\n{oloss_M.detach().cpu().numpy()}")
        return oloss_M


# EoF
