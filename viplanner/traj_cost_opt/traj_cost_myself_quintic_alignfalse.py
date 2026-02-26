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
        
        # 3. 归一化公式 (align_corners=False): 2 * ((x + 0.5) / size) - 1
        # 注意：这里 ind[..., 0] 是 row, ind[..., 1] 是 col
        norm_row = 2.0 * ((ind[..., 0] + 0.5) / H) - 1.0
        norm_col = 2.0 * ((ind[..., 1] + 0.5) / W) - 1.0
        
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
    
    
    # def _compute_motion_loss(self, waypoints, mask=None):
    #     """
    #     严谨考虑 Mask 的动力学 Loss 计算。 
    #     Args:
    #         waypoints: [Batch, N, 2]
    #         mask: [Batch, N] (True 表示无效/Padding, False 表示有效)
    #     """
    #     device = waypoints.device
    #     batch_size, num_p, dims = waypoints.shape
    #     eps = 1e-6
    #     # 如果没有传入 mask，创建一个全 False 的 mask (所有点有效)
    #     if mask is None:
    #         mask = torch.zeros((batch_size, num_p), dtype=torch.bool, device=device)
    #     # ==========================================
    #     # 1. 计算物理量 (Derivatives)
    #     # ==========================================
    #     # 速度向量 V [B, N-1, 2]
    #     v_vec = waypoints[:, 1:] - waypoints[:, :-1]
    #     # 速度 Mask [B, N-1] (如果终点 i+1 无效，则段 i 无效)
    #     mask_v = mask[:, 1:] 
    #     # 加速度向量 A [B, N-2, 2]
    #     a_vec = v_vec[:, 1:] - v_vec[:, :-1]
    #     # 加速度 Mask [B, N-2]
    #     mask_a = mask[:, 2:]
    #     # 加加速度向量 Jerk [B, N-3, 2]
    #     j_vec = a_vec[:, 1:] - a_vec[:, :-1]
    #     # Jerk Mask [B, N-3]
    #     mask_j = mask[:, 3:]
    #     # ==========================================
    #     # 2. 计算各项 Loss (Apply Mask)
    #     # ==========================================    
    #     # --- A. Accel Loss (平滑性 - 二阶导 L2) ---
    #     # 计算平方和: [B, N-2]
    #     acc_sq = torch.sum(a_vec ** 2, dim=-1)
    #     # 将无效部分的 Loss 设为 0
    #     acc_sq = acc_sq.masked_fill(mask_a, 0.0)
    #     # 计算分母：每个 Batch 有效的加速度点数
    #     valid_count_a = (~mask_a).sum(dim=1).clamp(min=1.0) # [B]
    #     # 求均值: 这里的均值是对“有效点”求均值
    #     acc_loss = torch.mean(torch.sum(acc_sq, dim=1) / valid_count_a)
    #     # --- B. Jerk Loss (舒适度 - 三阶导 L2) ---
    #     jerk_sq = torch.sum(j_vec ** 2, dim=-1)
    #     jerk_sq = jerk_sq.masked_fill(mask_j, 0.0)
    #     valid_count_j = (~mask_j).sum(dim=1).clamp(min=1.0)
    #     jerk_loss = torch.mean(torch.sum(jerk_sq, dim=1) / valid_count_j)
    #     # --- C. Curvature Loss (几何平滑 - 余弦相似度) ---
    #     # 归一化速度向量 (防止除以0)
    #     v_norm = torch.norm(v_vec, dim=-1, keepdim=True) + eps
    #     v_dir = v_vec / v_norm # [B, N-1, 2]
    #     # 计算相邻向量的点积 (Cosine Similarity) -> [B, N-2]
    #     cos_sim = torch.sum(v_dir[:, :-1] * v_dir[:, 1:], dim=-1)  
    #     # Loss = 1 - cos_sim (1.0是直线, -1.0是掉头)
    #     # 只有当两个向量都有效时，夹角才有效 -> 使用 mask_a
    #     curv_val = 1.0 - cos_sim
    #     curv_val = curv_val.masked_fill(mask_a, 0.0)
    #     # 分母使用 mask_a 的计数
    #     curvature_loss = torch.mean(torch.sum(curv_val, dim=1) / valid_count_a)
    #     # --- D. Uniformity Loss (速度均匀性) ---
    #     # 我们希望在一条轨迹内部，速度大小是恒定的 (方差为0)
    #     # 速度大小: [B, N-1]
    #     v_mags = torch.norm(v_vec, dim=-1) 
    #     # 1. 先把无效速度设为 0，方便求和
    #     v_mags_clean = v_mags.masked_fill(mask_v, 0.0)
    #     # 2. 计算每个 Batch 的平均速度
    #     valid_count_v = (~mask_v).sum(dim=1).clamp(min=1.0)
    #     mean_v = torch.sum(v_mags_clean, dim=1, keepdim=True) / valid_count_v.unsqueeze(1) # [B, 1]
    #     # 3. 计算方差 (v - mean_v)^2
    #     # 注意：这里不仅要 mask 计算结果，还要确保不要把 padding 的 0 和 mean_v 做差导致产生 loss
    #     var_v = (v_mags - mean_v) ** 2
    #     var_v = var_v.masked_fill(mask_v, 0.0)
    #     uniformity_loss = torch.mean(torch.sum(var_v, dim=1) / valid_count_v)
    #     # --- E. [新增] Length Loss (路径总长度最小化) ---
    #     # 目的：消除绕路，拉紧轨迹
    #     # 使用 v_mags (每段的长度)
    #     # 注意：这里我们求 SUM (总长度)，而不是 Mean (平均步长)
    #     # 1. 确保无效段长度为 0
    #     # --- E. [修改] Energy Loss (弹簧能量损失 - 长度平方和) ---
    #     # 目的：
    #     # 1. Shortest Path: 缩短总路径
    #     # 2. Uniformity: 强迫点与点之间间距相等 (因为 a^2+b^2 >= 2ab, 均分时最小)
    #     # 1. 计算长度平方 [B, N-1]
    #     dist_sq = v_mags ** 2
    #     # 2. Mask 掉无效段 (设为 0)
    #     dist_sq = dist_sq.masked_fill(mask_v, 0.0)
    #     # 3. 对每条轨迹求和 [B]
    #     # 注意：这里是 Sum，代表整条轨迹的总能量
    #     trajectory_energy = torch.sum(dist_sq, dim=1)
    #     # 4. Batch 平均
    #     length_loss = torch.mean(trajectory_energy)
    #     # ==========================================
    #     # 3. 组合权重
    #     # ==========================================
    #     # 权重建议：
    #     # Accel: 最核心，拉直轨迹
    #     # Jerk: 辅助，微调
    #     # Curv: 关键，防止急转弯 (数值通常在 0~2 之间，权重给高一点)
    #     # Uniform: 辅助，防止点堆积
    #     total_motion_loss = (
    #         1.0 * acc_loss + 
    #         0.5 * jerk_loss + 
    #         2.0 * curvature_loss +  # 提高权重，因为这个 Loss 很重要
    #         0.5 * uniformity_loss + 
    #         5 * length_loss
    #     )
    #     return total_motion_loss, acc_loss, curvature_loss
    
    
    # def _compute_motion_loss(self, waypoints, mask=None, dt=0.2):
    #     """
    #     针对包含速度的输入 [Batch, N, 4] 计算动力学 Loss
        
    #     Args:
    #         waypoints: [B, N, 4] -> (x, y, vx, vy)
    #         mask: [B, N] (True = 无效/Padding)
    #         dt: 时间步长 (假设为常数，例如 0.2s)
    #     """
    #     # --- 0. 数据解包与预处理 ---
    #     pred_pos = waypoints[..., 0:2]  # [B, N, 2]
    #     pred_vel = waypoints[..., 2:4]  # [B, N, 2]
        
    #     batch_size, num_p, _ = waypoints.shape
    #     device = waypoints.device

    #     if mask is None:
    #         mask = torch.zeros((batch_size, num_p), dtype=torch.bool, device=device)

    #     # 各种 Mask 的准备
    #     # Segment Mask (N-1段): 用于位置差分和速度一致性
    #     mask_seg = mask[:, 1:] 
    #     # Node Mask (N-2个): 用于加速度计算 (v_i+1 - v_i)
    #     # 注意：如果 v 是点属性，计算加速度会少一个点
    #     # mask_acc = mask[:, 1:] # 这里因为是 v 的差分，长度也是 N-1

    #     valid_cnt_seg = (~mask_seg).sum(dim=1).clamp(min=1.0)

    #     # --- 1. 运动学一致性 Loss (Kinematic Consistency) ---
    #     # 物理公式: p_{t+1} = p_t + v_{avg} * dt
    #     # 位置变化量
    #     pos_delta = pred_pos[:, 1:] - pred_pos[:, :-1] # [B, N-1, 2]
        
    #     # 速度积分 (使用梯形法则积分更准: (v_curr + v_next)/2 )
    #     # 或者简单点用 v_curr
    #     vel_avg = (pred_vel[:, :-1] + pred_vel[:, 1:]) / 2.0
    #     pos_expected = vel_avg * dt
        
    #     # 计算误差 (MSE)
    #     consistency_error = torch.sum((pos_delta - pos_expected) ** 2, dim=-1) # [B, N-1]
    #     consistency_error = consistency_error.masked_fill(mask_seg, 0.0)
    #     loss_consistency = torch.mean(torch.sum(consistency_error, dim=1) / valid_cnt_seg)

    #     # --- 2. 轨迹方向一致性 (Non-holonomic Constraint) ---
    #     # 目的：确保“速度的方向”和“位移的方向”是一致的 (防止侧滑/漂移)
    #     # 归一化向量 (加 eps 防止除零)
    #     dir_pos = pos_delta / (torch.norm(pos_delta, dim=-1, keepdim=True) + 1e-6)
    #     dir_vel = pred_vel[:, :-1] / (torch.norm(pred_vel[:, :-1], dim=-1, keepdim=True) + 1e-6)
        
    #     # Cosine Similarity: 1.0 = 完全一致
    #     # Loss = 1 - cos
    #     direction_diff = 1.0 - torch.sum(dir_pos * dir_vel, dim=-1)
    #     direction_diff = direction_diff.masked_fill(mask_seg, 0.0)
    #     loss_direction = torch.mean(torch.sum(direction_diff, dim=1) / valid_cnt_seg)

    #     # --- 3. 显式加速度 Loss (Smoothness via Explicit Velocity) ---
    #     # 直接利用预测的 v 计算 a
    #     # a = (v_{t+1} - v_t) / dt
    #     acc_vec = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
    #     acc_sq = torch.sum(acc_vec ** 2, dim=-1) # [B, N-1]
        
    #     # 应用 Mask (如果 v_{t+1} 无效，则该加速度无效)
    #     # 注意这里的 mask 也是 mask[:, 1:]，因为只要后一个点无效，差分就无效
    #     acc_sq = acc_sq.masked_fill(mask_seg, 0.0)
    #     loss_acc = torch.mean(torch.sum(acc_sq, dim=1) / valid_cnt_seg)

    #     # --- 4. 显式 Jerk Loss (可选) ---
    #     # j = (a_{t+1} - a_t) / dt
    #     # 长度变为 N-2
    #     jerk_vec = (acc_vec[:, 1:] - acc_vec[:, :-1]) / dt
    #     mask_jerk = mask[:, 2:]
    #     valid_cnt_jerk = (~mask_jerk).sum(dim=1).clamp(min=1.0)
        
    #     jerk_sq = torch.sum(jerk_vec ** 2, dim=-1)
    #     jerk_sq = jerk_sq.masked_fill(mask_jerk, 0.0)
    #     loss_jerk = torch.mean(torch.sum(jerk_sq, dim=1) / valid_cnt_jerk)

    #     dist_sq = torch.sum((pos_delta) ** 2, dim=-1) # [B, N-1]
    #     dist_sq = dist_sq.masked_fill(mask_seg, 0.0)
    
    #     # 这里用 Sum 而不是 Mean，为了惩罚长路径
    #     loss_energy = torch.mean(torch.sum(dist_sq, dim=1))

    #     # --- 5. 幅度约束 (Constraints) ---
    #     MAX_VEL = 1.0
    #     MAX_ACC = 0.5
        
    #     # 速度过大
    #     v_mags = torch.norm(pred_vel, dim=-1)
    #     v_excess = torch.relu(v_mags - MAX_VEL)
    #     v_excess = v_excess.masked_fill(mask, 0.0)  # 使用全长 Mask
    #     loss_v_limit = torch.mean(torch.sum(v_excess**2, dim=1) / valid_cnt_seg)  # 分母近似

    #     # 加速度过大
    #     a_mags = torch.norm(acc_vec, dim=-1)
    #     a_excess = torch.relu(a_mags - MAX_ACC)
    #     a_excess = a_excess.masked_fill(mask_seg, 0.0)
    #     loss_a_limit = torch.mean(torch.sum(a_excess**2, dim=1) / valid_cnt_seg)

    #     # mean_dist = torch.mean(dist_sq, dim=1, keepdim=True)
    #     # loss_uniform = torch.mean(torch.var(dist_sq, dim=1))

    #     # --- 6. 组合权重 ---
    #     total_loss = (
    #         3.0 * loss_consistency  # 【关键】必须保证物理自洽
    #         + 0.1 * loss_direction   # 【关键】防止漂移 (非完整约束)
    #         + 5.0 * loss_acc         # 平滑性
    #         # + 0.5 * loss_jerk        # 舒适度
    #         + 0.1 * loss_energy      # 路径长度 (能量)
    #         + 10.0 * loss_v_limit      # 硬约束
    #         + 10.0 * loss_a_limit
    #     )

    #     return total_loss, {
    #         "consistency": loss_consistency,
    #         "direction": loss_direction,
    #         "acc": loss_acc,
    #         "energy": loss_energy,
    #         "jerk": loss_jerk
    #     }
   
   
    def _compute_motion_loss(self, waypoints, mask=None, dt=0.2, odom=None, goal=None):
        """
        Args:
            waypoints: [B, N, 4] 预测的未来点 (P1, P2... PN), 不含起点
            mask: [B, N] (True = 无效/Padding)
            dt: 时间步长
            odom: [B, 6] 机器人的当前状态 (x, y, vx, vy, ax, ay)
            goal: [B, 6] 真实目标点 (gx, gy, gvx, gvy, gax, gay)
        """
        device = waypoints.device
        batch_size = waypoints.shape[0]

        # ==========================================
        # 1. 拼接起点 (Prepend Start State)
        # ==========================================
        # 构造完整序列：[Start, P1, P2, ... PN]
        if odom is not None:
            # 提取 odom 的前4维 (x, y, vx, vy)
            # 形状变换: [B, 6] -> [B, 1, 4]
            start_node = odom[:, 0:4].unsqueeze(1)
            
            # 拼接 waypoints
            full_waypoints = torch.cat([start_node, waypoints], dim=1) # [B, N+1, 4]
            
            # 处理 Mask: 起点永远是有效的 (False)
            if mask is not None:
                start_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
                full_mask = torch.cat([start_mask, mask], dim=1) # [B, N+1]
            else:
                full_mask = torch.zeros((batch_size, full_waypoints.shape[1]), dtype=torch.bool, device=device)
        else:
            full_waypoints = waypoints
            full_mask = mask if mask is not None else torch.zeros((batch_size, waypoints.shape[1]), dtype=torch.bool, device=device)

        # ==========================================
        # 2. 数据解包 (使用拼接后的数据)
        # ==========================================
        pred_pos = full_waypoints[..., 0:2]  # [B, N+1, 2]
        pred_vel = full_waypoints[..., 2:4]  # [B, N+1, 2]
        
        # 各种 Mask 的准备
        # Segment Mask (N段): 包含从 Start->P1 的第一段
        mask_seg = full_mask[:, 1:] 
        valid_cnt_seg = (~mask_seg).sum(dim=1).clamp(min=1.0)

        # ==========================================
        # 3. 计算动力学 Loss (自动包含起点约束)
        # ==========================================
        
        # --- A. 运动学一致性 (Kinematic Consistency) ---
        # 这一步会自动检查 P1 是否也是从 Start 按照 V 积分过来的
        pos_delta = pred_pos[:, 1:] - pred_pos[:, :-1] # [B, N, 2]
        vel_avg = (pred_vel[:, :-1] + pred_vel[:, 1:]) / 2.0
        pos_expected = vel_avg * dt
        
        consistency_error = torch.sum((pos_delta - pos_expected) ** 2, dim=-1)
        consistency_error = consistency_error.masked_fill(mask_seg, 0.0)
        loss_consistency = torch.mean(torch.sum(consistency_error, dim=1) / valid_cnt_seg)

        # --- B. 轨迹方向一致性 (Non-holonomic Constraint) ---
        dir_pos = pos_delta / (torch.norm(pos_delta, dim=-1, keepdim=True) + 1e-6)
        dir_vel = pred_vel[:, :-1] / (torch.norm(pred_vel[:, :-1], dim=-1, keepdim=True) + 1e-6)
        direction_diff = 1.0 - torch.sum(dir_pos * dir_vel, dim=-1)
        direction_diff = direction_diff.masked_fill(mask_seg, 0.0)
        loss_direction = torch.mean(torch.sum(direction_diff, dim=1) / valid_cnt_seg)

        # --- C. 加速度平滑性 (Smoothness) ---
        # 关键点：acc_vec[0] 现在计算的是 (V_1 - V_start) / dt
        # 如果预测的初速度 V_1 和真实初速度 V_start 不匹配，这个 Loss 会很高
        acc_vec = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
        acc_sq = torch.sum(acc_vec ** 2, dim=-1)
        acc_sq = acc_sq.masked_fill(mask_seg, 0.0)
        loss_acc = torch.mean(torch.sum(acc_sq, dim=1) / valid_cnt_seg)
        loss_start_acc_match = torch.tensor(0.0, device=device)
        # 2. 获取 Odom 真实加速度 (Ground Truth)
        if odom is not None:
            start_acc_gt = odom[:, 4:6] # [B, 2] (假设 odom 后两位是 ax, ay)
            start_acc_pred = acc_vec[:, 0] # [B, 2]
            # 3. 计算“起始加速度匹配 Loss”
            # 这会强迫网络生成的第一段轨迹，顺着当前的加速度趋势走，不要突变
            loss_start_acc_match = torch.mean(torch.sum((start_acc_pred - start_acc_gt)**2, dim=1))
        
        # --- D. 【核心修复】直接约束点间距离 (Point Spacing) ---
        # 目标：强制相邻点之间保持最小距离，防止点聚集
        
        # 1. 计算相邻点之间的实际距离 [B, N]
        point_distances = torch.norm(pos_delta, dim=-1)  # [B, N]
        
        # 2. 计算期望的最小距离
        # 【关键修复】使用真实 goal 位置，而不是预测的最后一个waypoint
        # 否则如果预测点都聚集在起点，expected_min_dist 也会变成0，loss就无法工作
        if goal is not None:
            goal_pos = goal[:, :2]  # [B, 2] 真实目标位置
        else:
            # 如果没有传入goal，使用最后一个waypoint（fallback）
            goal_pos = full_waypoints[:, -1, :2]  # [B, 2]
        
        start_pos = odom[:, :2] if odom is not None else full_waypoints[:, 0, :2]  # [B, 2]
        start_to_goal_dist = torch.norm(goal_pos - start_pos, dim=1)  # [B]
        expected_min_dist = start_to_goal_dist / (full_waypoints.shape[1] - 1)  # [B]
        
        # 3. 计算距离不足的惩罚 (只惩罚距离太小的情况)
        # dist_deficit: 当实际距离 < 期望距离时的差值
        dist_deficit = expected_min_dist.unsqueeze(1) - point_distances  # [B, N]
        dist_deficit = torch.relu(dist_deficit)  # 只保留正值（距离不足）
        dist_deficit = dist_deficit.masked_fill(mask_seg, 0.0)
        loss_min_spacing = torch.mean(torch.sum(dist_deficit ** 2, dim=1) / valid_cnt_seg)
        
        # 4. 【额外】惩罚最后一段距离过大（防止最后跳跃）
        # 取最后一段的距离
        last_seg_dist = point_distances[:, -1]  # [B]
        # 如果最后一段距离 > 2倍期望距离，进行惩罚
        last_seg_excess = torch.relu(last_seg_dist - 2.0 * expected_min_dist)
        loss_last_jump = torch.mean(last_seg_excess ** 2)
        
        # --- E. 能量/路径长度 ---
        dist_sq = torch.sum(pos_delta ** 2, dim=-1)
        dist_sq = dist_sq.masked_fill(mask_seg, 0.0)
        loss_energy = torch.mean(torch.sum(dist_sq, dim=1))
        # --- F. 硬约束 (Hard Constraints) ---
        MAX_VEL = 1.0
        MAX_ACC = 0.5
        # 速度限制 (全序列检查，包括起点)
        v_mags = torch.norm(pred_vel, dim=-1)
        v_excess = torch.relu(v_mags - MAX_VEL)
        v_excess = v_excess.masked_fill(full_mask, 0.0)
        valid_cnt_full = (~full_mask).sum(dim=1).clamp(min=1.0)
        loss_v_limit = torch.mean(torch.sum(v_excess**2, dim=1) / valid_cnt_full) 
        # 加速度限制
        a_mags = torch.norm(acc_vec, dim=-1)
        a_excess = torch.relu(a_mags - MAX_ACC)
        a_excess = a_excess.masked_fill(mask_seg, 0.0)
        loss_a_limit = torch.mean(torch.sum(a_excess**2, dim=1) / valid_cnt_seg)
        # ==========================================
        # G. 组合权重
        # ==========================================
        total_loss = (
            2.0 * loss_consistency 
            + 0.1 * loss_direction
            + 0.5 * loss_acc       
            + 1.5 * loss_start_acc_match
            + 20.0 * loss_min_spacing        # 【核心】最小点间距约束
            + 5.0 * loss_last_jump           # 【核心】防止最后跳跃
            # + 0.1 * loss_energy
            + 10.0 * loss_v_limit
            + 10.0 * loss_a_limit
        )
        return total_loss, {
            "consistency": loss_consistency,
            "direction": loss_direction,
            "acc": loss_acc,
            "min_spacing": loss_min_spacing,
            "last_jump": loss_last_jump,
            "energy": loss_energy,
        }
        
    
    def _compute_goal_loss(self, waypoints, goal, mask=None):
        """
        计算全状态终点代价 (Position + Velocity + Heading)
        
        Args:
            waypoints: [B, N, 4] -> (x, y, vx, vy)
            goal: [B, 6] -> (gx, gy, gvx, gvy, gax, gay)
            mask: [B, N] (True = 无效/Padding)
        """
        batch_size = waypoints.shape[0]
        device = waypoints.device
        
        # --- 1. 提取每个 Batch 的“最后一个有效点” ---
        if mask is not None:
            # 计算有效长度
            valid_len = (~mask).sum(dim=1).long()
            # 最后一个有效点的索引 = 长度 - 1
            last_idx = (valid_len - 1).clamp(min=0)
        else:
            last_idx = torch.full((batch_size,), waypoints.shape[1]-1, device=device, dtype=torch.long)
            
        # 使用 torch.gather 或高级索引提取终点状态
        # waypoints[B, N, 4] -> end_state[B, 4]
        # range(batch_size) 生成 [0, 1, ... B-1]
        batch_indices = torch.arange(batch_size, device=device)
        
        end_pos = waypoints[batch_indices, last_idx, 0:2]  # [B, 2] (x, y)
        end_vel = waypoints[batch_indices, last_idx, 2:4]  # [B, 2] (vx, vy)
        
        # 提取目标状态
        goal_pos = goal[:, 0:2]
        goal_vel = goal[:, 2:4]
        # 如果需要加速度约束：
        # goal_acc = goal[:, 4:6] 
        # end_acc = (end_vel - prev_vel) / dt  (需要额外计算，此处暂略)

        # --- 2. 位置代价 (Position Cost) ---
        # 保持原有的 Log 形式，对长尾分布更鲁棒
        dist_diff = torch.norm(end_pos - goal_pos, p=2, dim=1)
        loss_pos = torch.mean(torch.log(dist_diff + 1.0))

        # --- 3. 速度矢量代价 (Velocity Vector Cost) ---
        # 这是一个“二合一”的代价：同时约束了速度大小和朝向
        # 如果你的目标是让机器人停在终点 (goal_vel = 0)，这个 Loss 最有效
        vel_diff = torch.norm(end_vel - goal_vel, p=2, dim=1)
        loss_vel_vec = torch.mean(vel_diff) # 直接用 L2 距离

        # --- 4. [进阶] 分离的速度大小与朝向代价 (Speed & Heading) ---
        # 如果你需要更细粒度的控制（例如：必须到达终点且朝向正确，但速度稍微有点误差没关系）
        # 可以使用这一部分代替上面的 loss_vel_vec
        
        # # A. 速度大小误差 (Speed Magnitude Error)
        # pred_speed = torch.norm(end_vel, p=2, dim=1)
        # target_speed = torch.norm(goal_vel, p=2, dim=1)
        # loss_speed = torch.mean(torch.abs(pred_speed - target_speed))
        
        # # B. 朝向误差 (Heading/Cosine Error)
        # # 注意：只有当目标速度 > 0 时，朝向才有意义。如果目标是停车，朝向通常不重要（或者由单独的 yaw 给出）
        # # 为了数值稳定性，加 eps
        # pred_dir = end_vel / (pred_speed.unsqueeze(1) + 1e-6)
        # target_dir = goal_vel / (target_speed.unsqueeze(1) + 1e-6)
        
        # # Cosine Similarity: 1.0 = 同向, -1.0 = 反向
        # # Loss = 1 - cos
        # cosine_sim = torch.sum(pred_dir * target_dir, dim=1)
        
        # # 创建一个 Mask：只有当目标要求移动时 (target_speed > 0.1)，才强行约束朝向
        # # 如果目标是停车，我们就不关心它停下来时车头朝哪（除非你有专门的 yaw 输入）
        # moving_mask = (target_speed > 0.05).float()
        
        # loss_heading = 1.0 - cosine_sim
        # # 只计算需要移动的样本
        # loss_heading = torch.sum(loss_heading * moving_mask) / (moving_mask.sum() + 1e-6)

        # --- 5. 组合 Loss ---
        # 权重建议：
        # Pos: 最重要，决定是否到达
        # Vel: 重要，决定到达时的状态（是冲过去还是停下来）
        # Heading: 辅助，决定姿态
        
        total_goal_loss = (
            2.0 * loss_pos + 
            1.0 * loss_vel_vec   # 简单方案：直接用矢量差
            # 1.0 * loss_speed + 0.5 * loss_heading # 进阶方案：分离控制
        )
        
        return total_goal_loss, {"pos": loss_pos, "vel": loss_vel_vec}
    
    def _compute_fear_loss(self, waypoints, oloss_M, fear_pred, batch_size, ahead_dist=2.0, mask=None):
        """
        计算碰撞恐惧损失 (Fear Loss)
        Args:
            waypoints:  [Batch, N, 4] 路径点
            oloss_M:   [3 * Batch, N-1] 膨胀后的障碍物代价矩阵
            fear_pred: [Batch, 1] 网络预测的恐惧值
            batch_size: int
            ahead_dist: float 预瞄距离
            mask:      [Batch, N] (Optional) True表示无效点/Padding

        Returns:
            loss: scalar
        """
        # 1. 计算相邻点间距 (Step Distance)
        # 取前两维 (x, y) 计算距离
        # diff: [Batch, N-1, 2]
        pos_diff = waypoints[:, 1:, :2] - waypoints[:, :-1, :2]
        wp_ds = torch.norm(pos_diff, dim=-1) # [Batch, N-1]

        # 如果提供了 Mask，需要把无效段的距离设为 0
        # 否则 cumsum 会把乱飞的 Padding 点的距离累加进去，导致距离计算错误
        if mask is not None:
            # mask 是点的 mask (N)，wp_ds 是段 (N-1)
            # 只要段的终点无效，该段距离就视为 0
            seg_mask = mask[:, 1:] 
            wp_ds = wp_ds.masked_fill(seg_mask, 0.0)

        # 2. 计算路径累积距离 (Cumulative Distance)
        # goal_dists: [Batch, N-1]
        goal_dists = torch.cumsum(wp_ds, dim=1)
        
        # 3. 堆叠距离以匹配 oloss_M 的形状 (Robot Width Expansion)
        # oloss_M 形状通常是 [3*B, N-1] (Left, Center, Right)
        # 所以我们需要把距离矩阵也复制 3 份堆叠起来
        goal_dists_stacked = torch.vstack([goal_dists] * 3) # [3*B, N-1]
        
        # 4. 创建临时 Cost Map
        floss_M = oloss_M.clone()
        
        # 5. 距离遮罩：忽略超过预瞄距离后的障碍物
        # 逻辑：太远的障碍物还没到面前，不需要产生恐惧
        floss_M[goal_dists_stacked > ahead_dist] = 0.0
        
        # 6. 生成标签 (Label Generation)
        # A. 寻找每条轨迹上的最大障碍物代价
        # max_vals: [3*B, 1]
        max_vals, _ = torch.max(floss_M, dim=1, keepdim=True)
        
        # B. 阈值判断：超过阈值即视为危险
        is_collision = max_vals > self.obstalce_thread
        
        # C. 聚合三个分支 (Left/Center/Right)
        # 只要任意一边碰到障碍物，整条轨迹就是危险的 (Label=1)
        # view(3, B) -> any(dim=0) -> [B]
        fear_labels = is_collision.view(3, batch_size, -1).any(dim=0).float() # [B, 1]
        
        # 7. 计算 Loss
        if fear_pred.shape != fear_labels.shape:
            fear_pred = fear_pred.view_as(fear_labels)
            
        loss = nn.BCELoss()(fear_pred, fear_labels)
        
        return loss
    
    
    def _compute_quintic_spacing_loss(self, waypoints, odom, goal, total_time=5.0):
        """
        计算五次样条间距损失 (Quintic Spacing Loss)。
        
        该 Loss 不仅约束路径长度，还强迫预测点的分布符合 Minimum Jerk 动力学特性
        （例如：起步和停车时点密集，中间快时点稀疏），而不是简单的均匀分布。

        Args:
            waypoints: [Batch, N, 4] 实际预测并生成好的轨迹点 (包含 x, y, vx, vy)
            odom:      [Batch, 6] 起点状态 (x, y, vx, vy, ax, ay)
            goal:      [Batch, 6] 终点状态 (x, y, vx, vy, ax, ay)
            total_time: float, 理想参考轨迹的总时间 T。
                        这个值主要影响速度数值，但只要不为0，对点的相对疏密分布影响很小。
                        建议设为你的预测时域长度（如 2.0s 或 5.0s）。
        
        Returns:
            loss: scalar
        """
        # 1. 准备数据
        batch_size, num_p, _ = waypoints.shape
        # device = waypoints.device
        # 确保使用相同的 CubicSplineTorch 类（或直接通过 self.opt.cs_interp 调用）
        # 这里假设 self.opt.cs_interp 是 CubicSplineTorch 的实例
        # 也可以直接调用 CubicSplineTorch.generate_quintic_path (因为它是 @staticmethod)
        
        # 提取起点状态 (Start State)
        start_pos = odom[:, 0:2]
        start_vel = odom[:, 2:4]
        start_acc = odom[:, 4:6]

        # 提取终点状态 (Goal State)
        end_pos = goal[:, 0:2]
        end_vel = goal[:, 2:4]
        end_acc = goal[:, 4:6]

        # 2. 生成理想的五次样条参考轨迹 (Ideal Quintic Trajectory)
        # 这是一条从 Start 到 Goal 的“完美”轨迹（通常是平滑的曲线或直线）
        # 我们生成与 waypoints 相同数量的点 (num_p)，以便一一对应比较间距
        ideal_pos, _, _ = self.opt.cs_interp.generate_quintic_path(
            start_pos, start_vel, start_acc,
            end_pos, end_vel, end_acc,
            T=total_time,        # 设定一个总时间
            num_points=num_p     # 关键：点数必须与 waypoints 一致
        )
        # print(ideal_pos.shape)
        # print("--------------------")
        # print(waypoints[..., 0:2].shape)
        # 3. 计算“理想间距” (Desired Spacing)
        # 计算参考轨迹上相邻点之间的距离
        # ideal_diff: [Batch, N-1, 2]
        ideal_diff = ideal_pos[:, 1:, :] - ideal_pos[:, :-1, :]
        desired_ds = torch.norm(ideal_diff, dim=2)  # [Batch, N-1]

        # 4. 计算“实际预测间距” (Actual Spacing)
        # 计算你网络预测轨迹上相邻点之间的距离
        # waypoints 前两维通常是位置 (x, y)
        pred_pos = waypoints[..., 0:2]
        pred_diff = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        wp_ds = torch.norm(pred_diff, dim=2)  # [Batch, N-1]

        # 5. 计算累积路径长度 (Cumulative Path Length)
        # 这是关键：我们比较的是"到达第i个点时的累积距离"，而不是绝对位置
        desired_cumsum = torch.cumsum(desired_ds, dim=1)  # [Batch, N-1]
        actual_cumsum = torch.cumsum(wp_ds, dim=1)        # [Batch, N-1]
        
        # 6. 累积长度损失 (Cumulative Length Loss)
        # 这会确保：
        # - 第1个点距离起点 ~0.5m（而不是0.01m）
        # - 第2个点距离起点 ~1.5m（而不是0.02m）
        # - 最后一个点距离起点 ~总长度（而不是3.5m跳跃）
        # 但具体在哪个位置（绕障碍）是自由的！
        cumsum_loss = torch.abs(desired_cumsum - actual_cumsum)
        cumsum_loss = torch.mean(cumsum_loss)
        
        # 7. 【可选】间距平滑性损失 (Spacing Smoothness)
        # 防止相邻间距变化过于剧烈（从0.01突然跳到3.5）
        # 计算间距的二阶差分
        ds_diff = wp_ds[:, 1:] - wp_ds[:, :-1]  # [Batch, N-2]
        smoothness_loss = torch.mean(torch.abs(ds_diff))
        
        # 8. 综合损失
        # 权重说明：
        # - cumsum_loss (1.0): 主要约束，确保累积距离正确
        # - smoothness_loss (0.3): 辅助约束，防止间距突变
        total_loss = (
            1.0 * cumsum_loss +      # 确保路径长度分布合理
            0.3 * smoothness_loss    # 防止间距突变（0.01->3.5）
        )

        return total_loss, {
            "cumsum_loss": cumsum_loss,
            "smoothness_loss": smoothness_loss,
            "total_spacing_loss": total_loss
        }
    
    # 计算轨迹总代价 CostofTraj (核心函数)
    def CostofTraj(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        step: float = 0.2,
        dataset: str = "train",
        mask: Optional[torch.Tensor] = None,  # <--- 新增 mask 参数 [B, K] (True表示无效)
    ):
        batch_size, num_p, _ = waypoints.shape
        assert self.is_map, "Map has to be set for cost calculation"

        # 1. 计算原始 Obstacle Loss 矩阵 [B, K]
        # 你这一步是不是也要引入 mask？
        oloss_M = self._compute_oloss(waypoints, batch_size)
        
        # === 修正 A: 应用 Mask 到 oloss ===
        if mask is not None:
            # oloss_M 的长度是 N-1，而 mask 的长度是 N
            # 我们必须对 mask 进行切片以匹配 oloss_M
            # 策略: 如果点 i+1 无效，则认为从 i 到 i+1 的段（oloss_M[i]）无效
            seg_mask = mask[:, 1:]  # [B, N-1]
            
            # 使用切片后的 mask 进行填充
            oloss_M = oloss_M.masked_fill(seg_mask, 0.0)
            
            # 它统计了当前这条轨迹中，实际上有多少个有效的线段（Steps）
            valid_counts = (~seg_mask).sum(dim=1) + 1e-6
            # 注意：这里的 num_p 是 N，但 oloss_M 是 N-1。
            # 为了保持量级，乘以 (num_p - 1) 可能更准确，或者保持 num_p 也行，差别不大
            oloss = torch.mean(torch.sum(oloss_M, axis=1) / valid_counts * (num_p - 1))
        else:
            oloss = torch.mean(torch.sum(oloss_M, axis=1))

        # 2. 计算 Motion Loss (返回 loss 和 dict)
        mloss, mloss_dict = self._compute_motion_loss(waypoints, mask, dt=step, odom=odom, goal=goal)

        # 3. 计算 Goal Loss (返回 loss 和 dict)
        gloss, gloss_dict = self._compute_goal_loss(waypoints, goal, mask)
        
        # disloss, _ = self._compute_quintic_spacing_loss(waypoints, odom, goal, total_time=step * (num_p - 1))
        
        trajectory_loss = (
            self.w_obs * oloss + 
            self.w_motion * mloss + 
            self.w_goal * gloss
            # +
            # 80 * disloss
        )
        
        # Complete Trajectory Loss
        # trajectory_loss = self.w_obs * oloss + self.w_motion * self._compute_motion_loss(waypoints, mask, step) + self.w_goal * self._compute_goal_loss(waypoints, goal, mask)
            
        collision_probabilty_loss = self._compute_fear_loss(
            waypoints=waypoints,
            oloss_M=oloss_M,     # 传入膨胀后的 oloss
            fear_pred=fear,
            batch_size=batch_size,
            ahead_dist=ahead_dist,
            mask=mask            # 传入 Mask 以确保距离计算准确
        )
        # print(f"Collision Probabilty Loss: {collision_probabilty_loss:.4f}")
        # TODO: kinodynamics cost
        # return collision_probabilty_loss + trajectory_loss
        return trajectory_loss

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
