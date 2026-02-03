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
from traj_cost_opt.traj_opt_myself_cubic import TrajOpt

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
        
    
    def _compute_goal_loss(self, waypoints, goal, mask=None):
        """
        计算全状态终点代价 (Position + Velocity + Heading)
        
        Args:
            waypoints: [B, N, 2] -> (x, y)
            goal: [B, 2] -> (gx, gy)
            mask: [B, N] (True = 无效/Padding)
        """
        # 添加断言，确保输入维度正确
        assert waypoints.shape[-1] == 2, f"waypoints last dim must be == 2, got {waypoints.shape[-1]}"
        assert goal.shape[-1] == 2, f"goal last dim must be == 2, got {goal.shape[-1]}"
        
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
        
        end_pos = waypoints[batch_indices, last_idx, 0:2]  # [B, 2]
        
        # 提取目标状态
        goal_pos = goal[:, 0:2]

        # --- 2. 终点位置代价 ---
        dist_diff = torch.norm(end_pos - goal_pos, p=2, dim=1)
        loss_endpoint = torch.mean(torch.log(dist_diff + 1.0))
        
        # --- 3. 新增: 所有有效点到目标的平均距离 ---
        # 防止网络用mask"作弊"，只让最后一个点到达目标
        goal_expanded = goal_pos.unsqueeze(1)  # [B, 1, 2]
        dists_to_goal = torch.norm(waypoints - goal_expanded, p=2, dim=2)  # [B, N]
        
        if mask is not None:
            dists_to_goal = dists_to_goal.masked_fill(mask, 0.0)
            valid_counts = (~mask).sum(dim=1).float() + 1e-6
            avg_dist = torch.sum(dists_to_goal, dim=1) / valid_counts
        else:
            avg_dist = torch.mean(dists_to_goal, dim=1)
        
        loss_avg_dist = torch.mean(torch.log(avg_dist + 1.0))
        
        # --- 4. 新增: 方向一致性损失 - 惩罚远离目标的运动 ---
        # 计算每个路径点到目标的向量
        vec_to_goal = goal_expanded - waypoints  # [B, N, 2]
        # 计算路径段向量 (运动方向)
        motion_vec = waypoints[:, 1:, :] - waypoints[:, :-1, :]  # [B, N-1, 2]
        # 计算对应位置到目标的向量 (使用起点)
        vec_to_goal_seg = vec_to_goal[:, :-1, :]  # [B, N-1, 2]
        
        # 计算余弦相似度: cos(θ) = (a·b) / (|a||b|)
        dot_product = torch.sum(motion_vec * vec_to_goal_seg, dim=2)  # [B, N-1]
        norm_motion = torch.norm(motion_vec, dim=2) + 1e-6  # [B, N-1]
        norm_goal = torch.norm(vec_to_goal_seg, dim=2) + 1e-6  # [B, N-1]
        cos_similarity = dot_product / (norm_motion * norm_goal)  # [B, N-1], 范围[-1,1]
        
        # 我们希望cos接近+1 (同向), 惩罚<0 (反向)的情况
        # 使用 (1 - cos) 作为损失, 范围[0,2], 同向时为0, 反向时为2
        direction_loss_raw = 1.0 - cos_similarity  # [B, N-1]
        
        if mask is not None:
            seg_mask = mask[:, 1:]
            direction_loss_raw = direction_loss_raw.masked_fill(seg_mask, 0.0)
            valid_counts_seg = (~seg_mask).sum(dim=1).float() + 1e-6
            loss_direction = torch.mean(torch.sum(direction_loss_raw, dim=1) / valid_counts_seg)
        else:
            loss_direction = torch.mean(direction_loss_raw)
        
        # --- 5. 综合损失: 终点 + 平均距离 + 方向一致性 ---
        loss_pos = 0.3 * loss_endpoint + 0.3 * loss_avg_dist + 0.4 * loss_direction
        
        return loss_pos, {
            "pos": loss_pos, 
            "endpoint": loss_endpoint.item(),
            "avg": loss_avg_dist.item(),
            "direction": loss_direction.item()
        }
    
    def _compute_fear_loss(self, waypoints, oloss_M, fear_pred, batch_size, ahead_dist=2.0, mask=None):
        """
        计算碰撞恐惧损失 (Fear Loss)
        Args:
            waypoints:  [Batch, N, 2] 路径点
            oloss_M:   [3 * Batch, N-1] 膨胀后的障碍物代价矩阵
            fear_pred: [Batch, 1] 网络预测的恐惧值
            batch_size: int
            ahead_dist: float 预瞄距离
            mask:      [Batch, N] (Optional) True表示无效点/Padding

        Returns:
            loss: scalar
        """
        assert waypoints.shape[-1] == 2, f"waypoints last dim must be == 2, got {waypoints.shape[-1]}"
        # 1. 计算每个路径点到起点的累积距离
        pos_diff = waypoints[:, 1:, :] - waypoints[:, :-1, :]
        wp_ds = torch.norm(pos_diff, dim=-1)  # [Batch, N-1]

        if mask is not None:
            seg_mask = mask[:, 1:]
            wp_ds = wp_ds.masked_fill(seg_mask, 0.0)

        goal_dists = torch.cumsum(wp_ds, dim=1)
        
        goal_dists_stacked = torch.vstack([goal_dists] * 3)
        
        floss_M = oloss_M.clone()
        
        floss_M[goal_dists_stacked > ahead_dist] = 0.0
        
        max_vals, _ = torch.max(floss_M, dim=1, keepdim=True)
        
        # B. 阈值判断：超过阈值即视为危险
        is_collision = max_vals > self.obstalce_thread
        
        fear_labels = is_collision.view(3, batch_size, -1).any(dim=0).float()  # [B, 1]
        
        # 7. 计算 Loss
        if fear_pred.shape != fear_labels.shape:
            fear_pred = fear_pred.view_as(fear_labels)
            
        loss = nn.BCELoss()(fear_pred, fear_labels)
        
        return loss
    
    def _compute_motion_loss(self, waypoints: torch.Tensor, mask: Optional[torch.Tensor], dt: float, goal: torch.Tensor, num_keypoints: int = 32):
        """
            计算运动平滑损失 (Motion Smoothness Loss)
            
            关键修正: 参考轨迹应该使用 keypoints 数量(32)，而不是密集插值数量(321)
            - 从(0,0)到goal生成 num_keypoints 个均匀参考点
            - 对密集轨迹进行采样，取出对应 keypoints 的位置
            - 比较采样后的实际路径和参考路径的间距差异
            
            Args:
                waypoints: [Batch, N, 2] 插值后的密集路径点 (N=321)
                mask: [Batch, N] (Optional) True表示无效点/Padding
                dt: float 时间步长
                goal: [Batch, 2] 终点位置
                num_keypoints: int 关键点数量 (默认32)
            
            Returns:
                loss: scalar
        """
        batch_size, num_dense, _ = waypoints.shape
        
        # 生成参考轨迹: 从(0,0)到goal的 num_keypoints 个均匀点
        step_size = 1.0 / (num_keypoints - 1)
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:2], step=step_size)  # [B, num_keypoints, 2]
        
        # 从密集轨迹中采样出对应 keypoints 的位置
        # 计算采样索引: 均匀分布在 [0, num_dense-1]
        sample_indices = torch.linspace(0, num_dense - 1, num_keypoints, device=waypoints.device).long()  # [num_keypoints]
        sampled_waypoints = waypoints[:, sample_indices, :]  # [B, num_keypoints, 2]
        
        # 计算参考间距
        desired_ds = torch.norm(desired_wp[:, 1:, :] - desired_wp[:, :-1, :], dim=2)  # [B, num_keypoints-1]
        
        # 计算实际间距 (基于采样的 keypoints)
        wp_ds = torch.norm(sampled_waypoints[:, 1:, :] - sampled_waypoints[:, :-1, :], dim=2)  # [B, num_keypoints-1]
        
        # 应用mask (注意：这里的mask是针对密集轨迹的，需要采样)
        if mask is not None:
            sampled_mask = mask[:, sample_indices]  # [B, num_keypoints]
            seg_mask = sampled_mask[:, 1:]  # [B, num_keypoints-1]
            mloss = torch.abs(desired_ds - wp_ds)
            mloss = mloss.masked_fill(seg_mask, 0.0)
            valid_counts = (~seg_mask).sum(dim=1).float() + 1e-6
            mloss = torch.sum(mloss, dim=1) / valid_counts
            mloss = torch.mean(mloss)
        else:
            mloss = torch.abs(desired_ds - wp_ds)
            mloss = torch.sum(mloss, axis=1)
            mloss = torch.mean(mloss)
        
        return mloss
    
    # 计算轨迹总代价 CostofTraj (核心函数)
    def CostofTraj(
        self,
        waypoints: torch.Tensor,  # [B, N, 2] 插值后的密集轨迹
        goal: torch.Tensor,
        fear: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        step: float = 0.2,
        dataset: str = "train",
        mask: Optional[torch.Tensor] = None,  # [B, N] 密集轨迹的mask
        num_keypoints: int = 32,  # 关键点数量，用于motion loss计算
    ):
        
        # Use mask to keep only valid waypoints up to the max valid length across the batch
        if mask is not None:
            valid_len = (~mask).sum(dim=1).long()
            max_valid = int(torch.clamp(valid_len.max(), min=2).item())
            waypoints = waypoints[:, :max_valid, :]
            mask = mask[:, :max_valid]
        
        return self.CostofTrajVI(waypoints, None, goal, fear, log_step, ahead_dist, dataset)

    def CostofTrajVI(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        dataset: str = "train",
    ):
        batch_size, num_p, _ = waypoints.shape

        assert self.is_map, "Map has to be set for cost calculation"
        # world_ps = self.TransformPoints(odom, waypoints).tensor()

        # Obstacle loss
        oloss_M = self._compute_oloss(waypoints, batch_size)
        oloss = torch.mean(torch.sum(oloss_M, axis=1))

        # Goal Cost - Control Cost
        gloss_M = torch.norm(goal[:, :2] - waypoints[:, -1, :2], dim=1)
        # gloss = torch.mean(gloss_M)
        gloss = torch.mean(torch.log(gloss_M + 1.0))

        # Moving Loss - punish staying
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:2], step=1.0 / (num_p - 1))
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0 : num_p - 1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0 : num_p - 1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)

        # Complete Trajectory Loss + self.w_motion * mloss
        # 
        trajectory_loss = self.w_obs * oloss + self.w_goal * gloss + self.w_motion * mloss

        # Fear labels
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        goal_dists = torch.vstack([goal_dists] * 3)
        floss_M = torch.clone(oloss_M)
        floss_M[goal_dists > ahead_dist] = 0.0
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thread)
        fear_labels = fear_labels > self.obstalce_thread
        fear_labels = torch.any(fear_labels.reshape(3, batch_size).T, dim=1, keepdim=True).to(torch.float32)
        # Fear loss
        collision_probabilty_loss = nn.BCELoss()(fear, fear_labels.float())

        # log
        if self.log_data:
            try:
                wandb.log(
                    {f"Obstacle Loss {dataset}": self.w_obs * oloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Goal Loss {dataset}": self.w_goal * gloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Motion Loss {dataset}": self.w_motion * mloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Trajectory Loss {dataset}": trajectory_loss},
                    step=log_step,
                )
                wandb.log(
                    {f"Collision Loss {dataset}": collision_probabilty_loss},
                    step=log_step,
                )
            except:  # noqa: E722
                print("wandb log failed")

        # TODO: kinodynamics cost
        return trajectory_loss
        # return collision_probabilty_loss + trajectory_loss

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
