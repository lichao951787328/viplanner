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

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from omni.viplanner.cost_maps import CostMapPCD  # 处理点云地图或栅格地图的类

# visual-imperative-planning
from .traj_opt import TrajOpt

try:
    import pypose as pp  # only used for training 用于处理机器人位姿变换（SE3）
    import wandb  # only used for training 用于训练时的日志记录
except ModuleNotFoundError or ImportError:  # eval in issac sim  # TODO: check if all can be installed in Isaac Sim
    print("[Warning] pypose or wandb not found, only use for evaluation")


class TrajCost:
    debug = False

    def __init__(
        self,
        gpu_id: Optional[int] = 0,
        log_data: bool = False,
        w_obs: float = 0.25,
        w_height: float = 1.0,
        w_motion: float = 1.5,
        w_goal: float = 2.0,
        obstalce_thread: float = 0.75,
        robot_width: float = 0.6,
        robot_max_moving_distance: float = 0.15,
    ) -> None:
        # init map and optimizer
        self.gpu_id = gpu_id
        self.cost_map: CostMapPCD = None
        self.opt = TrajOpt()
        self.is_map = False
        self.neg_reward: torch.Tensor = None

        # loss weights
        self.w_obs = w_obs  # 障碍物代价权重
        self.w_height = w_height  # 地形高度代价权重
        self.w_motion = w_motion  # 运动平滑代价权重
        self.w_goal = w_goal  # 目标距离代价权重

        # fear label threshold value
        self.obstalce_thread = obstalce_thread

        # footprint radius
        self.robot_width = robot_width
        self.robot_max_moving_distance = robot_max_moving_distance

        # logging
        self.log_data = log_data
        return

    @staticmethod
    def TransformPoints(odom, points):
        # points: [batch_size, num_points, 3] (通常在机器人局部坐标系)
        # odom: [batch_size, 7] (机器人的世界坐标系位姿，位置+四元数)
        batch_size, num_p, _ = points.shape
        # 创建局部坐标系的 SE3 对象
        world_ps = pp.identity_SE3(
            batch_size,
            num_p,
            device=points.device,
            requires_grad=points.requires_grad,
        )
        # 核心变换：T_world = T_odom * T_local
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps

    # 加载 TSDF（截断符号距离场）地图或语义地图，后续用于查询障碍物信息。
    def SetMap(self, root_path, map_name):
        self.cost_map = CostMapPCD.ReadTSDFMap(root_path, map_name, self.gpu_id)
        self.is_map = True

        # get negative reward of cost-map
        self.neg_reward = torch.zeros(7, device=self.cost_map.device)
        if self.cost_map.cfg.semantics:
            self.neg_reward[2] = self.cost_map.cfg.sem_cost_map.negative_reward

        return

    

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
    ):
        batch_size, num_p, _ = waypoints.shape

        assert self.is_map, "Map has to be set for cost calculation"
        # 将路径点转换到世界坐标系
        world_ps = self.TransformPoints(odom, waypoints).tensor()

        # Obstacle loss # 调用内部函数 _compute_oloss 计算每个点的碰撞风险
        oloss_M = self._compute_oloss(world_ps, batch_size)
        oloss = torch.mean(torch.sum(oloss_M, axis=1))

        # Terrian Height loss # 将世界坐标转换为地图索引 (u, v)
        norm_inds, _ = self.cost_map.Pos2Ind(world_ps)
        # 获取高度图数据
        height_grid = self.cost_map.ground_array.T.expand(batch_size, 1, -1, -1)
        # 采样地图：查询路径点下方的地形高度
        hloss_M = (
            F.grid_sample(
                height_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # 计算路径点 Z 轴与地形高度 Z 轴的差异
        # 惩罚机器人“飞”在空中或钻入地下的情况
        hloss_M = torch.abs(world_ps[:, :, 2] - odom[:, None, 2] - hloss_M).to(
            torch.float32
        )  # world_ps - odom to have them on the ground to be comparable to the height map
        hloss_M = torch.sum(hloss_M, axis=1)
        hloss = torch.mean(hloss_M)

        # Goal Cost - Control Cost # 计算最后一个路径点与实际目标点的欧氏距离
        gloss_M = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        # gloss = torch.mean(gloss_M) # 使用 Log 形式可以对远距离不那么敏感，更关注近距离的收敛
        gloss = torch.mean(torch.log(gloss_M + 1.0))

        # Moving Loss - punish staying
        # 生成理想的等间距插值点
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0 / (num_p - 1))
        # 计算理想间距
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0 : num_p - 1, :], dim=2)
        # 计算实际预测路径点的间距
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0 : num_p - 1, :], dim=2)
        # 惩罚间距不均匀或静止不动的情况（防止网络输出一堆重叠的点）
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)

        # Complete Trajectory Loss
        trajectory_loss = self.w_obs * oloss + self.w_height * hloss + self.w_motion * mloss + self.w_goal * gloss

        # Fear labels # 计算路径累积距离
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        goal_dists = torch.vstack([goal_dists] * 3)
        floss_M = torch.clone(oloss_M)
        # 忽略超过一定预瞄距离(ahead_dist)后的障碍物影响
        floss_M[goal_dists > ahead_dist] = 0.0
        # 如果路径上某点的障碍物值超过阈值，则标记为“危险” (Label=1)
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thread) 
        fear_labels = fear_labels > self.obstalce_thread + self.neg_reward[2]
        fear_labels = torch.any(fear_labels.reshape(3, batch_size).T, dim=1, keepdim=True).to(torch.float32)
        # Fear loss # 计算二元交叉熵损失 (BCE Loss)，训练网络预测“恐惧”
        collision_probabilty_loss = nn.BCELoss()(fear, fear_labels.float())

        # log
        if self.log_data:
            try:
                wandb.log(
                    {f"Height Loss {dataset}": self.w_height * hloss},
                    step=log_step,
                )
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
        norm_inds, cost_idx = self.cost_map.Pos2Ind(world_ps_inflated)

        # Obstacle Cost
        # 采样 Cost Map
        # 任何落在障碍物区域的点都会采样到高 Cost
        cost_grid = self.cost_map.cost_array.T.expand(world_ps_inflated.shape[0], 1, -1, -1)
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
        oloss_M = oloss_M.to(torch.float32)

        if self.debug:
            # add negative reward for cost-map
            world_ps_inflated = world_ps_inflated + self.neg_reward

            import numpy as np

            # indexes in the cost map
            start_xy = torch.tensor(
                [self.cost_map.cfg.x_start, self.cost_map.cfg.y_start],
                dtype=torch.float64,
                device=world_ps_inflated.device,
            ).expand(1, 1, -1)
            H = (world_ps_inflated[:, :, 0:2] - start_xy) / self.cost_map.cfg.general.resolution
            cost_values = self.cost_map.cost_array[
                H[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy().astype(np.int64),
                H[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy().astype(np.int64),
            ]

            import matplotlib.pyplot as plt

            _, (ax1, ax2, ax3) = plt.subplots(1, 3)
            sc1 = ax1.scatter(
                world_ps_inflated[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy(),
                world_ps_inflated[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy(),
                c=oloss_M[[0, batch_size, batch_size * 2]].reshape(-1).detach().cpu().numpy(),
                cmap="rainbow",
                vmin=0,
                vmax=torch.max(cost_grid).item(),
            )
            ax1.set_aspect("equal", adjustable="box")
            ax2.scatter(
                H[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy(),
                H[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy(),
                c=cost_values.cpu().numpy(),
                cmap="rainbow",
                vmin=0,
                vmax=torch.max(cost_grid).item(),
            )
            ax2.set_aspect("equal", adjustable="box")
            cost_array = self.cost_map.cost_array.cpu().numpy()
            max_cost = torch.max(self.cost_map.cost_array).item()
            scale_factor = [1.4, 1.8]
            for idx, run_idx in enumerate([0, batch_size, batch_size * 2]):
                _, cost_idx = self.cost_map.Pos2Ind(world_ps_inflated[run_idx, :, :].unsqueeze(0))
                cost_array[
                    cost_idx.to(torch.int32).cpu().numpy()[:, 0],
                    cost_idx.to(torch.int32).cpu().numpy()[:, 1],
                ] = (
                    max_cost * scale_factor[idx]
                )
            ax3.imshow(cost_array)

            plt.figure()
            plt.title("cost_map")
            plt.imshow(cost_array)

            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                world_ps_inflated[[0, batch_size, batch_size * 2], :, :3].reshape(-1, 3).detach().cpu().numpy()
            )
            pcd.colors = o3d.utility.Vector3dVector(
                sc1.to_rgba(oloss_M[[0, batch_size, batch_size * 2]].reshape(-1).detach().cpu().numpy())[:, :3]
            )
            # pcd.colors = o3d.utility.Vector3dVector(sc2.to_rgba(cost_values[0].cpu().numpy())[:, :3])
            o3d.visualization.draw_geometries([self.cost_map.pcd_tsdf, pcd])

        return oloss_M


# EoF
