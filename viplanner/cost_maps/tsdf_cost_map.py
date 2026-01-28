# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# python
import os

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# imperative-cost-map
from omni.viplanner.config import GeneralCostMapConfig, TsdfCostMapConfig


class TsdfCostMap:
    """
    Cost Map based on geometric information
    """

    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        # set init flag
        self.is_map_ready = False
        # init point clouds
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        return

    # Numpy -> Open3D:
    # o3d.utility.Vector3dVector(P_obs) 将原始的 Numpy 数组转换为 Open3D 内部的点云格式，以便调用算法。
    # 处理:
    # 执行 voxel_down_sample。
    # Open3D -> Numpy:
    # np.asarray(self.obs_pcd.points) 将处理完的点云又转回 Numpy 数组，以便后续进行矩阵运算或存入数组。
    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd = self.obs_pcd.voxel_down_sample(self._cfg_general.resolution)
            self.free_pcd = self.free_pcd.voxel_down_sample(self._cfg_general.resolution * 0.85)

        self.obs_points = np.asarray(self.obs_pcd.points)
        self.free_points = np.asarray(self.free_pcd.points)
        print("number of obs points: %d, free points: %d" % (self.obs_points.shape[0], self.free_points.shape[0]))

    def ReadPointFromFile(self):
        pcd_load = o3d.io.read_point_cloud(os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file))
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return

    # 假设输入的是一个和全局地图对齐的点云，仅仅通过z值进行简单分割
    def TerrainAnalysis(self, input_points):
        obs_points = np.zeros(input_points.shape)
        free_poins = np.zeros(input_points.shape)
        obs_idx = 0
        free_idx = 0
        # naive approach with z values
        for p in input_points:
            p_height = p[2] + self._cfg_tsdf.offset_z
            if (p_height > self._cfg_tsdf.ground_height * 1.2) and (
                p_height < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor
            ):  # remove ground and ceiling
                obs_points[obs_idx, :] = p
                obs_idx = obs_idx + 1
            elif p_height < self._cfg_tsdf.ground_height and p_height > -self._cfg_tsdf.ground_height:
                free_poins[free_idx, :] = p
                free_idx = free_idx + 1
        return obs_points[:obs_idx, :], free_poins[:free_idx, :]

    # 根据障碍物点云的分布范围，自动计算出 2D 代价地图（Grid Map）的尺寸和原点位置。
    def UpdateMapParams(self):
        if self.obs_points.shape[0] == 0:
            print("No points received.")
            return
        # p.amax/amin: 找到所有障碍物点在 X 和 Y 轴上的最大值和最小值，这确定了点云的紧包围盒。
        # +/- clear_dist: 在紧包围盒的基础上，向四周各扩大一段安全距离（clear_dist）。
        # 目的：确保地图边缘有一定的“留白”，防止机器人走到地图边缘时因为数据截断而出错，或者为了包含一些边界外的安全缓冲区。
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist

        # 这里决定了地图在 X 和 Y 方向上各有多少个格子。
        # (max_x - min_x): 计算物理覆盖的总宽度（米）。
        # / resolution: 将物理宽度转换为格子数。
        self.num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10
        # 栅格索引 (0,0) 对应的物理世界坐标
        # (max_x + min_x) / 2.0: 计算点云物理范围的几何中心点。
        # self.num_x / 2.0 * resolution: 计算新生成的栅格地图总物理宽度的一半。
        # 公式含义：中心点坐标 - 半个地图宽度 = 地图左下角（起始）坐标。
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution

        print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    # 将之前步骤中提取和预处理的离散点云（Free & Obstacle Points），转换成一个连续的、带有梯度的 2D 代价网格（Grid Map）。
    def CreateTSDFMap(self):
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        # 初始化与栅格化 (Initialization & Rasterization)
        free_map = np.ones([self.num_x, self.num_y])  # 这里采用了“悲观假设”。默认所有区域都是未知的或不可通行的（值为1），
        obs_map = np.zeros([self.num_x, self.num_y])  # 障碍物地图初始化为0，表示没有障碍物。
        # 获取障碍物和自由空间点在栅格地图中的索引位置
        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)
        # create free place map
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=self._cfg_tsdf.sigma_expand)
        for i in free_I:
            if i[0] < self.num_x and i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)
        free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
        # assign obstacles 合并地图
        free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0

        print("occupancy map generation completed.")
        # Distance Transform
        # EDT (Euclidean Distance Transform)：
        # 这个函数计算的是非零元素（障碍物）到最近的零元素（安全区）的欧氏距离。
        # 结果含义：
        # 在安全区 (0)：结果为 0。
        # 在障碍物内 (1)：结果为 d，表示“你需要走多远才能逃出这个障碍物”。
        tsdf_array = ndimage.distance_transform_edt(free_map)
        # 对数缩放 (Log Scaling)：
        # 公式：Cost=ln(Distance+e)
        # Distance≈0时，ln(e)=1
        # 目的：软化梯度。线性的距离会导致优化器在深处产生过大的梯度，对数函数让惩罚增长得平缓一些，数值上更稳定。
        tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        # 再次高斯模糊，确保代价函数的导数连续，利于优化器（如梯度下降法）求解
        tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        # TODO: Using true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0
        return [tsdf_array, viz_points, ground_array], [
            float(self.start_x),
            float(self.start_y),
        ]

    def IndexArrayOfPs(self, points):
        indexes = points[:, :2] - np.array([self.start_x, self.start_y])
        indexes = (np.round(indexes / self._cfg_general.resolution)).astype(int)
        return indexes

    def FilterCloud(self, points, outlier_filter=True):
        # crop points 空间裁剪
        
        # 只要配置文件 (self._cfg_general) 中设置了 x_max, x_min, y_max, y_min 中的任意一个，就会执行裁剪
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            # 伪代码逻辑
            # if self._cfg_general.x_max is not None:
            #     # 如果配置里设置了 x_max，就计算哪些点的 X 坐标小于这个最大值
            #     points_x_idx_upper = (points[:, 0] < self._cfg_general.x_max)
            # else:
            #     # 如果没设置 x_max (为 None)，就生成一个全为 True 的数组（即保留所有点）
            #     points_x_idx_upper = np.ones(points.shape[0], dtype=bool)
            points_x_idx_upper = (
                (points[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_x_idx_lower = (
                (points[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_upper = (
                (points[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_lower = (
                (points[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            # 这行代码将四个掩码堆叠在一起，然后对每一列（即每一个点）进行“与”运算 (all)。
            # points = points[...] 利用 Numpy 的布尔索引直接过滤掉不符合要求的点。
            points = points[
                np.vstack(
                    (
                        points_x_idx_lower,
                        points_x_idx_upper,
                        points_y_idx_upper,
                        points_y_idx_lower,
                    )
                ).all(axis=0)
            ]

        if outlier_filter:
            # Filter outlier in points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self._cfg_tsdf.nb_neighbors,
                std_ratio=self._cfg_tsdf.std_ratio,
            )
            points = np.asarray(cl.points)

        return points

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])  # visualize point cloud


# EoF
