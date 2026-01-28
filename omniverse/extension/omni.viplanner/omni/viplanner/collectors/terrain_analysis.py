# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins

import carb
import networkx as nx
import numpy as np
import omni.isaac.core.utils.prims as prims_utils
import scipy.spatial.transform as tf
import torch
from omni.isaac.core.utils.semantics import get_semantics
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import RayCaster, RayCasterCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import raycast_mesh
from omni.isaac.matterport.domains import MatterportRayCaster, MatterportRayCasterCamera
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import KDTree
from scipy.stats import qmc
from skimage.draw import line

from .terrain_analysis_cfg import TerrainAnalysisCfg
from .utils import get_all_meshes


class TerrainAnalysis:
    # 输入: 地形分析配置和场景
    def __init__(self, cfg: TerrainAnalysisCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene
    # 析构函数，用于清理射线检测器（Raycaster）资源，防止内存泄漏。
    def __del__(self):
        if hasattr(self, "_raycaster"):
            del self._raycaster
            del self._raycaster_mesh_param
    # 返回计算设备（如 "cuda:0" 或 "cpu"）
    # 指明当前的计算硬件是 CPU 还是 GPU（以及是哪一块 GPU），以确保所有数据都在同一个地方进行运算
    # PyTorch 张量的“同设备”原则、避免数据搬运（性能优化）
    @property
    def device(self) -> str:
        return self.scene.device
    
    # 检查分析是否完成（即图和样本数据是否已生成）
    @property
    def complete(self) -> bool:
        return hasattr(self, "graph") and hasattr(self, "samples")

    # 懒加载：如果高度图不存在，先设置射线检测器并构建高度图，然后返回。
    @property
    def height_grid(self) -> torch.Tensor:
        if not hasattr(self, "_height_grid"):
            self._setup_raycaster()
            self.construct_height_map()
        return self._height_grid
    
    # 懒加载：获取地形的边界范围 (x_max, y_max, x_min, y_min)。

    @property
    def mesh_dimensions(self) -> tuple[float, float, float, float]:
        if not hasattr(self, "_mesh_dimensions"):
            self._setup_raycaster()
        return self._mesh_dimensions

    ###
    # Operations
    ###

    def analyse(self):
        print("[INFO] Starting terrain analysis...")
        # get raycaster and mesh dimensions
        self._setup_raycaster()
        # build height grid of the environment
        self.construct_height_map()
        # get the points and sample the graph
        self._sample_points()
        self._construct_graph()
        
    # 输入: 一组 (N, 2) 的 XY 坐标
    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Given position coordinates will return their respective height in the height map

        Args:
            positions: Coordinates of positions (Shape: [N, 2])

        Returns:
            The height of the positions (Shape: [N])
        """
        # get the indexes of the positions 将连续坐标转换为网格索引，从 height_grid 中查表
        pos_idx = (
            (positions.cpu() - torch.tensor([self.mesh_dimensions[2], self.mesh_dimensions[3]]))
            / self.cfg.grid_resolution
        ).int()
        # clamp the indexes to the grid
        pos_idx[:, 0] = torch.clamp(pos_idx[:, 0], 0, self.height_grid.shape[0] - 1)
        pos_idx[:, 1] = torch.clamp(pos_idx[:, 1], 0, self.height_grid.shape[1] - 1)
        # get the height of the positions 对应位置的地形高度 Z
        return self.height_grid[pos_idx[:, 0], pos_idx[:, 1]]

    # 输入: 机器人位置。
    # 功能: 从位置向下发射射线，测量到地面的距离。
    # 输出: 布尔值 Tensor，判断该位置下方是否有足够的空间（例如是否悬空太高或位置合法）。注意：这里的逻辑是 distance > robot_height，通常用于检查机器人是否已经“生成”在足够高的地方，或者是检查某种特定的间隙。
    # 验证“通过性” (Headroom / Tunnel Check)：
    # 验证“悬空安全性” (Flying / Dropout Check)：
    def check_clearance(self, positions: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
        """Check if the given positions have enough clearance from the terrain.

        Args:
            positions: Coordinates of positions (Shape: [N, 3])
            eps: Small offset above the terrain to start the raycast
        Returns:
            Boolean tensor indicating whether the positions have enough clearance (Shape: [N])
        """
        # raycast downwards from the positions
        ray_directions = torch.zeros((positions.shape[0], 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0
        ray_starts = positions.clone()
        ray_starts[:, 2] += eps
        
        # 默认情况：如果用户使用的是标准的 Isaac Lab 地形（单 Mesh），直接上 GPU 加速，秒级完成分析。
        # 复杂情况：如果用户导入了一个极其复杂的外部 USD 场景（多 Mesh 组合），GPU Raycaster 可能不支持或无法正确绑定，此时代码不会崩溃，而是自动降级（Fallback）到 CPU 物理引擎查询。虽然慢一点，但能保证功能正常运行。
        # 依赖：Isaac Lab / Warp / CUDA。
        # 工作原理：它直接利用 GPU 并行计算能力。
        # 优势：极快。可以在几毫秒内并行处理成千上万条射线。这对于强化学习训练或大范围地形分析（需要撒几千个点）是必须的。
        # 局限性：
        # 代码中提到它目前通常只支持单一 Mesh（Single Mesh）。
        # 如果你的场景是由程序化生成的单一地形（Terrain），它工作得很完美。
        if self._raycaster is not None:
            distance = raycast_mesh(
                ray_starts=ray_starts.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                return_distance=True,
                **self._raycaster_mesh_param,
            )[1].squeeze(0)
        else:
            distance = self._raycast_usd_stage(
                ray_starts=ray_starts,
                ray_directions=ray_directions,
                return_distance=True,
            )[1]

        # check if the distance is greater than the required clearance
        return distance > self.cfg.robot_height + eps


    ###
    # Helper functions
    ###

    def _sample_points(self):
        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False) # # 使用 Halton 序列进行准随机采样（比纯随机更均匀）
        sampled_nb_points = 0
        sampled_points = []

        print(f"[INFO] Sampling {self.cfg.sample_points} points...")
        # 循环直到采集到足够的点
        while sampled_nb_points < self.cfg.sample_points:
            # get raw samples origins
            points = sampler.random(self.cfg.sample_points)
            points = qmc.scale(
                points,
                [self._mesh_dimensions[2], self._mesh_dimensions[3]],
                [self._mesh_dimensions[0], self._mesh_dimensions[1]],
            ) # 缩放点到地图尺寸
            # 目的：它暂时把所有采样点的 Z 轴（高度） 都设定在一个较高的位置（墙的高度）。这是为了后续做“从上往下”的射线检测准备的
            heights = np.ones((self.cfg.sample_points, 1)) * self.cfg.wall_height
            # np.hstack 将 X 和 Y 坐标与 Z 坐标（高度）合并成一个 (N, 3) 的点云
            # torch.from_numpy 把 NumPy 数组（Python 科学计算的标准格式）转换成 PyTorch Tensor（深度学习张量格式）
            # .to(self.device) 数据搬运 它将这个张量从 CPU 内存 复制到 GPU 显存（如果 self.device 是 "cuda:0"）
            ray_origins = torch.from_numpy(np.hstack((points, heights))).type(torch.float32).to(self.device)

            # filter points that are outside the mesh or inside walls
            ray_origins, heights = self._point_filter_wall(ray_origins)

            # filter points that are too close to walls
            ray_origins, heights = self._point_filter_wall_closeness(ray_origins, heights)

            # filter points based on semantic cost
            if self.cfg.semantic_cost_mapping is not None:
                ray_origins, heights = self._point_filter_semantic_cost(ray_origins, heights)

            # set z height of samples to be at the robot's height above the terrain.
            ray_origins[:, 2] = heights + self.cfg.robot_height

            sampled_points.append(torch.clone(ray_origins))
            sampled_nb_points += ray_origins.shape[0]

        self.points = torch.vstack(sampled_points)
        self.points = self.points[: self.cfg.sample_points]
        return

    def _construct_graph(self):
        # construct kdtree to find nearest neighbors of points # 构建 KDTree 用于快速查找最近邻
        kdtree = KDTree(self.points.cpu().numpy())
        _, nearest_neighbors_idx = kdtree.query(self.points.cpu().numpy(), k=self.cfg.num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = torch.tensor(nearest_neighbors_idx[:, 1:], dtype=torch.int64, device=self.device)

        # filter connections that collide with the environment
        idx_edge_start, idx_edge_end, distance = self._edge_filter_mesh_collisions(nearest_neighbors_idx)

        (
            idx_edge_start,
            idx_edge_end,
            distance,
            idx_edge_start_filtered,
            idx_edge_end_filtered,
        ) = self._edge_filter_height_diff(idx_edge_start, idx_edge_end, distance)

        # filter edges based on semantic cost
        if self.cfg.semantic_cost_mapping is not None:
            (
                idx_edge_start,
                idx_edge_end,
                distance,
                idx_edge_start_filtered_sem,
                idx_edge_end_filtered_sem,
            ) = self._edge_filter_semantic_cost(idx_edge_start, idx_edge_end, distance)

        # init graph
        print(f"[INFO] Constructing graph with {idx_edge_start.shape[0]} edges")
        self.graph = nx.Graph() # 创建 NetworkX 图
        # add nodes with position attributes
        # 含义：给每个节点“贴标签”，记录它的真实 3D 坐标。
        # 关键点 self.points[i].cpu().numpy()：
        # 之前的 self.points 是存在 GPU 上的 PyTorch Tensor。
        # 但是 NetworkX 是一个运行在 CPU 上的纯 Python 库，它看不懂 GPU 数据。
        # 所以这里必须先把数据 .cpu() (搬回内存) 再 .numpy() (转成 Numpy 数组)，才能存进 NetworkX 的图里。
        # 结果：现在图里的节点 0 知道自己位置在 (x0, y0, z0)，节点 1 知道自己位置在 (x1, y1, z1)，以此类推。
        self.graph.add_nodes_from(list(range(self.cfg.sample_points)))
        pos_attr = {i: {"pos": self.points[i].cpu().numpy()} for i in range(self.cfg.sample_points)}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        # NOTE: as the shortest path searching algorithm only stores integers
        # 含义：在节点之间建立连接。
        # 数据来源：idx_edge_start 和 idx_edge_end 是之前经过层层筛选（去墙壁、去悬崖）后剩下的合法的起点和终点索引。
        # 操作：
        # np.stack(..., axis=1)：把起点数组 [0, 1] 和终点数组 [5, 6] 拼成 [[0, 5], [1, 6]]。
        # map(tuple, ...)：转换成元组格式 (0, 5), (1, 6)，这是 NetworkX 要求的格式。
        # 结果：节点 0 和 节点 5 连通了，表示机器人可以直接从 0 走到 5。
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
        
        # 含义：给每条边（路）标记“长度/成本”。
        # 为什么需要这个？：
        # 虽然节点 A 和节点 B 连通，节点 A 和节点 C 也连通。但是 A-B 距离 1 米，A-C 距离 10 米。
        # 如果不记录距离，寻路算法会以为走哪条路代价都一样。
        # 这里把物理距离 distance 赋值给边的 distance 属性，Dijkstra 算法就会利用这个属性来寻找“最短”路径。
        distance_attr = {
            (i, j): {"distance": distance[idx]} for idx, (i, j) in enumerate(zip(idx_edge_start, idx_edge_end))
        }
        nx.set_edge_attributes(self.graph, distance_attr)

        # remove nodes with no edges
        # NOTE: while the nodes are removed, the node ids and the self.point id are still the same, thus we don't prune
        #   the self.points tensor
        self.isolated_points_ids = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(self.isolated_points_ids)
        print(f"[INFO] Removed {len(self.isolated_points_ids)} isolated nodes")

        # get all shortest paths
        # nx.all_pairs_dijkstra_path_length
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, cutoff=self.cfg.max_path_length, weight="distance")
        )

        # summarize to samples
        # samples are in the format (node, connected neighbor, distance)
        # 之前的数据都在 CPU 上用 NetworkX 处理，现在需要把结果打包好送回 GPU，供后续的强化学习训练使用。
        samples = []
        for key, value in odom_goal_distances.items():
            curr_samples = torch.zeros((len(value), 3))
            curr_samples[:, 0] = key # node id
            curr_samples[:, 1] = torch.tensor(list(value.keys())) # connected neighbor id
            curr_samples[:, 2] = torch.tensor(list(value.values())) # distance
            samples.append(curr_samples)
        self.samples = torch.vstack(samples).to(self.device)

        # debug visualization 在 Isaac Sim 的 3D 视口中，把计算出来的**导航图（Graph）**画出来，让开发者能直观地看到地形分析的结果。具体来说，它会把采样点画成点，把连通关系画成线，并用颜色区分“可行路径”和“被过滤掉的路径”。
        if self.cfg.viz_graph:
            env_render_steps = 1000
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing graph. Will do {env_render_steps} render steps...")
            else:
                print("[INFO] Visualizing graph.")

            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                # 获取 Isaac Sim 的 DebugDraw 工具。这是一个用于在 3D 场景中绘制简单的点、线、框的 API。
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                # 绘制节点（画点）
                draw_interface.draw_points(
                    self.points.tolist(),
                    [(1.0, 0.5, 0, 1)] * self.cfg.sample_points,
                    [5] * self.cfg.sample_points,
                )
                # 绘制可行边（画绿线）
                for start_idx, goal_idx in zip(idx_edge_start, idx_edge_end):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(0, 1, 0, 1)],
                        [1],
                    )
                # 绘制被过滤掉的边（画红线）
                for start_idx, goal_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(1, 0, 0, 1)],
                        [1],
                    )
                # 绘制语义被过滤掉的边（画红线）
                if self.cfg.semantic_cost_mapping is not None:
                    for start_idx, goal_idx in zip(idx_edge_start_filtered_sem, idx_edge_end_filtered_sem):
                        draw_interface.draw_lines(
                            [self.points[start_idx].tolist()],
                            [self.points[goal_idx].tolist()],
                            [(1, 0, 0, 1)],
                            [1],
                        )
                # 渲染循环与清理
                if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                    sim = SimulationContext.instance()
                    for _ in range(env_render_steps):
                        sim.render()

                    # clear the drawn points and lines
                    draw_interface.clear_points()
                    draw_interface.clear_lines()

                    print("[INFO] Finished visualizing graph.")

            except ImportError:
                print("[WARNING] Graph Visualization is not available in headless mode.")

    ###
    # Mesh dimensions
    ###
    # 从 Raycaster 的 Mesh 数据中计算边界
    def _get_mesh_dimensions(self) -> tuple[float, float, float, float]:
        # get min, max of the mesh in the xy plane
        # Get bounds of the terrain
        bounds = []
        for mesh in self._raycaster.meshes.values():
            curr_bounds = torch.zeros((2, 3))
            # for new RSL implementation of raycaster
            # FIXME: @pascal-roth: this is a temporary fix until the new raycaster is merged into the public main branch
            if isinstance(mesh, list):
                curr_bounds[0] = torch.tensor(mesh[0][0].points).max(dim=0)[0]
                curr_bounds[1] = torch.tensor(mesh[0][0].points).min(dim=0)[0]
            else:
                curr_bounds[0] = torch.tensor(mesh.points).max(dim=0)[0]
                curr_bounds[1] = torch.tensor(mesh.points).min(dim=0)[0]
            bounds.append(curr_bounds)
        bounds = torch.vstack(bounds)
        x_min, y_min = bounds[:, 0].min().item(), bounds[:, 1].min().item()
        x_max, y_max = bounds[:, 0].max().item(), bounds[:, 1].max().item()
        return x_max, y_max, x_min, y_min

    # 如果没有 Raycaster，则遍历 USD 场景中的 Prim 计算边界（较慢）。
    def _get_usd_stage_dimensions(self) -> tuple[float, float, float, float]:
        # get all mesh prims
        mesh_prims, mesh_prims_name = get_all_meshes(self.scene.terrain.cfg.prim_path)

        # if space limiter is given, only consider the meshes with the space limiter in the name
        if self.cfg.dim_limiter_prim:
            mesh_idx = [
                idx
                for idx, prim_name in enumerate(mesh_prims_name)
                if self.cfg.dim_limiter_prim.lower() in prim_name.lower()
            ]
        else:
            # remove ground plane since has infinite extent
            mesh_idx = [idx for idx, prim_name in enumerate(mesh_prims_name) if "groundplane" not in prim_name.lower()]

        mesh_prims = [mesh_prims[idx] for idx in mesh_idx]

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bbox = [self.compute_bbox_with_cache(bbox_cache, curr_prim) for curr_prim in mesh_prims]
        prim_max = np.vstack([list(prim_range.GetMax()) for prim_range in bbox])
        prim_min = np.vstack([list(prim_range.GetMin()) for prim_range in bbox])
        x_min, y_min, z_min = np.min(prim_min, axis=0)
        x_max, y_max, z_max = np.max(prim_max, axis=0)

        return x_max, y_max, x_min, y_min

    # 计算 USD Prim 包围盒的工具函数
    @staticmethod
    def compute_bbox_with_cache(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Range3d:
        """
        Compute Bounding Box using ComputeWorldBound at UsdGeom.BBoxCache. More efficient if used multiple times.
        See https://graphics.pixar.com/usd/dev/api/class_usd_geom_b_box_cache.html

        Args:
            cache: A cached, i.e. `UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])`
            prim: A prim to compute the bounding box.
        Returns:
            A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html

        """
        bound = cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()
        return bound_range

    ###
    # Point filter functions
    ###

    # 从上方发射射线。如果击中点高度 hit_z 大于 wall_height（意味着点在墙顶）或射线未击中（在地图外），则过滤掉。
    def _point_filter_wall(self, ray_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get ray directions in negative z direction
        ray_directions = torch.zeros((self.cfg.sample_points, 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0
        # elevate the ray origins to be above the height of the walls
        ray_origins[:, 2] += 1.0

        if self._raycaster is not None:
            hit_point = raycast_mesh(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                **self._raycaster_mesh_param,
            )[0].squeeze(0)
        else:
            hit_point = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
            )[0]

        # filter points outside the mesh and within walls
        filter_inside_mesh = torch.isfinite(hit_point[..., 2])  # outside mesh
        filter_outside_wall = hit_point[..., 2] < self.cfg.wall_height  # inside wall
        # torch.all(..., dim=1)：
        # 沿着行（dim=1）检查：“这一行里的所有值都是 True 吗？”
        # 只有两个条件同时满足，结果才是 True。
        filter_combined = torch.all(torch.stack((filter_inside_mesh, filter_outside_wall), dim=1), dim=1)
        print(
            f"[DEBUG] filtered {round(float((1 - filter_combined.sum() / self.cfg.sample_points) * 100), 4)} % of"
            f" points ({self.cfg.sample_points - filter_inside_mesh.sum()} outside of the mesh and"
            f" {self.cfg.sample_points - filter_outside_wall.sum()} points inside wall)"
        )

        return ray_origins[filter_combined].type(torch.float32), hit_point[filter_combined, 2]

    # 向四周 360 度发射射线。
    # 如果任意方向在 robot_buffer_spawn 距离内击中障碍物，则认为离墙太近，过滤掉。
    def _point_filter_wall_closeness(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # reduce ground height to check for closeness to walls and other objects
        ray_origins[:, 2] = heights + self.cfg.robot_height
        # enforce a minimum distance to the walls
        angles = np.linspace(-np.pi, np.pi, 20)
        ray_directions = tf.Rotation.from_euler("z", angles, degrees=False).as_matrix() @ np.array([1, 0, 0])
        ray_hit = []

        for ray_direction in ray_directions:
            ray_direction_torch = (
                torch.from_numpy(ray_direction).repeat(ray_origins.shape[0], 1).type(torch.float32).to(self.device)
            )
            if self._raycaster is not None:
                distance = raycast_mesh(
                    ray_starts=ray_origins.unsqueeze(0),
                    ray_directions=ray_direction_torch.unsqueeze(0),
                    max_dist=self.cfg.robot_buffer_spawn, # 这里是阈值
                    return_distance=True,
                    **self._raycaster_mesh_param,
                )[1].squeeze(0)
            else:
                distance = self._raycast_usd_stage(
                    ray_starts=ray_origins,
                    ray_directions=ray_direction_torch,
                    max_dist=self.cfg.robot_buffer_spawn,
                    return_distance=True,
                )[1]
            ray_hit.append(torch.isinf(distance))

        # check if every point has the minimum distance in every direction
        without_wall = torch.all(torch.vstack(ray_hit), dim=0)

        print(f"[DEBUG] filtered {ray_origins.shape[0] - without_wall.sum().item()} points too close to walls")
        ray_origins = ray_origins[without_wall].type(torch.float32)
        heights = heights[without_wall]
        return ray_origins, heights

    # 基于语义标签（例如：草地、水、路面）给予不同的成本，过滤掉成本过高的区域。
    # 输入：ray_origins（采样点的 3D 坐标），heights（对应的高度）。
    # 输出：过滤后剩下的点和高度。
    def _point_filter_semantic_cost(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # raycast vertically down and get the corresponding face id
        ray_directions = torch.zeros((ray_origins.shape[0], 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0
        
        # 如果我们使用的是专门为 Matterport（一种室内 3D 扫描数据集）设计的 RayCaster。
        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            
            # 射线检测：向下发射射线。return_face_id=True：这次我们不关心距离，只关心射线击中了网格上的哪一个三角形面片（Triangle Face）。[3]：取出返回元组中的第 4 个元素，即 face_id。
            ray_face_ids = raycast_mesh(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                max_dist=self.cfg.wall_height * 2,
                return_face_id=True,
                **self._raycaster_mesh_param,
            )[3]

            # assign each hit the semantic class
            # 类别归约（原始类别 -> 标准类别 mpcat40）：
            # 原始扫描数据可能有几百种标签。mpcat40 是一个标准集，只有 40 种常见类别（如 floor, wall, chair, table 等）。
            # 这里把复杂的原始 ID 映射到这 40 个通用 ID 上。
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]

            # get class_id to cost mapping
            # 初始化成本表：
            # 创建一个长度为 40（类别总数）的向量 class_id_to_cost。
            # 默认值设为最大成本（假设所有东西都不可走，除非配置里说了可以走）。
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max(
                list(self.cfg.semantic_cost_mapping.to_dict().values())
            )
            # 读取用户的配置 semantic_cost_mapping（例如：{'floor': 0.0, 'water': 100.0}）。
            # 找到 'floor' 在 mpcat40 中的索引，把对应的成本改为 0.0
            for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost

            # get cost
            cost = class_id_to_cost[class_id.cpu()]
        else:
            # 通用检测：如果不使用专用 RayCaster，调用通用的 _raycast_usd_stage。
            # return_class=True：这次请求 PhysX 返回击中物体的语义标签（Semantics）。返回的通常是一个字符串列表（例如 ["floor", "wall", "door"]）。
            ray_classes = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
                max_dist=self.cfg.wall_height * 2,
                return_class=True,
            )[3]

            # get class to cost mapping
            # 字符串匹配计算成本：
            # 这里用了一个 Python 列表推导式（List Comprehension）。
            # 遍历每一个射线击中的类别名称 ray_class。
            # 如果在配置字典里找到了这个名字，就用配置的成本；如果没找到（或者射线没击中），就给 max_cost。
            # 最后转为 Tensor。
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            cost = torch.tensor(
                [
                    self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )

        # filter points based on cost
        # 生成“布尔掩码” (Boolean Mask)filter_cost 是一个布尔张量（Boolean Tensor），内容是 [True, False, True, False]。
        filter_cost = cost < self.cfg.semantic_cost_threshold
        print(f"[DEBUG] filtered {ray_origins.shape[0] - filter_cost.sum().item()} points based on semantic cost")
        # 保留对应位置为 True 的元素，丢弃对应位置为 False 的元素
        return ray_origins[filter_cost].type(torch.float32), heights[filter_cost]

    ###
    # Edge filtering functions
    ###

    # 检查边的两端以及边经过的网格路径上，高度变化是否超过阈值（防止机器人走悬崖或陡坡）。
    def _edge_filter_height_diff(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        # compute height difference
        # 目的：计算高度网格中每个像素点与其相邻像素点的高度差（梯度）。
        # torch.diff(..., dim=0)：计算行与行之间的差异（X方向梯度）。
        height_diff = torch.diff(
            self._height_grid, dim=0, append=torch.zeros(1, self._height_grid.shape[1], device=self.device)
        ) + torch.diff(self._height_grid, dim=1, append=torch.zeros(self._height_grid.shape[0], 1, device=self.device))
        # 目的：生成一个布尔网格（Boolean Grid），即“悬崖地图”。
        # 逻辑：如果某个格子的邻域高度变化超过了阈值（height_diff_threshold），就标记为 True（危险/陡峭），否则为 False（平坦）。
        height_diff = np.abs(height_diff.cpu().numpy()) > self.cfg.height_diff_threshold

        # identify which edges are on different heights
        # 优化逻辑：并不是所有的边都需要去跑复杂的像素级检测。
        # edge_idx（此时的角色是待检查列表）：
        # 如果起终点的高度差大于 0.1 米，说明这条边是有坡度的，嫌疑很大，需要通过后续步骤细查。
        # 如果起终点高度几乎一样，代码假设它们之间是平的，直接放行（这就跳过了后面昂贵的 line 循环）。
        # 注：else 分支是全选，即不开启优化，检查所有边。
        if self.cfg.height_diff_edge_filter:
            edge_idx = torch.abs(self.points[idx_edge_start, 2] - self.points[idx_edge_end, 2]) > 0.1
        else:
            edge_idx = torch.ones(self.points[idx_edge_start, 2].shape[0], dtype=bool, device=self.device)

        # filter edges that are on different heights
        # 筛选：利用刚才生成的 edge_idx 掩码，只提取出那些“有高度差、需要检查”的边。
        check_idx_edge_start = idx_edge_start[edge_idx.cpu().numpy()]
        check_idx_edge_end = idx_edge_end[edge_idx.cpu().numpy()]

        # 坐标系转换：将 世界坐标 (World Position, float) 转换为 网格索引 (Grid Index, int)。
        # 公式：(当前坐标 - 地图原点坐标) / 分辨率，然后取整。
        # 目的：因为前面的 height_diff 是一个图片（网格），我们要去图片上查像素，必须用整数索引。
        check_grid_idx_start = (
            (
                (
                    self.points[check_idx_edge_start, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )
        check_grid_idx_end = (
            (
                (
                    self.points[check_idx_edge_end, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )
        # 循环：遍历每一条待检查的边。
        # line(...)：这是一个来自 skimage.draw 的函数。给定起点和终点像素，它会返回两点连线上经过的所有像素坐标。
        # np.any(...)：检查这条连线经过的像素中，有没有任意一个像素在 height_diff 图中是 True（陡峭的）。
        # 结果：如果连线跨过了悬崖，filter_idx[idx] 设为 True（表示这条边该被切断）。
        filter_idx = np.zeros(check_idx_edge_start.shape[0], dtype=bool)

        for idx, (edge_start_idx, edge_end_idx) in enumerate(zip(check_grid_idx_start, check_grid_idx_end)):
            grid_idx_x, grid_idx_y = line(edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1])

            filter_idx[idx] = np.any(height_diff[grid_idx_x, grid_idx_y])

        # set the indexes that should be removed in edge_idx to true
        edge_idx[edge_idx.clone()] = torch.tensor(filter_idx, device=self.device)
        edge_idx = edge_idx.cpu().numpy()
        # filter edges
        idx_edge_start_filtered = idx_edge_start[edge_idx]
        idx_edge_end_filtered = idx_edge_end[edge_idx]

        idx_edge_start = idx_edge_start[~edge_idx]
        idx_edge_end = idx_edge_end[~edge_idx]
        distance = distance[~edge_idx]

        return idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered

    # 在两点之间发射射线，如果发生碰撞且距离小于两点间距，说明中间有障碍物，断开边。
    def _edge_filter_mesh_collisions(
        self, nearest_neighbors_idx: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter connections that collide with the environment."""
        # define origin and neighbor points
        origin_point = torch.repeat_interleave(self.points, repeats=self.cfg.num_connections, axis=0)
        neighbor_points = self.points[nearest_neighbors_idx, :].reshape(-1, 3)
        min_distance = torch.norm(origin_point - neighbor_points, dim=1)

        # check for collision with raycasting
        if self._raycaster is not None:
            distance = raycast_mesh(
                ray_starts=origin_point.unsqueeze(0),
                ray_directions=(origin_point - neighbor_points).unsqueeze(0),
                max_dist=self.cfg.max_path_length,
                return_distance=True,
                **self._raycaster_mesh_param,
            )[1]
        else:
            distance = self._raycast_usd_stage(
                ray_starts=origin_point,
                ray_directions=(origin_point - neighbor_points),
                max_dist=self.cfg.max_path_length,
                return_distance=True,
            )[1]

        distance[torch.isinf(distance)] = self.cfg.max_path_length
        # filter connections that collide with the environment
        collision = (distance < min_distance).reshape(-1, self.cfg.num_connections)

        # get edge indices
        idx_edge_start = np.repeat(np.arange(self.cfg.sample_points), repeats=self.cfg.num_connections, axis=0)
        idx_edge_end = nearest_neighbors_idx.reshape(-1).cpu().numpy()

        # filter collision edges and distances
        idx_edge_end = idx_edge_end[~collision.reshape(-1).cpu().numpy()]
        idx_edge_start = idx_edge_start[~collision.reshape(-1).cpu().numpy()]
        distance = min_distance[~collision.reshape(-1)].cpu().numpy()

        return idx_edge_start, idx_edge_end, distance

    def _edge_filter_semantic_cost(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                self._mesh_dimensions[2],
                self._mesh_dimensions[0],
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
            torch.linspace(
                self._mesh_dimensions[3],
                self._mesh_dimensions[1],
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
        )
        grid_z = torch.ones_like(grid_x, device=self.device) * max(
            list(self.cfg.semantic_cost_mapping.to_dict().values())
        )
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points, device=self.device)
        direction[:, 2] = -1.0

        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            # check for collision with raycasting
            ray_face_ids = raycast_mesh(
                ray_starts=grid_points.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=self.cfg.wall_height * 2,
                return_face_id=True,
                **self._raycaster_mesh_param,
            )[3].squeeze(0)

            # assign each hit the semantic class
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]

            # get class_id to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40)) * max(
                list(self.cfg.semantic_cost_mapping.to_dict().values())
            )
            for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost

            cost = class_id_to_cost[class_id.cpu()]
        else:
            ray_classes = self._raycast_usd_stage(
                ray_starts=grid_points,
                ray_directions=direction,
                max_dist=self.cfg.wall_height * 2,
                return_class=True,
            )[3]

            # get class to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            cost = torch.tensor(
                [
                    self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )

        # get cost grid
        cost_grid = (
            cost.reshape(
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
            )
            .cpu()
            .numpy()
        )

        # get grid indexes of edges
        check_grid_idx_start = (
            (
                (
                    self.points[idx_edge_start, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.points.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )
        check_grid_idx_end = (
            (
                (
                    self.points[idx_edge_end, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.points.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )

        filter_idx = np.zeros(check_grid_idx_start.shape[0], dtype=bool)

        for idx, (edge_start_idx, edge_end_idx) in enumerate(zip(check_grid_idx_start, check_grid_idx_end)):
            grid_idx_x, grid_idx_y = line(edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1])

            filter_idx[idx] = np.any(cost_grid[grid_idx_x, grid_idx_y] > self.cfg.semantic_cost_threshold)

        # filter edges
        idx_edge_start_filtered = idx_edge_start[filter_idx]
        idx_edge_end_filtered = idx_edge_end[filter_idx]

        idx_edge_start = idx_edge_start[~filter_idx]
        idx_edge_end = idx_edge_end[~filter_idx]
        distance = distance[~filter_idx]

        return idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered

    def _setup_raycaster(self):
        # get the raycaster sensor that should be used to raycast against all the ground meshes
        if isinstance(
            self.scene.sensors[self.cfg.raycaster_sensor],
            MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera,
        ):
            self._raycaster: MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera = (
                self.scene.sensors[self.cfg.raycaster_sensor]
            )

            if isinstance(self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]], list):
                # for new RSL implementation of raycaster
                # FIXME: @pascal-roth: this is a temporary fix until the new raycaster is merged into the public main branch
                self._raycaster_mesh_param = {"mesh_id": self._raycaster._mesh_ids_wp.numpy()[0][0]}
            else:
                self._raycaster_mesh_param = {"mesh": self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]]}

            # get mesh dimensions [x_max, y_max, x_min, y_min]
            self._mesh_dimensions = self._get_mesh_dimensions()
        else:
            # raycaster is not available in multi-mesh scenes (i.e. unreal meshes) as it only works with a single mesh
            # TODO (@pascal-roth) change when raycaster can handle multiple meshes
            self._raycaster = None

            # get mesh dimensions [x_max, y_max, x_min, y_min]
            self._mesh_dimensions = self._get_usd_stage_dimensions()

        self._mesh_dimensions = list(self._mesh_dimensions)

        # limit the size of the mesh if required (otherwise run out of memory)
        if self.cfg.max_terrain_size is not None:
            if self._mesh_dimensions[0] - self._mesh_dimensions[2] > self.cfg.max_terrain_size:
                print(f"[WARNING] Mesh is too large in the x dimension, limiting to {self.cfg.max_terrain_size} max")
                mesh_over_limit = (self._mesh_dimensions[0] - self._mesh_dimensions[2] - self.cfg.max_terrain_size) / 2
                self._mesh_dimensions[0] -= mesh_over_limit
                self._mesh_dimensions[2] += mesh_over_limit
            if self._mesh_dimensions[1] - self._mesh_dimensions[3] > self.cfg.max_terrain_size:
                print(f"[WARNING] Mesh is too large in the y dimension, limiting to {self.cfg.max_terrain_size} max")
                mesh_over_limit = (self._mesh_dimensions[1] - self._mesh_dimensions[3] - self.cfg.max_terrain_size) / 2
                self._mesh_dimensions[1] -= mesh_over_limit
                self._mesh_dimensions[3] += mesh_over_limit

    ###
    # Construct height map of the environment
    ###

    def construct_height_map(self):
        # get dimensions and construct height grid with raycasting
        # 目的：创建一个覆盖整个地图的 XY 网格。
        grid_x, grid_y = torch.meshgrid( # 将两个一维数列（x轴坐标和y轴坐标）组合成两个二维矩阵 grid_x 和 grid_y，分别表示网格中每个像素点的 X 和 Y 坐标。
            torch.linspace( # torch.linspace：生成从最小值到最大值的等间距数列。点的数量由 (总长度 / 分辨率) 计算得出。
                self._mesh_dimensions[2],
                self._mesh_dimensions[0],
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
            torch.linspace(
                self._mesh_dimensions[3],
                self._mesh_dimensions[1],
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
        )
        # 目的：设定射线的起始高度。
        # 逻辑：创建一个和网格一样大的矩阵，所有值设为 wall_height * 2（比如墙高2米，这里就设为4米）。这是为了保证射线从足够高的地方发射，能覆盖所有地形特征。
        grid_z = torch.ones_like(grid_x) * (self.cfg.wall_height * 2)
        # 目的：将 X, Y, Z 三个矩阵压扁并堆叠，转置成 (N, 3) 的格式。
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        # 目的：设定射线方向。
        direction = torch.zeros_like(grid_points)
        direction[:, 2] = -1.0

        # check for collision with raycasting from the top
        # 目的：执行物理查询，测量地形高度。
        if self._raycaster is not None:
            hit_point = raycast_mesh(
                ray_starts=grid_points.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=15,
                **self._raycaster_mesh_param,
            )[0].squeeze(0)
        else:
            hit_point = self._raycast_usd_stage(
                ray_starts=grid_points,
                ray_directions=direction,
                max_dist=15,
            )[0]
        # 问题：如果击中了门框顶部（Z=2.0），高度图会认为这里高 2 米，机器人就会认为这里是墙，过不去。但实际上机器人可以从门框下面钻过去。
        
        # detection of doors inside walls
        # we raycast one more time shortly above the ground up and down, if the up raycast hits and is lower than the
        # initial raycast, the height of the down raycast will be used
        if self.cfg.door_filtering:
            # adopt the height
            grid_points[..., 2] = 0.5

            # check for potential hit downwards
            if self._raycaster is not None:
                hit_point_down = raycast_mesh(
                    ray_starts=grid_points.unsqueeze(0),
                    ray_directions=direction.unsqueeze(0),
                    max_dist=15,
                    **self._raycaster_mesh_param,
                )[0].squeeze(0)
            else:
                hit_point_down = self._raycast_usd_stage(
                    ray_starts=grid_points,
                    ray_directions=direction,
                    max_dist=15,
                )[0]

            # change the direction of the raycaster to the top
            direction[:, 2] = 1.0

            # check for potential hit upwards
            if self._raycaster is not None:
                hit_point_up = raycast_mesh(
                    ray_starts=grid_points.unsqueeze(0),
                    ray_directions=direction.unsqueeze(0),
                    max_dist=15,
                    **self._raycaster_mesh_param,
                )[0].squeeze(0)
            else:
                hit_point_up = self._raycast_usd_stage(
                    ray_starts=grid_points,
                    ray_directions=direction,
                    max_dist=15,
                )[0]

            # check if up height scan receives a hit and is lower than the initial height scan (from above the wall height)
            # and where the difference is larger than the height difference threshold
            # hit_point_up < hit_point：向上看击中的点（门框底面）比最初从天上看击中的点（门框顶面或屋顶）要低。这证明我们处于某种结构的“内部”或“下方”。
            # isfinite(hit_point_up)：向上看确实击中了东西（不是露天）。
            # (hit_point_up - hit_point_down) > threshold：净空高度（门框底面 - 地面）足够大，机器人钻得过去。
            # isfinite(hit_point_down)：脚下确实有地面。
            lower_height = (
                (hit_point_up[..., 2] < (hit_point[..., 2] - 1e-3))
                & torch.isfinite(hit_point_up[..., 2])
                & ((hit_point_up[..., 2] - hit_point_down[..., 2]) > self.cfg.door_height_threshold)
                & torch.isfinite(hit_point_down[..., 2])
            )
            # override height with the lower height
            hit_point[lower_height] = hit_point_down[lower_height]

        # get the height grid
        self._height_grid = hit_point[:, 2].reshape(
            int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
            int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
        )

        if self.cfg.viz_height_map:
            env_render_steps = 1000
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing height map. Will do {env_render_steps} render steps...")
            else:
                print("[INFO] Visualizing height map.")

            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                # add small offset to height grid to visualize it
                hit_point[:, 2] += 0.1

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                if self.cfg.door_filtering:
                    # 绿色点：被“门框过滤”修正过的点（可以通行）
                    draw_interface.draw_points(
                        hit_point[lower_height].cpu().tolist(),
                        [(0.0, 0.7, 0.0, 1)] * hit_point[lower_height].shape[0],
                        [5] * hit_point[lower_height].shape[0],
                    )
                    # 蓝色点：普通的点（地面、墙壁顶部等）
                    draw_interface.draw_points(
                        hit_point[~lower_height].cpu().tolist(),
                        [(0.0, 0.0, 0.7, 1)] * hit_point[~lower_height].shape[0],
                        [5] * hit_point[~lower_height].shape[0],
                    )
                else:
                    draw_interface.draw_points(
                        hit_point.cpu().tolist(),
                        [(0.0, 0.0, 0.7, 1)] * hit_point.shape[0],
                        [5] * hit_point.shape[0],
                    )

                if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                    sim = SimulationContext.instance()
                    for _ in range(env_render_steps):
                        sim.render()

                    # clear the drawn points and lines
                    draw_interface.clear_points()

                    print("[INFO] Finished visualizing height map.")

            except ImportError:
                print("[WARNING] Height Map Visualization is not available in headless mode.")

    
    ###
    # Helper function when isaaclab raycaster is not available
    ###
    # 依赖：PhysX 物理引擎的 CPU 查询接口 (get_physx_scene_query_interface)。
    # 工作原理：它通过 CPU 遍历查询场景中的物理碰撞体。
    # 优势：兼容性极强。它不管场景里有几个 Mesh，也不管模型是从 Unreal 导入的、Maya 画的，还是由 100 个零件组成的复杂环境（如 Matterport 扫描的室内场景），只要有碰撞体（Collider），它就能测出来。
    # 局限性：慢。
    # 看代码中的实现：[... for ... in ...] 是一个 Python 层面的循环列表推导式。
    # 如果采样 10,000 个点，它就要在 CPU 上循环 10,000 次调用 PhysX 接口，速度比 GPU 慢几个数量级。
    def _raycast_usd_stage(
        self,
        ray_starts: torch.Tensor,
        ray_directions: torch.Tensor,
        max_dist: float = 1e6,
        return_distance: bool = False,
        return_normal: bool = False,
        return_class: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list | None]:
        """
        Perform raycasting over the entire loaded stage.

        Interface is the same as the normal raycast_mesh function without the option to provide specific meshes.
        """

        hits = [
            get_physx_scene_query_interface().raycast_closest(carb.Float3(ray_single), carb.Float3(ray_dir), max_dist)
            for ray_single, ray_dir in zip(ray_starts.cpu().numpy(), ray_directions.cpu().numpy())
        ]

        # get all hit idx
        hit_idx = [idx for idx, single_hit in enumerate(hits) if single_hit["hit"]]

        # hit positions
        hit_positions = torch.zeros_like(ray_starts).fill_(float("inf"))
        hit_positions[hit_idx] = torch.tensor([single_hit["position"] for single_hit in hits if single_hit["hit"]]).to(
            ray_starts.device
        )

        # get distance
        if return_distance:
            ray_distance = torch.zeros(ray_starts.shape[0], device=ray_starts.device).fill_(float("inf"))
            ray_distance[hit_idx] = torch.tensor(
                [single_hit["distance"] for single_hit in hits if single_hit["hit"]]
            ).to(ray_starts.device)
        else:
            ray_distance = None

        # get normal
        if return_normal:
            ray_normal = torch.zeros_like(ray_starts).fill_(float("inf"))
            ray_normal[hit_idx] = torch.tensor([single_hit["normal"] for single_hit in hits if single_hit["hit"]])
        else:
            ray_normal = None

        # get class
        if return_class:
            ray_class = [
                (
                    get_semantics(prims_utils.get_prim_at_path(single_hit["collision"]))["Semantics"][1]
                    if single_hit["hit"]
                    else None
                )
                for single_hit in hits
            ]
        else:
            ray_class = None

        return hit_positions, ray_distance, ray_normal, ray_class
