# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import os

import carb
import networkx as nx
import numpy as np
try:
    import omni.isaac.core.utils.prims as prims_utils
except Exception:
    prims_utils = None
import scipy.spatial.transform as tf
import torch
try:
    from omni.isaac.core.utils.semantics import get_semantics
except Exception:
    def get_semantics(_):
        return {"Semantics": []}
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import RayCaster, RayCasterCamera
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import raycast_mesh, convert_to_warp_mesh

try:
    from omni.isaac.matterport.domains import MatterportRayCaster, MatterportRayCasterCamera
except Exception:
    # 提供简化占位类，便于 isinstance 检查走到通用 USD-raycast 分支
    class MatterportRayCaster:  # type: ignore
        pass

    class MatterportRayCasterCamera:  # type: ignore
        pass
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import KDTree
from scipy.stats import qmc
from skimage.draw import line

from .terrain_analysis_cfg import TerrainAnalysisCfg
from .utils import get_all_meshes

# Optional YAML support for keyword-based semantic mapping
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


class TerrainAnalysis:
    def __init__(self, cfg: TerrainAnalysisCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene
        # Load keyword-based semantic map from environment if available
        self._keyword_map: dict[str, list[str]] | None = None
        try:
            map_path = os.environ.get("VIPLANNER_SEMANTIC_MAP")
            if map_path and yaml is not None and os.path.isfile(map_path):
                with open(map_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                # Expect format: {class_name: [keywords...]}
                if isinstance(data, dict):
                    self._keyword_map = {
                        str(k).lower(): list(map(lambda s: str(s).lower(), v))
                        for k, v in data.items()
                        if isinstance(v, (list, tuple))
                    }
                # Optional debug print to confirm mapping loaded
                if os.environ.get("VIPLANNER_DEBUG_SEM", "0") == "1":
                    print(f"[VIPLANNER_DEBUG] Loaded keyword map from {map_path}: {self._keyword_map}")
        except Exception:
            self._keyword_map = None

    def __del__(self):
        if hasattr(self, "_raycaster"):
            del self._raycaster
            del self._raycaster_mesh_param

    @property
    def device(self) -> str:
        return self.scene.device

    @property
    def complete(self) -> bool:
        return hasattr(self, "graph") and hasattr(self, "samples")

    @property
    def height_grid(self) -> torch.Tensor:
        if not hasattr(self, "_height_grid"):
            self._setup_raycaster()
            self.construct_height_map()
        return self._height_grid

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

    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Given position coordinates will return their respective height in the height map

        Args:
            positions: Coordinates of positions (Shape: [N, 2])

        Returns:
            The height of the positions (Shape: [N])
        """
        # get the indexes of the positions
        pos_idx = (
            (positions.cpu() - torch.tensor([self.mesh_dimensions[2], self.mesh_dimensions[3]]))
            / self.cfg.grid_resolution
        ).int()
        # clamp the indexes to the grid
        pos_idx[:, 0] = torch.clamp(pos_idx[:, 0], 0, self.height_grid.shape[0] - 1)
        pos_idx[:, 1] = torch.clamp(pos_idx[:, 1], 0, self.height_grid.shape[1] - 1)
        # get the height of the positions
        return self.height_grid[pos_idx[:, 0], pos_idx[:, 1]]

    ###
    # Helper functions
    ###

    def _sample_points(self):
        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False)
        sampled_nb_points = 0
        sampled_points = []

        print(f"[INFO] Sampling {self.cfg.sample_points} points...")
        while sampled_nb_points < self.cfg.sample_points:
            # get raw samples origins
            points = sampler.random(self.cfg.sample_points)
            points = qmc.scale(
                points,
                [self._mesh_dimensions[2], self._mesh_dimensions[3]],
                [self._mesh_dimensions[0], self._mesh_dimensions[1]],
            )
            heights = np.ones((self.cfg.sample_points, 1)) * self.cfg.wall_height

            ray_origins = torch.from_numpy(np.hstack((points, heights))).type(torch.float32).to(self.device)
            # print("--- IGNORE ---")
            # filter points that are outside the mesh or inside walls
            ray_origins, heights = self._point_filter_wall(ray_origins)
            # print("ray origins after wall filter:", ray_origins.shape[0])
            # filter points that are too close to walls
            ray_origins, heights = self._point_filter_wall_closeness(ray_origins, heights)
            
            if self.cfg.filter_indoor:
                # print("[INFO] Applying indoor filtering on sampled points...")
                ray_origins, heights = self._point_filter_indoor(ray_origins, heights)
            
            
            # print("ray origins after wall closeness filter:", ray_origins.shape[0])
            # filter points based on semantic cost
            if self.cfg.semantic_cost_mapping is not None:
                # print("[INFO] Applying semantic cost filtering on sampled points...")
                ray_origins, heights = self._point_filter_semantic_cost(ray_origins, heights)
                # print("ray origins after semantic cost filter:", ray_origins.shape[0])

            # set z height of samples to be at the robot's height above the terrain.
            ray_origins[:, 2] = heights + self.cfg.robot_height

            sampled_points.append(torch.clone(ray_origins))
            sampled_nb_points += ray_origins.shape[0]
            print(f"[INFO] Sampled {sampled_nb_points}/{self.cfg.sample_points} points...")

        self.points = torch.vstack(sampled_points)
        self.points = self.points[: self.cfg.sample_points]
        return

    def _construct_graph(self):
        # construct kdtree to find nearest neighbors of points
        print("[INFO] Constructing terrain graph...")
        kdtree = KDTree(self.points.cpu().numpy())
        _, nearest_neighbors_idx = kdtree.query(self.points.cpu().numpy(), k=self.cfg.num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = torch.tensor(nearest_neighbors_idx[:, 1:], dtype=torch.int64, device=self.device)

        # filter connections that collide with the environment
        print("[INFO] Filtering edges based on mesh collisions and height differences...")
        idx_edge_start, idx_edge_end, distance = self._edge_filter_mesh_collisions(nearest_neighbors_idx)

        (
            idx_edge_start,
            idx_edge_end,
            distance,
            idx_edge_start_filtered,
            idx_edge_end_filtered,
        ) = self._edge_filter_height_diff(idx_edge_start, idx_edge_end, distance)


        # filter edges based on semantic cost
        print("[INFO] Filtering edges based on semantic cost...")
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
        self.graph = nx.Graph()
        # add nodes with position attributes
        self.graph.add_nodes_from(list(range(self.cfg.sample_points)))
        pos_attr = {i: {"pos": self.points[i].cpu().numpy()} for i in range(self.cfg.sample_points)}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        # NOTE: as the shortest path searching algorithm only stores integers
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
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
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, cutoff=self.cfg.max_path_length, weight="distance")
        )

        # summarize to samples
        # samples are in the format (node, connected neighbor, distance)
        samples = []
        for key, value in odom_goal_distances.items():
            curr_samples = torch.zeros((len(value), 3))
            curr_samples[:, 0] = key
            curr_samples[:, 1] = torch.tensor(list(value.keys()))
            curr_samples[:, 2] = torch.tensor(list(value.values()))
            samples.append(curr_samples)
        self.samples = torch.vstack(samples).to(self.device)

        # --- Matplotlib 2D Graph Visualization (if enabled) ---
        if self.cfg.viz_graph:
            try:
                
            #     int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
            # int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) 
                import matplotlib.pyplot as plt
                # 动态设置画布尺寸，基于 mesh 尺寸自动缩放
                x_size = abs(self._mesh_dimensions[0] - self._mesh_dimensions[2])
                y_size = abs(self._mesh_dimensions[1] - self._mesh_dimensions[3])
                # 每个单位长度用 2 英寸，最小 6 英寸，最大 20 英寸
                fig_width = min(max(x_size * 2, 6), 20)
                fig_height = min(max(y_size * 2, 6), 20)
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                fig, ax = plt.subplots(figsize=(10, 10))
                # 画所有点
                points_np = self.points.cpu().numpy()
                ax.scatter(points_np[:, 0], points_np[:, 1], s=8, c="orange", label="Sample Points", alpha=0.8)
                # 画所有边
                for start_idx, end_idx in zip(idx_edge_start, idx_edge_end):
                    x = [points_np[start_idx, 0], points_np[end_idx, 0]]
                    y = [points_np[start_idx, 1], points_np[end_idx, 1]]
                    ax.plot(x, y, color="green", linewidth=0.5, alpha=0.5)
                # 画被过滤掉的边（高度差）
                for start_idx, end_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
                    x = [points_np[start_idx, 0], points_np[end_idx, 0]]
                    y = [points_np[start_idx, 1], points_np[end_idx, 1]]
                    ax.plot(x, y, color="red", linewidth=0.7, alpha=0.7, linestyle="--", label="Filtered (height)" if start_idx == idx_edge_start_filtered[0] and end_idx == idx_edge_end_filtered[0] else "")
                # 画被过滤掉的边（语义）
                if self.cfg.semantic_cost_mapping is not None and 'idx_edge_start_filtered_sem' in locals():
                    for start_idx, end_idx in zip(idx_edge_start_filtered_sem, idx_edge_end_filtered_sem):
                        x = [points_np[start_idx, 0], points_np[end_idx, 0]]
                        y = [points_np[start_idx, 1], points_np[end_idx, 1]]
                        ax.plot(x, y, color="purple", linewidth=0.7, alpha=0.7, linestyle=":", label="Filtered (semantic)" if start_idx == idx_edge_start_filtered_sem[0] and end_idx == idx_edge_end_filtered_sem[0] else "")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("Terrain Graph 2D Visualization")
                ax.set_aspect("equal")
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                plt.tight_layout()
                output_dir = "/home/eai/VLN/viplanner/viplanner_debug"
                os.makedirs(output_dir, exist_ok=True)
                fig_path = os.path.join(output_dir, "terrain_graph_2d.png")
                plt.savefig(fig_path, dpi=200)
                plt.close(fig)
                print(f"[INFO] 2D graph visualization saved to: {fig_path}")
            except Exception as e:
                print(f"[WARNING] Failed to plot 2D graph with matplotlib: {e}")
                    

        # debug visualization
        # if self.cfg.viz_graph:
        #     env_render_steps = 1000
        #     if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        #         print(f"[INFO] Visualizing graph. Will do {env_render_steps} render steps...")
        #     else:
        #         print("[INFO] Visualizing graph.")
            
        #     # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        #     try:
        #         import omni.isaac.debug_draw._debug_draw as omni_debug_draw

        #         draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        #         draw_interface.draw_points(
        #             self.points.tolist(),
        #             [(1.0, 0.5, 0, 1)] * self.cfg.sample_points,
        #             [5] * self.cfg.sample_points,
        #         )
        #         for start_idx, goal_idx in zip(idx_edge_start, idx_edge_end):
        #             draw_interface.draw_lines(
        #                 [self.points[start_idx].tolist()],
        #                 [self.points[goal_idx].tolist()],
        #                 [(0, 1, 0, 1)],
        #                 [1],
        #             )
        #         for start_idx, goal_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
        #             draw_interface.draw_lines(
        #                 [self.points[start_idx].tolist()],
        #                 [self.points[goal_idx].tolist()],
        #                 [(1, 0, 0, 1)],
        #                 [1],
        #             )
        #         if self.cfg.semantic_cost_mapping is not None:
        #             for start_idx, goal_idx in zip(idx_edge_start_filtered_sem, idx_edge_end_filtered_sem):
        #                 draw_interface.draw_lines(
        #                     [self.points[start_idx].tolist()],
        #                     [self.points[goal_idx].tolist()],
        #                     [(1, 0, 0, 1)],
        #                     [1],
        #                 )

        #         if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
        #             sim = SimulationContext.instance()
        #             for _ in range(env_render_steps):
        #                 sim.render()

        #             # clear the drawn points and lines
        #             draw_interface.clear_points()
        #             draw_interface.clear_lines()

        #             print("[INFO] Finished visualizing graph.")

        #     except ImportError as e:
        #         print(f"[ERROR] Import failed details: {e}")
        #         print("[WARNING] Graph Visualization is not available in headless mode.")

    ###
    # Mesh dimensions
    ###

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

    def _get_usd_stage_dimensions(self) -> tuple[float, float, float, float]:
        # get all mesh prims
        all_mesh_prims, all_mesh_prims_name = get_all_meshes(self.scene.terrain.cfg.prim_path)

        # if space limiter is given, only consider the meshes with the space limiter in the name
        if self.cfg.dim_limiter_prim:
            mesh_idx = [
                idx
                for idx, prim_name in enumerate(all_mesh_prims_name)
                if self.cfg.dim_limiter_prim.lower() in prim_name.lower()
            ]
        else:
            # remove ground plane since it has very large extent, but keep it as fallback
            mesh_idx = [idx for idx, prim_name in enumerate(all_mesh_prims_name) if "groundplane" not in prim_name.lower()]

        mesh_prims = [all_mesh_prims[idx] for idx in mesh_idx]
        # Fallback: if nothing matched (e.g., missing props), include groundplane to ensure non-empty bounds
        if len(mesh_prims) == 0:
            gp_idx = [idx for idx, prim_name in enumerate(all_mesh_prims_name) if "groundplane" in prim_name.lower()]
            if gp_idx:
                # rebuild from original list using indices
                mesh_prims = [all_mesh_prims[i] for i in gp_idx]
                try:
                    print("[DEBUG] Using groundplane for bounds fallback (no other meshes matched).")
                except Exception:
                    pass

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bbox = [self.compute_bbox_with_cache(bbox_cache, curr_prim) for curr_prim in mesh_prims]
        prim_max = np.vstack([list(prim_range.GetMax()) for prim_range in bbox])
        prim_min = np.vstack([list(prim_range.GetMin()) for prim_range in bbox])
        x_min, y_min, z_min = np.min(prim_min, axis=0)
        x_max, y_max, z_max = np.max(prim_max, axis=0)
        try:
            print(
                f"[DEBUG] USD bounds from {len(mesh_prims)} meshes: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}]"
            )
        except Exception:
            pass
        return x_max, y_max, x_min, y_min

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
                max_dist=self.cfg.wall_height * 2,
                **self._raycaster_mesh_param,
            )[0].squeeze(0)
        else:
            hit_point = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
                max_dist=self.cfg.wall_height * 2,
            )[0]

        # filter points outside the mesh and within walls
        filter_inside_mesh = torch.isfinite(hit_point[..., 2])  # outside mesh
        filter_outside_wall = hit_point[..., 2] < self.cfg.wall_height  # inside wall
        filter_combined = torch.all(torch.stack((filter_inside_mesh, filter_outside_wall), dim=1), dim=1)
        try:
            finite_hits = filter_inside_mesh.sum().item()
            sample_heights = hit_point[torch.isfinite(hit_point[..., 2])][:5, 2].tolist()
            print(
                f"[DEBUG] filtered {round(float((1 - filter_combined.sum() / self.cfg.sample_points) * 100), 4)} % of"
                f" points ({self.cfg.sample_points - filter_inside_mesh.sum()} outside of the mesh and"
                f" {self.cfg.sample_points - filter_outside_wall.sum()} points inside wall)."
                f" finite_hits={finite_hits}, sample_z={sample_heights}"
            )
        except Exception:
            pass

        return ray_origins[filter_combined].type(torch.float32), hit_point[filter_combined, 2]

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
                    max_dist=self.cfg.robot_buffer_spawn,
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
    
    
    # 加一个如果在室内就过滤掉该点，仅针对室外场景
    def _point_filter_indoor(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        check_origins = ray_origins.clone()
        check_origins[:, 2] = heights.squeeze() + self.cfg.robot_height
        is_indoor = self.check_if_indoor(
            ray_origins=check_origins,
            max_dist=self.cfg.indoor_distance_threshold
        )
        num_filtered = is_indoor.sum().item()
        if num_filtered > 0:
            print(f"[DEBUG] filtered {num_filtered} indoor points based on raycast hits")
        mask_outdoor = ~is_indoor
        ray_origins = ray_origins[mask_outdoor].type(torch.float32)
        if heights.ndim == 2:
            heights = heights[mask_outdoor, :]
        else:
            heights = heights[mask_outdoor]
        print(f"[DEBUG] Remaining points after indoor filter: {ray_origins.shape[0]}")
        return ray_origins, heights

    def _point_filter_semantic_cost(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # raycast vertically down and get the corresponding face id
        ray_directions = torch.zeros((ray_origins.shape[0], 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0

        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            ray_face_ids = raycast_mesh(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                max_dist=self.cfg.wall_height * 2,
                return_face_id=True,  # <--- 关键点：请求返回击中面片的ID
                **self._raycaster_mesh_param,
            )[3]

            # assign each hit the semantic class
            # self._raycaster.face_id_category_mapping，巨大的查找表（Lookup Table）键 (Key)：Mesh 的路径字符串（例如 "/World/House/Mesh_0"）。
            # 值 (Value)：一个巨大的 PyTorch 张量 (Tensor)。
            # 这个张量的长度等于该 Mesh 中三角形面片（Face）的总数量。
            # 这个张量的索引 (Index) 对应面片 ID（Face ID）。
            # 这个张量的数值 (Value) 对应这个面片的原始语义类别 ID。
            
            # self._raycaster.cfg：传感器的配置对象。
            # .mesh_prim_paths：一个列表，里面存了当前场景中所有被光线检测覆盖的 Mesh 的路径字符串。
            # [0]：取出列表里的第一个路径。通常 Matterport 场景把整个房子合并成了一个巨大的 Mesh，所以列表里通常只有一个路径。
            
            # ray_face_ids：
            # 这是上一行 raycast_mesh 函数的返回值。
            # 假设我们发射了 3 条射线，分别击中了第 10、第 100、第 5 号面片。
            # 它的形状可能是二维的 (1, 3)，内容是 [[10, 100, 5]]。
            # .flatten() (展平)：
            # PyTorch 的索引通常喜欢一维数组。
            # 这个操作把二维的 [[10, 100, 5]] 变成了 一维的 [10, 100, 5]。
            # .type(torch.long) (类型转换)：
            # 非常重要。PyTorch 规定，做索引（Index）用的张量，数据类型必须是 64位整型 (LongTensor / int64)。
            # 如果 ray_face_ids 原本是 32位整型 (int32) 或者浮点数 (float)，直接拿去查表会报错。这步操作就是强制转换类型，确保合规。
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            # Matterport3D 数据集的原始数据非常详细，里面的物体标签可能有成百上千种（例如：“扶手椅”、“办公椅”、“折叠椅”、“高脚凳”...）。
            # 但在机器人导航任务中，我们通常不需要分得这么细，我们只关心它是不是**“椅子”**。
            # 原始 ID (Raw ID)：可能是 102, 58, 99...（成百上千种）。
            # 标准 ID (MpCat40)：这是 Matterport 定义的 40 种通用类别。比如 ID 5 代表所有类型的“椅子”。
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]
            # get class_id to cost mapping 初始化“最坏情况”成本表
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            # class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max(
            #     list(self.cfg.semantic_cost_mapping.to_dict().values())
            # )
            mapping = self.cfg.semantic_cost_mapping
            max_cost = max(list(mapping.values()))
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max_cost
            
            # 根据配置“降价”（更新成本）
            # for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
            #     class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost  # 布尔掩码（Mask）

            for class_name, class_cost in mapping.items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost 

            # get cost
            cost = class_id_to_cost[class_id.cpu()]
        else:
            # 获取语义标签
            ray_classes = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
                max_dist=self.cfg.wall_height * 2,
                return_class=True,
            )[3]

            # get class to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            # max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            mapping = self.cfg.semantic_cost_mapping
            max_cost = max(list(mapping.values()))
            cost = torch.tensor(
                [
                    # 循环查找每个击中点对应的语义类别的成本
                    # self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
                    # for ray_class in ray_classes
                    mapping.get(ray_class, max_cost) if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )

        # filter points based on cost
        filter_cost = cost < self.cfg.semantic_cost_threshold
        print(f"[DEBUG] filtered {ray_origins.shape[0] - filter_cost.sum().item()} points based on semantic cost")
        return ray_origins[filter_cost].type(torch.float32), heights[filter_cost]

    ###
    # Edge filtering functions
    ###

    def _edge_filter_height_diff(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        # compute height difference
        height_diff = torch.diff(
            self._height_grid, dim=0, append=torch.zeros(1, self._height_grid.shape[1], device=self.device)
        ) + torch.diff(self._height_grid, dim=1, append=torch.zeros(self._height_grid.shape[0], 1, device=self.device))
        height_diff = np.abs(height_diff.cpu().numpy()) > self.cfg.height_diff_threshold

        # identify which edges are on different heights
        if self.cfg.height_diff_edge_filter:
            edge_idx = torch.abs(self.points[idx_edge_start, 2] - self.points[idx_edge_end, 2]) > 0.1
        else:
            edge_idx = torch.ones(self.points[idx_edge_start, 2].shape[0], dtype=bool, device=self.device)

        # filter edges that are on different heights
        check_idx_edge_start = idx_edge_start[edge_idx.cpu().numpy()]
        check_idx_edge_end = idx_edge_end[edge_idx.cpu().numpy()]

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
        
        # grid_z = torch.ones_like(grid_x, device=self.device) * max( list(self.cfg.semantic_cost_mapping.to_dict().values()) )
        
        mapping = self.cfg.semantic_cost_mapping
        max_cost = max(list(mapping.values()))
        print(f"[DEBUG] Semantic cost mapping: {mapping}, max_cost: {max_cost}")    
        grid_z = torch.ones_like(grid_x, device=self.device) * max_cost
        # 分辨率决定的
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points, device=self.device)
        direction[:, 2] = -1.0
        
        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            # check for collision with raycasting
            print(f"[DEBUG] Semantic cost grid stats: raycasting")
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
            # class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40)) * max(
            #     list(self.cfg.semantic_cost_mapping.to_dict().values())
            # )
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max_cost
            
            # for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
            #     class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost
            
            for class_name, class_cost in mapping.items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost 

            cost = class_id_to_cost[class_id.cpu()]
        else:
            # print(f"[DEBUG] Semantic cost grid stats: USD raycasting")
            # ray_classes = self._raycast_usd_stage(
            #     ray_starts=grid_points,
            #     ray_directions=direction,
            #     max_dist=self.cfg.wall_height * 2,
            #     return_class=True,
            # )[3]
            # print(f"[DEBUG] Semantic cost grid stats: ray_classes obtained")
            # # get class to cost mapping
            # assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            # # max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            
            # # mapping = self.cfg.semantic_cost_mapping
            # # max_cost = max(list(mapping.values()))
            
            # cost = torch.tensor(
            #     [
            #         # self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
            #         # for ray_class in ray_classes
            #         mapping.get(ray_class, max_cost) if ray_class is not None else max_cost
            #         for ray_class in ray_classes
            #     ],
            #     device=self.device,
            # )
            
            # 配置通用的 Isaac Lab RayCaster 传感器。这会让代码在几何计算（高度图、墙壁碰撞）时使用 GPU 加速。对于语义计算，由于通用 USD 不支持 Warp 语义查询，它仍会走 CPU 分支（也就是我上一条回答建议的“分批处理”是必须的）。
            
            print(f"[DEBUG] Semantic cost grid stats: USD raycasting (Batch Processing)")
            
            # 定义批次大小，例如每次处理 10,000 个点
            # 根据你的内存大小调整，10000-50000 通常是安全的
            batch_size = 40000 
            num_points = grid_points.shape[0]
            all_ray_classes = []
            
            import math
            num_batches = math.ceil(num_points / batch_size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_points)
                
                # 切片获取当前批次的数据
                batch_starts = grid_points[start_idx:end_idx]
                batch_dirs = direction[start_idx:end_idx]
                
                print(f"[DEBUG] Processing batch {i+1}/{num_batches} (Points: {len(batch_starts)})...")
                
                # 对当前批次进行射线检测
                # 注意：这里我们不需要 distance 和 normal，只关心 class
                batch_classes = self._raycast_usd_stage(
                    ray_starts=batch_starts,
                    ray_directions=batch_dirs,
                    max_dist=self.cfg.wall_height * 2,
                    return_class=True,
                )[3]
                
                all_ray_classes.extend(batch_classes)
                
            ray_classes = all_ray_classes
            print(f"[DEBUG] Semantic cost grid stats: ray_classes obtained ({len(ray_classes)} total)")

            # get class to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            
            mapping = self.cfg.semantic_cost_mapping
            max_cost = max(list(mapping.values()))
            
            cost = torch.tensor(
                [
                    mapping.get(ray_class, max_cost) if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )
        print(f"[DEBUG] Semantic cost grid stats: get")
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
        print(f"[DEBUG] Semantic cost grid stats: prepared indices")
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

    # 构建环境的几何表示：为了能让光线“看见”虚拟世界，必须把 USD 场景中的几何体（Mesh）提取出来。
    # 双重构建策略：
    # 通用 Mesh (self._warp_mesh)：包含所有物体（地面、墙、桌子），用于物理射线检测（Raycasting），计算地形高度和碰撞。
    # 障碍物 Mesh (self._warp_mesh_obstacles)：专门把“非地面”的物体提取出来，构建一个独立的网格。这通常用于更高级的语义可通行性检查（例如：光线打到了东西，如果是打到 _warp_mesh_obstacles，那就坚决不能走）。
    def _setup_raycaster(self):
        # [优化] 缓存检查：如果已经构建过 Warp Mesh，直接返回，避免重复计算
        if hasattr(self, "_warp_mesh") and self._warp_mesh is not None:
            # 同时也检查一下障碍物 mesh 是否存在
            if hasattr(self, "_warp_mesh_obstacles") and self._warp_mesh_obstacles is not None:
                # 已经初始化过了，无需重复操作
                return
        
        # get the raycaster sensor that should be used to raycast against all the ground meshes
        # 尝试使用现成的 Raycaster 传感器
        sensor_key = self.cfg.raycaster_sensor
        # 检查配置里是否指定了 Raycaster，以及场景里是否真的有这个传感器
        use_sensor = (
            sensor_key is not None
            and hasattr(self.scene, "sensors")
            and sensor_key in self.scene.sensors
            and isinstance(
                self.scene.sensors[sensor_key],
                MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera,
            )
        )

        if use_sensor:
            # 动作：从 self.scene.sensors 字典中，根据配置的 sensor_key（例如 "main_lidar" 或 "terrain_scanner"）取出对应的传感器对象。
            # 赋值：将其赋值给 self._raycaster。
            # 类型注解：中间那一长串 MatterportRayCaster | ... 是 Python 的类型提示，告诉阅读者和 IDE，这个变量可能是这四种传感器类型中的任意一种。这说明代码设计时考虑了兼容不同的传感器实现。
            self._raycaster: MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera = (
                self.scene.sensors[sensor_key]
            )
            # 兼容性检查
            if isinstance(self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]], list):
                # for new RSL implementation of raycaster
                # FIXME: @pascal-roth: this is a temporary fix until the new raycaster is merged into the public main branch
                # 注释：明确指出这是一个 FIXME（待修复的临时补丁）。说明新版 Raycaster 还没合并到公共主分支，所以API有些不同。
                self._raycaster_mesh_param = {"mesh_id": self._raycaster._mesh_ids_wp.numpy()[0][0]}
            else:
                self._raycaster_mesh_param = {"mesh": self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]]}

            # get mesh dimensions [x_max, y_max, x_min, y_min] 获取网格尺寸
            self._mesh_dimensions = self._get_mesh_dimensions()
        else:
            # raycaster is not available in multi-mesh scenes (i.e. unreal meshes) as it only works with a single mesh
            # TODO (@pascal-roth) change when raycaster can handle multiple meshes
            self._raycaster = None

            # get mesh dimensions [x_max, y_max, x_min, y_min] # 1. 计算场景边界
            self._mesh_dimensions = self._get_usd_stage_dimensions()

            # 这段代码的主要目的是：当场景中没有现成的 RayCaster 传感器时，手动从 USD 场景中提取几何信息，构建一个“Warp Mesh”对象，以便利用 GPU 加速进行地形分析。
            # Build a combined Warp mesh from USD stage to enable geometry raycasts without PhysX colliders
            try:
                # # 获取场景中所有的 Mesh Prim（图元）
                all_mesh_prims, all_mesh_names = get_all_meshes(self.scene.terrain.cfg.prim_path)
                # Filter per limiter (exclude groundplane unless used as fallback)
                if self.cfg.dim_limiter_prim:
                    selected = [
                        idx for idx, name in enumerate(all_mesh_names) if self.cfg.dim_limiter_prim.lower() in name.lower()
                    ]
                else:
                    selected = [idx for idx, name in enumerate(all_mesh_names) if "groundplane" not in name.lower()]
                mesh_prims = [all_mesh_prims[idx] for idx in selected]
                # Fallback to groundplane(s) if empty
                if len(mesh_prims) == 0:
                    gp_idx = [idx for idx, name in enumerate(all_mesh_names) if "groundplane" in name.lower()]
                    mesh_prims = [all_mesh_prims[i] for i in gp_idx]

                # 初始化 USD 几何工具
                from pxr import UsdGeom, Usd
                xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default()) # 用于计算世界坐标变换
                # 准备列表存储顶点和面
                verts_all: list[np.ndarray] = []
                faces_all: list[np.ndarray] = []
                v_offset = 0
                for prim in mesh_prims:
                    try:
                        mesh = UsdGeom.Mesh(prim)
                        pts = mesh.GetPointsAttr().Get()
                        fvi = mesh.GetFaceVertexIndicesAttr().Get()
                        if pts is None or fvi is None:
                            continue
                        # world transform
                        xf = xform_cache.GetLocalToWorldTransform(prim)
                        world_pts = np.array([list(xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in pts], dtype=np.float32)
                        fvi = np.asarray(fvi, dtype=np.int64)
                        if fvi.size % 3 != 0:
                            # skip non-triangulated mesh for now
                            continue
                        faces = fvi.reshape(-1, 3) + v_offset
                        verts_all.append(world_pts)
                        faces_all.append(faces)
                        v_offset += world_pts.shape[0]
                    except Exception:
                        continue

                if len(verts_all) > 0 and len(faces_all) > 0:
                    verts_np = np.vstack(verts_all)
                    faces_np = np.vstack(faces_all)
                    device = "cuda" if "cuda" in self.device else "cpu"
                    self._warp_mesh = convert_to_warp_mesh(verts_np, faces_np, device=device)
                    self._raycaster_mesh_param = {"mesh": self._warp_mesh}
                    # sentinel to use raycast_mesh path
                    self._raycaster = self
                    try:
                        print(
                            f"[DEBUG] Built warp mesh for raycasting: V={verts_np.shape[0]}, F={faces_np.shape[0]} from {len(mesh_prims)} prims"
                        )
                    except Exception:
                        pass
                else:
                    try:
                        print("[WARNING] Failed to build warp mesh (no valid vertices/faces). Falling back to PhysX stage queries.")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    print(f"[WARNING] Exception while building warp mesh: {e}. Falling back to PhysX stage queries.")
                except Exception:
                    pass

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

        # Build obstacle-only warp mesh for semantic traversability checking (if keyword map is available)
        # 构建障碍物mesh：所有不属于floor的mesh都视为障碍物
        self._warp_mesh_obstacles = None
        self._obstacle_class = None
        if self._keyword_map is not None:
            try:
                all_mesh_prims, all_mesh_names = get_all_meshes(self.scene.terrain.cfg.prim_path)
                
                # 从 keyword_map 中获取 floor 关键词（可通行区域）# road sidewalk crosswalk floor
                # 所有不匹配 floor 关键词的 mesh 都将被视为障碍物
                floor_keywords = []
                # 支持多个可通行类别（如 road, sidewalk, crosswalk, floor）
                traversable_classes = ["road", "sidewalk", "crosswalk", "floor"]
                floor_keywords = []
                for cls in traversable_classes:
                    if cls in self._keyword_map:
                        floor_keywords.extend([kw.lower() for kw in self._keyword_map[cls]])
                
                # 收集障碍物mesh（所有非floor的mesh）
                obstacle_verts: list[np.ndarray] = []
                obstacle_faces: list[np.ndarray] = []
                v_offset_obstacles = 0
                obstacle_count = 0

                from pxr import UsdGeom, Usd
                xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                
                for prim, name in zip(all_mesh_prims, all_mesh_names):
                    name_lower = name.lower()
                    path_lower = prim.GetPath().pathString.lower()
                    
                    # 检查是否是 floor（可通行地面）
                    is_floor = any(kw in name_lower or kw in path_lower for kw in floor_keywords)
                    
                    # 只要不是 floor，就视为障碍物
                    if is_floor:
                        continue
                    
                    try:
                        mesh = UsdGeom.Mesh(prim)
                        pts = mesh.GetPointsAttr().Get()
                        fvi = mesh.GetFaceVertexIndicesAttr().Get()
                        if pts is None or fvi is None:
                            continue
                        
                        # 坐标变换到世界坐标系
                        xf = xform_cache.GetLocalToWorldTransform(prim)
                        world_pts = np.array([list(xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in pts], dtype=np.float32)
                        
                        # 类型转换
                        fvi = np.asarray(fvi, dtype=np.int64)
                        if fvi.size % 3 != 0:
                            continue
                        
                        # 面索引调整（加全局偏移量）
                        faces = fvi.reshape(-1, 3) + v_offset_obstacles
                        
                        obstacle_verts.append(world_pts)
                        obstacle_faces.append(faces)
                        v_offset_obstacles += world_pts.shape[0]
                        obstacle_count += 1
                        
                    except Exception:
                        continue

                # 合并所有障碍物mesh
                if len(obstacle_verts) > 0 and len(obstacle_faces) > 0:
                    verts_obstacles_np = np.vstack(obstacle_verts)
                    faces_obstacles_np = np.vstack(obstacle_faces)
                    device = "cuda" if "cuda" in self.device else "cpu"
                    self._warp_mesh_obstacles = convert_to_warp_mesh(verts_obstacles_np, faces_obstacles_np, device=device)
                    self._obstacle_class = "obstacles"
                    
                    # 保存障碍物mesh到OBJ文件，方便检查
                    try:
                        import os
                        output_dir = "/home/eai/VLN/viplanner/viplanner_debug"
                        os.makedirs(output_dir, exist_ok=True)
                        obj_path = os.path.join(output_dir, "obstacle_mesh.obj")
                        ply_path = os.path.join(output_dir, "obstacle_mesh.ply")
                        
                        print(f"[DEBUG] Saving obstacle mesh ...")
                        print(f"[DEBUG] Vertices: {verts_obstacles_np.shape}, dtype: {verts_obstacles_np.dtype}")
                        print(f"[DEBUG] Faces: {faces_obstacles_np.shape}, dtype: {faces_obstacles_np.dtype}")
                        
                        # 方法1: 使用 trimesh (更可靠)
                        try:
                            import trimesh
                            mesh_obj = trimesh.Trimesh(
                                vertices=verts_obstacles_np, 
                                faces=faces_obstacles_np,
                                process=False  # 不要自动处理，保持原始数据
                            )
                            mesh_obj.export(obj_path)
                            mesh_obj.export(ply_path)
                            
                            obj_size_mb = os.path.getsize(obj_path) / (1024 * 1024)
                            ply_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
                            
                            print(f"[DEBUG] ✓ Saved OBJ: {obj_path} ({obj_size_mb:.2f} MB)")
                            print(f"[DEBUG] ✓ Saved PLY: {ply_path} ({ply_size_mb:.2f} MB)")
                            
                        except ImportError:
                            # 方法2: 手动写入（如果没有trimesh）
                            print("[DEBUG] trimesh not found, using manual export...")
                            
                            # 保存为PLY格式（更简单，MeshLab支持更好）
                            with open(ply_path, 'w') as f:
                                f.write("ply\n")
                                f.write("format ascii 1.0\n")
                                f.write(f"element vertex {verts_obstacles_np.shape[0]}\n")
                                f.write("property float x\n")
                                f.write("property float y\n")
                                f.write("property float z\n")
                                f.write(f"element face {faces_obstacles_np.shape[0]}\n")
                                f.write("property list uchar int vertex_indices\n")
                                f.write("end_header\n")
                                
                                # 写入顶点
                                for v in verts_obstacles_np:
                                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                                
                                # 写入面
                                for face in faces_obstacles_np:
                                    f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
                            
                            ply_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
                            print(f"[DEBUG] ✓ Saved PLY: {ply_path} ({ply_size_mb:.2f} MB)")
                        
                        print(
                            f"[DEBUG] Built obstacle warp mesh: V={verts_obstacles_np.shape[0]}, F={faces_obstacles_np.shape[0]} "
                            f"from {obstacle_count} obstacle meshes (non-traversable)"
                        )
                        print(f"[DEBUG] ✓ Save completed! Try opening with MeshLab or CloudCompare")
                        
                    except Exception as save_err:
                        print(f"[WARNING] Failed to save obstacle mesh: {save_err}")
                        import traceback
                        traceback.print_exc()
                else:
                    try:
                        print(f"[WARNING] No obstacle meshes found. Traversability check will not consider obstacles.")
                    except Exception:
                        pass
                        
            except Exception as e:
                try:
                    print(f"[WARNING] Failed to build obstacle warp mesh: {e}")
                except Exception:
                    pass

    ###
    # Construct height map of the environment
    ###

    def construct_height_map(self):
        # get dimensions and construct height grid with raycasting
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
        grid_z = torch.ones_like(grid_x) * (self.cfg.wall_height * 2)
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points)
        direction[:, 2] = -1.0

        # check for collision with raycasting from the top
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
                    draw_interface.draw_points(
                        hit_point[lower_height].cpu().tolist(),
                        [(0.0, 0.7, 0.0, 1)] * hit_point[lower_height].shape[0],
                        [5] * hit_point[lower_height].shape[0],
                    )
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

            except ImportError as e:
                print(f"[ERROR] Import failed details: {e}")
                print("[WARNING] Height Map Visualization is not available in headless mode.")

    def check_if_indoor(self, ray_origins: torch.Tensor, max_dist: float) -> torch.Tensor:
        """
        判断点是否在室内（即周围大部分方向都被障碍物包围）。
        返回: Bool Tensor (True=室内, False=室外)
        """
        device = ray_origins.device
        num_points = ray_origins.shape[0]
        
        if num_points == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        
        # 1. 生成射线方向 (20个方向，360度覆盖)
        # endpoint=False 避免 -pi 和 pi 重复
        angles = np.linspace(-np.pi, np.pi, 20, endpoint=False)
        
        # 计算方向向量 (20, 3)
        ray_directions_np = tf.Rotation.from_euler("z", angles, degrees=False).as_matrix() @ np.array([1, 0, 0])
        
        hits_list = []

        # 2. 循环执行 Raycast
        for ray_dir in ray_directions_np:
            # 扩展方向向量: (N, 3)
            ray_direction_torch = (
                torch.from_numpy(ray_dir)
                .type(torch.float32)
                .to(device)
                .unsqueeze(0)
                .repeat(num_points, 1)
            )

            # 执行 Raycast
            if self._raycaster is not None:
                # 使用 Warp Mesh 加速
                result = raycast_mesh(
                    ray_starts=ray_origins.unsqueeze(0),
                    ray_directions=ray_direction_torch.unsqueeze(0),
                    max_dist=max_dist,
                    return_distance=True,
                    **self._raycaster_mesh_param,
                )
                distance = result[1].squeeze(0)
            else:
                # 使用 USD Stage (较慢)
                result = self._raycast_usd_stage(
                    ray_starts=ray_origins,
                    ray_directions=ray_direction_torch,
                    max_dist=max_dist,
                    return_distance=True,
                )
                distance = result[1]

            # 3. 判定击中
            # 击中 = (距离不是无穷大) AND (距离小于设定阈值)
            has_hit = (~torch.isinf(distance)) & (distance < max_dist)
            hits_list.append(has_hit)

        # 4. 判定室内
        if not hits_list:
            return torch.zeros(num_points, dtype=torch.bool, device=device)
        # shape: [num_directions, num_points]
        all_hits = torch.stack(hits_list, dim=0)
        # 计算包围比例
        hit_ratio = torch.mean(all_hits.float(), dim=0)
        # 如果超过 90% 的方向都被挡住，认为是室内
        is_indoor = hit_ratio > 0.9
        return is_indoor


    ###
    # Helper function when isaaclab raycaster is not available
    ### 

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

        if os.environ.get("VIPLANNER_DEBUG_SEM", "0") == "1":
            total_rays = len(hits)
            hit_count = sum(1 for h in hits if h["hit"])
            sample_paths = [str(h.get("collision", "")) for h in hits if h["hit"]][:5]
            print(f"[VIPLANNER_DEBUG] USD raycast: total_rays={total_rays}, hit_count={hit_count}")
            if sample_paths:
                print(f"[VIPLANNER_DEBUG] Sample collision paths: {sample_paths}")

        # record collision paths for debugging/mapping refinement
        try:
            self._last_ray_collision_paths = [
                (str(single_hit.get("collision", "")) if single_hit["hit"] else "") for single_hit in hits
            ]
        except Exception:
            self._last_ray_collision_paths = []

        # get all hit idx
        hit_idx = [idx for idx, single_hit in enumerate(hits) if single_hit["hit"]]

        # hit positions
        hit_positions = torch.zeros_like(ray_starts).fill_(float("inf"))
        if len(hit_idx) > 0:
            _pos_list = [single_hit["position"] for single_hit in hits if single_hit["hit"]]
            if len(_pos_list) > 0:
                _pos = torch.tensor(_pos_list, device=ray_starts.device, dtype=ray_starts.dtype)
                hit_positions[hit_idx] = _pos

        # get distance
        if return_distance:
            ray_distance = torch.zeros(ray_starts.shape[0], device=ray_starts.device).fill_(float("inf"))
            if len(hit_idx) > 0:
                _dist_list = [single_hit["distance"] for single_hit in hits if single_hit["hit"]]
                if len(_dist_list) > 0:
                    _dist = torch.tensor(_dist_list, device=ray_starts.device, dtype=ray_starts.dtype)
                    ray_distance[hit_idx] = _dist
        else:
            ray_distance = None

        # get normal
        if return_normal:
            ray_normal = torch.zeros_like(ray_starts).fill_(float("inf"))
            if len(hit_idx) > 0:
                _nrm_list = [single_hit["normal"] for single_hit in hits if single_hit["hit"]]
                if len(_nrm_list) > 0:
                    _nrm = torch.tensor(_nrm_list, device=ray_starts.device, dtype=ray_starts.dtype)
                    ray_normal[hit_idx] = _nrm
        else:
            ray_normal = None

        # get class
        if return_class:
            if self._keyword_map is not None:
                # Prefer YAML keyword-based mapping when available
                ray_class = []
                for single_hit in hits:
                    if single_hit["hit"]:
                        path_str = str(single_hit.get("collision", "")).lower()
                        assigned = None
                        # First match wins; order of classes in YAML decides precedence
                        for cls_name, keywords in self._keyword_map.items():
                            if any(kw in path_str for kw in keywords):
                                assigned = cls_name
                                break
                        ray_class.append(assigned)
                    else:
                        ray_class.append(None)
            elif prims_utils is not None:
                # Fall back to Isaac Core semantics if no keyword map is provided
                ray_class = [
                    (
                        get_semantics(prims_utils.get_prim_at_path(single_hit["collision"]))["Semantics"][1]
                        if single_hit["hit"]
                        else None
                    )
                    for single_hit in hits
                ]
            else:
                # No semantics source available
                ray_class = [None for _ in hits]
        else:
            ray_class = None

        return hit_positions, ray_distance, ray_normal, ray_class
