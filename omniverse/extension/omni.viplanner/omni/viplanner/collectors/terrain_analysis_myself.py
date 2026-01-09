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
        # construct kdtree to find nearest neighbors of points
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

        # debug visualization
        if self.cfg.viz_graph:
            env_render_steps = 1000
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing graph. Will do {env_render_steps} render steps...")
            else:
                print("[INFO] Visualizing graph.")

            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                draw_interface.draw_points(
                    self.points.tolist(),
                    [(1.0, 0.5, 0, 1)] * self.cfg.sample_points,
                    [5] * self.cfg.sample_points,
                )
                for start_idx, goal_idx in zip(idx_edge_start, idx_edge_end):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(0, 1, 0, 1)],
                        [1],
                    )
                for start_idx, goal_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(1, 0, 0, 1)],
                        [1],
                    )
                if self.cfg.semantic_cost_mapping is not None:
                    for start_idx, goal_idx in zip(idx_edge_start_filtered_sem, idx_edge_end_filtered_sem):
                        draw_interface.draw_lines(
                            [self.points[start_idx].tolist()],
                            [self.points[goal_idx].tolist()],
                            [(1, 0, 0, 1)],
                            [1],
                        )

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
                return_face_id=True,
                **self._raycaster_mesh_param,
            )[3]

            # assign each hit the semantic class
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]

            # get class_id to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max(
                list(self.cfg.semantic_cost_mapping.to_dict().values())
            )
            for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost

            # get cost
            cost = class_id_to_cost[class_id.cpu()]
        else:
            ray_classes = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
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
        sensor_key = self.cfg.raycaster_sensor
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
            self._raycaster: MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera = (
                self.scene.sensors[sensor_key]
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

            # Build a combined Warp mesh from USD stage to enable geometry raycasts without PhysX colliders
            try:
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

                from pxr import UsdGeom, Usd
                xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
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
                
                # 从 keyword_map 中获取 floor 关键词（可通行区域）
                # 所有不匹配 floor 关键词的 mesh 都将被视为障碍物
                floor_keywords = []
                if 'floor' in self._keyword_map:
                    floor_keywords = [kw.lower() for kw in self._keyword_map['floor']]
                
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

            except ImportError:
                print("[WARNING] Height Map Visualization is not available in headless mode.")

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
