# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class TerrainAnalysisCfg:
    robot_height: float = 0.6
    """Height of the robot

    机器人高度（相机中心/采样点在地面上的抬升高度）。
    - 生效: 是（在 TerrainAnalysis._point_filter_wall_closeness / _sample_points 中用于设置采样点 z）
    """

    wall_height: float = 1.4  # 对应室外场景
    # wall_height: float = 1.0  # 对应室内场景
    
    """Height of the walls.

    Wall filtering will start rays from that height and filter all that hit the mesh within 0.3m."""
    # 生效: 是（构建高度图与墙体/外侧过滤、门洞检测等垂直射线起点高度）

    robot_buffer_spawn: float = 0.5
    """Robot buffer for spawn location

    采样点与墙/障碍最小安全距离（水平多方向射线检查）。
    - 生效: 是（_point_filter_wall_closeness）
    """

    indoor_distance_threshold: float = 10.0
    """Threshold distance to consider a point as indoor
    If the distance to the nearest wall is less than this threshold, the point is considered indoor.
    - 生效: 是（_point_filter_indoor）
    """

    filter_indoor: bool = True

    sample_points: int = 1000
    """Number of nodes in the tree

    地形分析阶段在地面上采样的节点数（图的节点数上限）。
    - 生效: 是（_sample_points / 构图与最短路）
    """

    max_path_length: float = 10.0
    """Maximum distance from the start location to the goal location

    构图后最短路搜索的截断距离（仅统计不超过该距离的对）。
    - 生效: 是（all_pairs_dijkstra_path_length 截止）
    """

    height_diff_edge_filter: bool = False
    """Filter investigated edges in the height difference filter is both are on the same height. Default is False.

    If True, the height difference filter will only be applied if the two points are on different heights.
    This can lead to a speed up if the graph is large. If False, the height difference filter will be applied to all.
    - 生效: 是（_edge_filter_height_diff 里决定是否仅在不同高度上检查网格高度差）
    """

    door_filtering: bool = False
    """Account for doors when doing the height difference based edge filtering. Default is False.

    Normally, the height of the terrain is just determined by top-down raycasting. If True, there will be an additional
    raycasting 0.1m above the ground. If a upward pointing ray does not yield the same height as the top-down ray, the
    algorithms assumes that there is a door and a new height is determined.
    - 生效: 是（construct_height_map 门洞二次扫描分支）
    """

    door_height_threshold: float = 1.5
    """Threshold of the door height for the door detection.

    As some objects are composed out of multiple layers of meshes (e.g. stairs as combination of boxes), a door will be
    identified as a height difference of the top-down ray and the upward ray of at least this threshold.
    - 生效: 是（construct_height_map 门洞替换高度的阈值）
    """

    num_connections: int = 5
    """Number of connections to make in the graph

    KDTree 最近邻的连接数（每点保留的候选边数量）。
    - 生效: 是（_construct_graph / _edge_filter_*）
    """

    raycaster_sensor: str | None = None
    """Name of the raycaster sensor to use for terrain analysis.

    If None, the terrain analysis will be done on the USD stage. For matterport environments,
    the IsaacLab raycaster sensor can be used as the ply mesh is a single mesh. On the contrary,
    for unreal engine meshes (as they consists out of multiple meshes), raycasting should be
    performed over the USD stage. Default is None.
    - 生效: 是（_setup_raycaster 选择使用 RayCaster 还是 USD 全场景 PhysX 射线）
    """

    # 如果场景小可以使用更高的分辨率
    grid_resolution: float = 0.5
    """Resolution of the grid to check for not traversable edges

    高度图与代价网格的分辨率（米）。
    - 生效: 是（construct_height_map / 语义代价网格 / 高度差边检查采样）
    """

    height_diff_threshold: float = 0.3
    """Threshold for height difference between two points

    网格高度差判定阈值（超过则判定为不可通行边）。
    - 生效: 是（_edge_filter_height_diff）
    """

    viz_graph: bool = True
    """Visualize the graph after the construction for a short amount of time.

    - 生效: 是（构图完成后的调试可视化）
    """

    viz_height_map: bool = True
    """Visualize the height map after the construction for a short amount of time.

    - 生效: 是（高度图构建后的调试可视化）
    """

    # semantic_cost_mapping: object | None = None
    semantic_cost_mapping: dict[str, float] | None = {
        "road": 0.3,   # 道路，代价 1.5
        "sidewalk": 0.0, 
        "crosswalk": 0.0,
        "floor": 0.0,
        "vehicle": 2.0,
        "building": 2.0,
        "wall": 2.0,    # 语义标签为 "wall" 的物体，代价为 1.0
        "fence": 2.0,
        "pole": 2.0,
        "traffic_sign": 2.0,
        "traffic_light": 2.0,
        "bench": 2.0,
        "vegetation": 1.0,
        "terrain": 1.0,
        "water_surface": 2.0,
        "sky": 2.0,
        "dynamic": 2.0,
        "static": 2.0,
        "furniture": 1.0,
    }
    """Mapping of semantic categories to costs for filtering edges and nodes

    语义类别到代价的映射（越高越不可取）。
    - 生效: 是（节点与边的语义代价过滤；仅当非 None 时启用）
    """

    semantic_cost_threshold: float = 0.5
    """Threshold for semantic cost filtering

    语义代价阈值（超过阈值的点/边将被过滤）。
    - 生效: 是（_point_filter_semantic_cost / _edge_filter_semantic_cost）
    """

    # dimension limiting
    dim_limiter_prim: str | None = None
    """Prim name that should be used to limit the dimensions of the mesh.

    All meshes including this prim string are used to set the range in which the graph is constructed and samples are
    generated. If None, all meshes are considered.

    .. note::
        Only used if not a raycaster sensor is passed to the terrain analysis.
    - 生效: 是（当未使用 raycaster_sensor 时，限制参与计算的网格范围）
    """

    max_terrain_size: float | None = None
    """Maximum size of the terrain in meters.

    This can be useful when e.g. a ground plan is given and the entire anlaysis would run out of memory. If None, the
    entire terrain is considered.

    - 生效: 是（_setup_raycaster 里用于裁剪过大的场景范围）
    """
