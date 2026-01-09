#!/usr/bin/env python3
"""
Build a traversability grid by combining geometry (height/slope/obstacles)
and semantics from floor-only warp mesh.

This version uses a dedicated floor-only warp mesh (built from YAML keywords)
to generate semantic labels independently of PhysX scene queries.

Outputs: 
  - grid.npy (0/1): geometric traversability
  - height.npy (meters): surface height
  - sem_mask.npy (0/1 or multi-class): semantic labels
  - grid.png, height.png, sem_mask.png: visualizations
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Ensure the omni.viplanner package is importable from the repo extension folder
_EXT_ROOT = Path(__file__).resolve().parent.parent / "extension"
_OMNI_VIPLANNER_PKG = (_EXT_ROOT / "omni.viplanner").as_posix()
if _OMNI_VIPLANNER_PKG not in sys.path:
    sys.path.insert(0, _OMNI_VIPLANNER_PKG)
_EXT_ROOT_POSIX = _EXT_ROOT.as_posix()
if _EXT_ROOT_POSIX not in sys.path:
    sys.path.append(_EXT_ROOT_POSIX)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traversability grid demo with floor-only warp mesh semantics")
    p.add_argument("--scene", default="warehouse", choices=["warehouse", "carla", "matterport"], help="Scene")
    p.add_argument("--grid_res", type=float, default=0.25, help="Grid resolution (meters)")
    p.add_argument("--slope_deg", type=float, default=15.0, help="Max slope degrees considered traversable")
    # 表达式：最小距离墙壁的缓冲区
    p.add_argument("--buffer", type=float, default=0.4, help="Min distance from walls (meters)")
    p.add_argument("--seed", type=int, default=1, help="Random seed (for internal sampling consistency)")
    p.add_argument("--save_dir", type=str, default=None, help="Output directory")
    
    # Sampling region parameters
    p.add_argument("--offset_x", type=float, default=-6.0, help="X offset from scene center (meters)")
    p.add_argument("--offset_y", type=float, default=12.0, help="Y offset from scene center (meters)")
    p.add_argument("--capture_size", type=float, default=8.0, help="Size of capture region (meters)")
    p.add_argument("--camera_height", type=float, default=9.0, help="Camera height above ground (meters)")
    
    # Isaac app args
    AppLauncher.add_app_launcher_args(p)
    return p


def make_scene(scene_name: str):
    from importlib import util as importlib_util
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    import types as _types

    _pkg_omni_dir = (_EXT_ROOT / "omni.viplanner" / "omni").as_posix()
    _pkg_viplanner_dir = (_EXT_ROOT / "omni.viplanner" / "omni" / "viplanner").as_posix()
    _pkg_config_dir = (_EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config").as_posix()
    if "omni" not in sys.modules:
        _omni = _types.ModuleType("omni")
        _omni.__path__ = [_pkg_omni_dir]
        sys.modules["omni"] = _omni
    if "omni.viplanner" not in sys.modules:
        _ov = _types.ModuleType("omni.viplanner")
        _ov.__path__ = [_pkg_viplanner_dir]
        sys.modules["omni.viplanner"] = _ov
    if "omni.viplanner.config" not in sys.modules:
        _ovc = _types.ModuleType("omni.viplanner.config")
        _ovc.__path__ = [_pkg_config_dir]
        sys.modules["omni.viplanner.config"] = _ovc

    cfg_root = Path(_pkg_config_dir)
    mod_name = f"omni.viplanner.config.{scene_name}_cfg"
    mod_path = (cfg_root / f"{scene_name}_cfg.py").as_posix()
    spec = importlib_util.spec_from_file_location(mod_name, mod_path)
    assert spec and spec.loader
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    SceneCfg = getattr(mod, "TerrainSceneCfg")

    scene_cfg = SceneCfg(1, env_spacing=1.0)
    scene_cfg.robot = None
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None
    if hasattr(scene_cfg, "depth_camera"):
        scene_cfg.depth_camera = None
    # Keep semantic_camera for RGB rendering
    # if hasattr(scene_cfg, "semantic_camera"):
    #     scene_cfg.semantic_camera = None

    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return scene, sim


def render_top_down_rgb_local(sim, scene, center_x, center_y, size, camera_height, output_path, resolution=512):
    """
    Render a top-down RGB view for a specific local region
    """
    try:
        # Check if scene has semantic camera with RGB capability
        if not hasattr(scene, 'sensors') or 'semantic_camera' not in scene.sensors:
            print("[WARNING] No semantic_camera found in scene")
            return False
        
        camera = scene.sensors['semantic_camera']
        
        # Check if camera has RGB data type
        if 'rgb' not in camera.data.output:
            print("[WARNING] semantic_camera does not have RGB data")
            return False
        
        print(f"[INFO] Capturing top-down RGB for local region...")
        print(f"  Position: ({center_x:.1f}, {center_y:.1f}, {camera_height:.1f})")
        print(f"  Coverage: {size:.1f}m x {size:.1f}m")
        
        # Create pose: position + orientation (looking straight down)
        import torch
        from isaaclab.utils import math as math_utils
        
        # Position
        position = torch.tensor([[center_x, center_y, camera_height]], dtype=torch.float32, device="cuda:0")
        
        # Orientation for TRUE top-down view: rotate +90 degrees around Y axis
        x_angle = torch.tensor([0.0], dtype=torch.float32, device="cpu")
        y_angle = torch.tensor([90.0], dtype=torch.float32, device="cpu")
        z_angle = torch.tensor([0.0], dtype=torch.float32, device="cpu")
        
        orientation = math_utils.quat_from_euler_xyz(
            torch.deg2rad(x_angle), 
            torch.deg2rad(y_angle), 
            torch.deg2rad(z_angle)
        ).to("cuda:0")
        
        print(f"[DEBUG] Camera angles (deg): pitch={x_angle.item():.1f}, roll={y_angle.item():.1f}, yaw={z_angle.item():.1f}")
        
        # Set camera pose
        camera.set_world_poses(
            positions=position,
            orientations=orientation,
            env_ids=torch.tensor([0], device="cuda:0"),
            convention="world",
        )
        
        # Write to sim and render multiple times
        scene.write_data_to_sim()
        for _ in range(10):
            sim.render()
        scene.update(sim.get_physics_dt())
        
        # Get RGB data
        rgb_data = camera.data.output["rgb"][0]
        
        print(f"[DEBUG] RGB data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}, range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
        
        # Convert to numpy
        rgb_array = rgb_data.cpu().numpy()
        
        # Convert from float [0, 1] to uint8 [0, 255] if needed
        if rgb_array.dtype in [np.float32, np.float64]:
            rgb_array = (np.clip(rgb_array, 0, 1) * 255).astype(np.uint8)
        elif rgb_array.dtype == np.uint8:
            pass
        
        # Take RGB channels
        if rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
            rgb_img = rgb_array[:, :, :3]
        else:
            rgb_img = rgb_array
        
        print(f"[DEBUG] Final image: {rgb_img.shape}, dtype: {rgb_img.dtype}, range: [{rgb_img.min()}, {rgb_img.max()}]")
        
        # Resize to target resolution
        if rgb_img.shape[0] != resolution or rgb_img.shape[1] != resolution:
            from PIL import Image
            rgb_img = np.array(Image.fromarray(rgb_img).resize((resolution, resolution)))
        
        # Save
        from PIL import Image
        Image.fromarray(rgb_img).save(output_path)
        print(f"[INFO] ✓ Saved top-down RGB to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Failed to render top-down RGB: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_top_down_rgb(sim, scene, bounds, output_path, resolution=1024):
    """
    Render a top-down RGB view using the existing semantic camera
    Following data_collect_myself.py's approach
    """
    try:
        # Check if scene has semantic camera with RGB capability
        if not hasattr(scene, 'sensors') or 'semantic_camera' not in scene.sensors:
            print("[WARNING] No semantic_camera found in scene")
            return False
        
        camera = scene.sensors['semantic_camera']
        
        # Check if camera has RGB data type
        if 'rgb' not in camera.data.output:
            print("[WARNING] semantic_camera does not have RGB data")
            return False
        
        x_min, x_max, y_min, y_max = bounds
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Offset position to capture a different area with more objects
        # Try different corner area
        offset_x = -11.0  # Move -11m in X direction (west)
        offset_y = -11.0  # Move -11m in Y direction (south)
        capture_x = center_x + offset_x
        capture_y = center_y + offset_y
        
        # Use a smaller area for better visibility: 5m x 5m
        capture_size = 5.0  # meters
        camera_height = 6.0  # 6m above ground
        
        print(f"[INFO] Capturing top-down RGB using semantic camera...")
        print(f"  Position: ({capture_x:.1f}, {capture_y:.1f}, {camera_height:.1f})")
        print(f"  Offset from center: ({offset_x:.1f}, {offset_y:.1f})")
        print(f"  Coverage: {capture_size:.1f}m x {capture_size:.1f}m")
        print(f"  Full bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
        
        # Create pose: position + orientation (looking straight down)
        import torch
        from isaaclab.utils import math as math_utils
        
        # Position (using offset position instead of center)
        position = torch.tensor([[capture_x, capture_y, camera_height]], dtype=torch.float32, device="cuda:0")
        
        # Orientation for TRUE top-down view:
        # To look straight down, rotate +90 degrees around Y axis
        # x_angle = 0 (no pitch relative to horizontal)
        # y_angle = +90 deg (rotate camera to point down)
        # z_angle = 0 (no yaw)
        x_angle = torch.tensor([0.0], dtype=torch.float32, device="cpu")      # pitch: keep horizontal
        y_angle = torch.tensor([90.0], dtype=torch.float32, device="cpu")     # roll: rotate to look down
        z_angle = torch.tensor([0.0], dtype=torch.float32, device="cpu")      # yaw: facing forward
        
        x_angle_rad = torch.deg2rad(x_angle)
        y_angle_rad = torch.deg2rad(y_angle)
        z_angle_rad = torch.deg2rad(z_angle)
        
        # Convert to quaternion using IsaacLab's function (same as data_collect_myself.py)
        orientation = math_utils.quat_from_euler_xyz(x_angle_rad, y_angle_rad, z_angle_rad).to("cuda:0")
        
        print(f"[DEBUG] Camera angles (deg): pitch={x_angle.item():.1f}, roll={y_angle.item():.1f}, yaw={z_angle.item():.1f}")
        print(f"[DEBUG] Camera orientation (wxyz): {orientation}")
        
        # Set camera pose (like data_collect_myself.py does)
        camera.set_world_poses(
            positions=position,
            orientations=orientation,
            env_ids=torch.tensor([0], device="cuda:0"),
            convention="world",
        )
        
        # Write to sim and render multiple times (following data_collect_myself.py)
        scene.write_data_to_sim()
        for _ in range(10):
            sim.render()
        scene.update(sim.get_physics_dt())
        
        # Get RGB data (like data_collect_myself.py)
        rgb_data = camera.data.output["rgb"][0]  # First environment
        
        # Debug: print data info
        print(f"[DEBUG] RGB data shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
        print(f"[DEBUG] RGB data range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
        
        # Convert to numpy
        rgb_array = rgb_data.cpu().numpy()
        
        # Convert from float [0, 1] to uint8 [0, 255] if needed
        if rgb_array.dtype in [np.float32, np.float64]:
            rgb_array = (np.clip(rgb_array, 0, 1) * 255).astype(np.uint8)
        elif rgb_array.dtype == np.uint8:
            pass  # Already uint8
        
        # Take RGB channels (ignore alpha if present)
        if rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
            rgb_img = rgb_array[:, :, :3]
        else:
            rgb_img = rgb_array
        
        print(f"[DEBUG] Final image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}, range: [{rgb_img.min()}, {rgb_img.max()}]")
        
        # Resize to target resolution if needed
        if rgb_img.shape[0] != resolution or rgb_img.shape[1] != resolution:
            from PIL import Image
            rgb_img = np.array(Image.fromarray(rgb_img).resize((resolution, resolution)))
        
        # Save
        from PIL import Image
        Image.fromarray(rgb_img).save(output_path)
        print(f"[INFO] ✓ Saved top-down RGB to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Failed to render top-down RGB: {e}")
        import traceback
        traceback.print_exc()
        return False


def compute_local_maps(scene, center_x, center_y, size, grid_res):
    """
    Generate local height and semantic maps for a specific region
    
    Args:
        scene: InteractiveScene
        center_x, center_y: Center of the region
        size: Size of the region (meters)
        grid_res: Grid resolution (meters)
    
    Returns:
        dict with 'height', 'sem_mask', 'bounds', etc.
    """
    from importlib import util as importlib_util
    from pathlib import Path as _Path
    import types as _types
    import sys as _sys

    _collectors_dir = _Path(__file__).resolve().parent.parent / "extension" / "omni.viplanner" / "omni" / "viplanner" / "collectors"
    if "omni.viplanner.collectors" not in _sys.modules:
        _pkg_collectors = _types.ModuleType("omni.viplanner.collectors")
        _pkg_collectors.__path__ = [_collectors_dir.as_posix()]
        _sys.modules["omni.viplanner.collectors"] = _pkg_collectors

    _cfg_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_cfg",
        (_collectors_dir / "terrain_analysis_cfg.py").as_posix(),
    )
    assert _cfg_spec and _cfg_spec.loader
    _cfg_mod = importlib_util.module_from_spec(_cfg_spec)
    _sys.modules[_cfg_spec.name] = _cfg_mod
    _cfg_spec.loader.exec_module(_cfg_mod)
    TerrainAnalysisCfg = getattr(_cfg_mod, "TerrainAnalysisCfg")

    _ta_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_myself",
        (_collectors_dir / "terrain_analysis_myself.py").as_posix(),
    )
    assert _ta_spec and _ta_spec.loader
    _ta_mod = importlib_util.module_from_spec(_ta_spec)
    _sys.modules[_ta_spec.name] = _ta_mod
    _ta_spec.loader.exec_module(_ta_mod)
    TerrainAnalysis = getattr(_ta_mod, "TerrainAnalysis")

    from isaaclab.utils.warp import raycast_mesh
    import torch

    # Setup terrain analysis with specified grid resolution
    tac = TerrainAnalysisCfg()
    tac.grid_resolution = grid_res
    tac.semantic_cost_mapping = None

    ta = TerrainAnalysis(tac, scene)
    ta._setup_raycaster()

    # Define local bounds
    half_size = size / 2.0
    x_min = center_x - half_size
    x_max = center_x + half_size
    y_min = center_y - half_size
    y_max = center_y + half_size

    # Create grid for local region with exact resolution
    # Add 1 to include both endpoints: [0, 0.1, 0.2, ..., 5.0]
    num_points = int(np.round(size / grid_res)) + 1
    xs = np.linspace(x_min, x_max, num_points)
    ys = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    
    actual_res_x = (xs[1] - xs[0]) if len(xs) > 1 else grid_res
    actual_res_y = (ys[1] - ys[0]) if len(ys) > 1 else grid_res

    print(f"[INFO] Local map grid: {num_points}x{num_points} points")
    print(f"[INFO] Requested resolution: {grid_res}m, actual: x={actual_res_x:.4f}m, y={actual_res_y:.4f}m")
    print(f"[INFO] Local bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    # Use a much higher starting point for raycast to ensure we're above everything
    # 这个地方这么写的原因是：warehouse场景下只存在地面、货架和墙壁，而地面以上3米的高度范围足够机器人行走
    raycast_start_height = 3.0  # Start from 3m above ground
    raycast_max_dist = 3.1      # Allow rays to travel 3.1m down
    
    print(f"[DEBUG] Step 1: Raycast DOWN to find ground height (multi-sample)")
    print(f"[DEBUG]   start_height={raycast_start_height}m, max_dist={raycast_max_dist}m")

    # Step 1: Multi-sample raycast DOWN to find ground height
    # Use 5 rays per point with small offsets to avoid mesh gaps
    # Use larger offsets (0.05m) to better cover 0.1m grid resolution
    # 只设定一个点扫描
    ray_offsets = [
        (0.0, 0.0),      # center
        # (0.05, 0.0),     # right
        # (-0.05, 0.0),    # left
        # (0.0, 0.05),     # forward
        # (0.0, -0.05),    # backward
    ]
    
    all_ground_hits = []
    for offset_x, offset_y in ray_offsets:
        grid_points_down = torch.from_numpy(
            np.column_stack([
                X.flatten() + offset_x, 
                Y.flatten() + offset_y, 
                np.ones_like(X.flatten()) * raycast_start_height
            ])
        ).float().to(ta.device)
        direction_down = torch.zeros_like(grid_points_down)
        direction_down[:, 2] = -1.0  # Point down
        # CRITICAL: raycast_mesh 只返回第一个命中点！
        # 从3m向下→命中[0m,3m]范围内的第一个表面(可能是货架顶部、箱子、或地面)
        # 多层建筑场景：会跳过起点下方的其他楼层,只返回最先遇到的表面
        ground_hit = raycast_mesh(
            ray_starts=grid_points_down.unsqueeze(0),
            ray_directions=direction_down.unsqueeze(0),
            max_dist=raycast_max_dist,
            mesh=ta._warp_mesh,
        )[0].squeeze(0)

        all_ground_hits.append(ground_hit[:, 2].cpu().numpy())
    
    # Take median of all samples to be robust
    all_ground_hits = np.array(all_ground_hits)  # shape: (5, num_points)
    ground_height = np.nanmedian(all_ground_hits, axis=0)  # Ground height for each point
    
    # Debug ground height
    valid_ground = ground_height[np.isfinite(ground_height)]
    if len(valid_ground) > 0:
        print(f"[DEBUG]   Ground found: {len(valid_ground)}/{len(ground_height)} points ({100*len(valid_ground)/len(ground_height):.1f}%)")
        print(f"[DEBUG]   Ground height range: [{valid_ground.min():.2f}, {valid_ground.max():.2f}] meters")
    else:
        print(f"[WARNING]   No ground found! All rays missed.")
        # If no ground, return empty maps
        return {
            "height": np.full(X.shape, np.nan),
            "sem_mask": np.zeros(X.shape, dtype=np.uint8),
            "bounds": (x_min, x_max, y_min, y_max),
            "grid_res": grid_res,
            "used_semantics": False,
        }
    
    # obstacle_scan_height = 2.48   # Scan 3m above ground for obstacles
    # ground_offset = 2.5         # Start 0.02m above ground to avoid noise
    
    # Step 2: Multi-sample raycast UP from ground to detect obstacles within 3m height
    # print(f"[DEBUG] Step 2: Raycast UP from ground to detect obstacles (multi-sample)")
    # print(f"[DEBUG]   scan_height={obstacle_scan_height}m, ground_offset={ground_offset}m")
    
    # Step 2: Multi-sample raycast UP from ground to detect obstacles within 3m height
    # print(f"[DEBUG] Step 2: Raycast UP from ground to detect obstacles (multi-sample)")
    # print(f"[DEBUG]   scan_height={obstacle_scan_height}m, ground_offset={ground_offset}m")
    
    # 根据上一步测量的地面高度确定扫描起点
    # For each grid point, start from ground + 0.02m and shoot upward
    # Use multi-sampling to avoid mesh gaps
    # all_obstacle_hits = []
    # for offset_x, offset_y in ray_offsets:
    #     ray_start_z = ground_height + ground_offset
    #     grid_points_up = torch.from_numpy(
    #         np.column_stack([
    #             X.flatten() + offset_x, 
    #             Y.flatten() + offset_y, 
    #             ray_start_z
    #         ])
    #     ).float().to(ta.device)
    #     direction_up = torch.zeros_like(grid_points_up)
    #     direction_up[:, 2] = -1.0  # Point DOWN (not UP! Fixed direction)

    #     # CRITICAL: raycast_mesh 只返回第一个命中点！
    #     # 从地面+2.5m向下→命中第一个遇到的障碍物(货架底部、墙壁等)
    #     # 注意：这不是"最高点",而是从起点沿射线方向的第一个交点
    #     obstacle_hit = raycast_mesh(
    #         ray_starts=grid_points_up.unsqueeze(0),
    #         ray_directions=direction_up.unsqueeze(0),
    #         max_dist=obstacle_scan_height,
    #         mesh=ta._warp_mesh,
    #     )[0].squeeze(0)

    #     all_obstacle_hits.append(obstacle_hit[:, 2].cpu().numpy())
    
    # Take median of obstacle heights
    # all_obstacle_hits = np.array(all_obstacle_hits)  # shape: (5, num_points)
    # obstacle_height = np.nanmedian(all_obstacle_hits, axis=0)
    # has_any_obstacle = np.isfinite(obstacle_height)
    
    # Process height map: use obstacle height if hit, otherwise ground height
    # height_map = np.where(
    #     has_any_obstacle,
    #     obstacle_height,  # Hit obstacle: use obstacle height
    #     ground_height     # No obstacle: use ground height
    # ).reshape(X.shape)
    
    # Use ground height directly as the height map
    height_map = ground_height.reshape(X.shape)
    
    # Mark points without valid ground as having obstacles
    # has_any_obstacle = ~np.isfinite(ground_height)
    
    # Debug obstacle detection
    # print(f"[DEBUG]   Any obstacles detected: {has_any_obstacle.sum()}/{len(obstacle_height)} points ({100*has_any_obstacle.sum()/len(obstacle_height):.1f}%)")
    # if has_any_obstacle.any():
    #     obstacle_heights_valid = obstacle_height[has_any_obstacle]
    #     print(f"[DEBUG]   Obstacle height range: [{obstacle_heights_valid.min():.2f}, {obstacle_heights_valid.max():.2f}] meters")
    
    # Step 3: 语义检查 - 检测是否有障碍物遮挡
    # 使用障碍物mesh判断：击中障碍物=不可通行，未击中=可通行
    # print(f"[DEBUG] Step 3: Check for obstacles (raycast down to obstacle-only mesh)")
    
    has_obstacle = None
    if hasattr(ta, "_warp_mesh_obstacles") and ta._warp_mesh_obstacles is not None:
        # Multi-sample raycast DOWN to obstacle-only mesh
        # 从上往下raycast，检查是否击中障碍物
        all_obstacle_hits = []
        for offset_x, offset_y in ray_offsets:
            grid_points_down_obstacle = torch.from_numpy(
                np.column_stack([
                    X.flatten() + offset_x,
                    Y.flatten() + offset_y,
                    np.ones_like(X.flatten()) * raycast_start_height
                ])
            ).float().to(ta.device)
            direction_down_obstacle = torch.zeros_like(grid_points_down_obstacle)
            direction_down_obstacle[:, 2] = -1.0
            
            obstacle_hit_result = raycast_mesh(
                ray_starts=grid_points_down_obstacle.unsqueeze(0),
                ray_directions=direction_down_obstacle.unsqueeze(0),
                max_dist=raycast_max_dist,
                mesh=ta._warp_mesh_obstacles,
            )[0].squeeze(0)
            # 判断是否击中：如果 z 坐标是有限值（isfinite），说明击中了障碍物
            all_obstacle_hits.append(torch.isfinite(obstacle_hit_result[:, 2]).cpu().numpy())
        
        # A point has obstacle if ANY of the samples hit obstacle-only mesh
        
        # 这里上面的函数是一个类似于并行计算的核函数的形式，生成的数据类似于
        # all_obstacle_hits = [
        #     array([True, False, True, False, ...]),   # 第1次采样结果（offset_x=0.0, offset_y=0.0）
        #     array([False, False, True, False, ...]),  # 第2次采样结果（offset_x=0.05, offset_y=0.0）
        #     array([True, False, True, False, ...]),   # 第3次采样结果（offset_x=-0.05, offset_y=0.0）
        #     # ... 更多采样
        # ]
        # 其中每一列表示每一个栅个点，每一行表示同一个栅格点内不同采样点的结果
        # 转换后：变成 2D NumPy 数组
        # all_obstacle_hits = np.array([
        #     [True,  False, True,  False, ...],  # 采样1
        #     [False, False, True,  False, ...],  # 采样2
        #     [True,  False, True,  False, ...],  # 采样3
        #     # ...
        # ])
        all_obstacle_hits = np.array(all_obstacle_hits)  # shape: (num_samples, num_points)
        
        # 沿着 axis=0（第一个维度）进行逻辑或运算，只要有一个采样点检测到障碍物，就认为该栅格点有障碍物
        # all_obstacle_hits = np.array([
        #     # 点0   点1   点2
        #     [True,  False, True],   # 采样1
        #     [False, False, True],   # 采样2
        #     [True,  False, False],  # 采样3
        #     [False, False, True],   # 采样4
        #     [False, True,  False],  # 采样5
        # ])
        # has_obstacle = [True, True, True]
        has_obstacle = np.any(all_obstacle_hits, axis=0)
        print(f"[DEBUG]   Obstacles detected: {has_obstacle.sum()}/{len(has_obstacle)} points ({100*has_obstacle.sum()/len(has_obstacle):.1f}%)")
        
        # 可通行性 = 有ground且没有obstacle
        is_traversable = np.isfinite(ground_height) & ~has_obstacle
    else:
        print(f"[WARNING]   No obstacle-only mesh available, assuming all ground is traversable")
        # If no obstacle mesh, assume all points with valid ground are traversable
        is_traversable = np.isfinite(ground_height)
    
    # Step 4: Compute traversability mask
    # Logic: traversable if (has_ground=True AND no_obstacle=True)
    print(f"[DEBUG] Step 4: Computing traversability mask")
    sem_mask = is_traversable.astype(np.uint8).reshape(X.shape)
    print(f"[DEBUG]   Traversable points: {sem_mask.sum()}/{sem_mask.size} ({100*sem_mask.sum()/sem_mask.size:.1f}%)")
    
    return {
        "height": height_map,
        "sem_mask": sem_mask,
        "bounds": (x_min, x_max, y_min, y_max),
        "grid_res": grid_res,
        "used_semantics": True,
    }


def compute_traversability_and_semantics(
    scene,
    grid_res: float,
    slope_deg: float,
    buffer_m: float,
):
    from importlib import util as importlib_util
    from pathlib import Path as _Path
    import types as _types
    import sys as _sys

    _collectors_dir = _Path(__file__).resolve().parent.parent / "extension" / "omni.viplanner" / "omni" / "viplanner" / "collectors"
    if "omni.viplanner.collectors" not in _sys.modules:
        _pkg_collectors = _types.ModuleType("omni.viplanner.collectors")
        _pkg_collectors.__path__ = [_collectors_dir.as_posix()]
        _sys.modules["omni.viplanner.collectors"] = _pkg_collectors

    _cfg_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_cfg",
        (_collectors_dir / "terrain_analysis_cfg.py").as_posix(),
    )
    assert _cfg_spec and _cfg_spec.loader
    _cfg_mod = importlib_util.module_from_spec(_cfg_spec)
    _sys.modules[_cfg_spec.name] = _cfg_mod
    _cfg_spec.loader.exec_module(_cfg_mod)  # type: ignore
    TerrainAnalysisCfg = getattr(_cfg_mod, "TerrainAnalysisCfg")

    _ta_spec = importlib_util.spec_from_file_location(
        "omni.viplanner.collectors.terrain_analysis_myself",
        (_collectors_dir / "terrain_analysis_myself.py").as_posix(),
    )
    assert _ta_spec and _ta_spec.loader
    _ta_mod = importlib_util.module_from_spec(_ta_spec)
    _sys.modules[_ta_spec.name] = _ta_mod
    _ta_spec.loader.exec_module(_ta_mod)  # type: ignore
    TerrainAnalysis = getattr(_ta_mod, "TerrainAnalysis")

    from isaaclab.utils.warp import raycast_mesh

    tac = TerrainAnalysisCfg()
    tac.grid_resolution = grid_res
    tac.robot_buffer_spawn = buffer_m
    tac.semantic_cost_mapping = None

    ta = TerrainAnalysis(tac, scene)
    ta._setup_raycaster()
    ta.construct_height_map()

    hg = ta.height_grid
    x_max, y_max, x_min, y_min = ta.mesh_dimensions
    xs = np.linspace(x_min, x_max, hg.shape[0])
    ys = np.linspace(y_min, y_max, hg.shape[1])

    hg_np = hg.cpu().numpy()
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Skip grid traversability computation for now
    # Just focus on height and semantic mask
    
    # Extract warp mesh data for inspection
    warp_mesh_global = None
    warp_mesh_sem = None
    if hasattr(ta, "_warp_mesh") and ta._warp_mesh is not None:
        try:
            # Extract mesh vertices and faces from warp mesh
            warp_mesh_global = {
                "vertices": ta._warp_mesh.points.numpy() if hasattr(ta._warp_mesh.points, 'numpy') else None,
                "faces": ta._warp_mesh.indices.numpy() if hasattr(ta._warp_mesh.indices, 'numpy') else None,
            }
        except Exception as e:
            print(f"[WARNING] Failed to extract global warp mesh data: {e}")
    
    if hasattr(ta, "_warp_mesh_sem") and ta._warp_mesh_sem is not None:
        try:
            warp_mesh_sem = {
                "vertices": ta._warp_mesh_sem.points.numpy() if hasattr(ta._warp_mesh_sem.points, 'numpy') else None,
                "faces": ta._warp_mesh_sem.indices.numpy() if hasattr(ta._warp_mesh_sem.indices, 'numpy') else None,
            }
        except Exception as e:
            print(f"[WARNING] Failed to extract semantic warp mesh data: {e}")

    sem_mask = None
    used_semantics = False
    if hasattr(ta, "_warp_mesh_sem") and ta._warp_mesh_sem is not None:
        try:
            grid_points = torch.from_numpy(
                np.column_stack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * (tac.wall_height * 2)])
            ).float().to(ta.device)
            direction = torch.zeros_like(grid_points)
            direction[:, 2] = -1.0

            hit_point = raycast_mesh(
                ray_starts=grid_points.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=tac.wall_height * 2,
                mesh=ta._warp_mesh_sem,
            )[0].squeeze(0)

            sem_hit = torch.isfinite(hit_point[:, 2]).cpu().numpy()
            sem_mask = sem_hit.reshape(X.shape).astype(np.uint8)
            used_semantics = True

            if os.environ.get("VIPLANNER_DEBUG_SEM", "0") == "1":
                print(f"[VIPLANNER_DEBUG] Semantic warp mesh raycast: total={sem_hit.size}, hit_count={sem_hit.sum()}")
                if hasattr(ta, "_warp_mesh_sem_class") and ta._warp_mesh_sem_class:
                    print(f"[VIPLANNER_DEBUG] Semantic class: {ta._warp_mesh_sem_class}")
        except Exception as e:
            print(f"[WARNING] Semantic warp mesh raycast failed: {e}")
            used_semantics = False

    return {
        "height": hg_np,
        "xs": xs,
        "ys": ys,
        "used_semantics": used_semantics,
        "sem_mask": sem_mask,
        "warp_mesh_global": warp_mesh_global,
        "warp_mesh_sem": warp_mesh_sem,
        "bounds": (x_min, x_max, y_min, y_max),
    }


def main():
    # 读取参数
    parser = build_parser()
    args = parser.parse_args()

    # 
    os.environ.setdefault("VIPLANNER_DISABLE_PEOPLE", "1")
    app = AppLauncher(args).app

    scene, sim = make_scene(args.scene)

    # Use sampling parameters from command line arguments
    # This is the SINGLE place to change sampling location
    offset_x = args.offset_x
    offset_y = args.offset_y
    capture_size = args.capture_size
    camera_height = args.camera_height
    
    # Note: In warehouse scene, center is (0, 0)
    capture_x = 0.0 + offset_x
    capture_y = 0.0 + offset_y

    print(f"\n{'='*60}")
    print(f"[INFO] Generating LOCAL maps for sampling region")
    print(f"{'='*60}")
    print(f"  Center: ({capture_x:.1f}, {capture_y:.1f})")
    print(f"  Offset: ({offset_x:.1f}, {offset_y:.1f})")
    print(f"  Size: {capture_size:.1f}m x {capture_size:.1f}m")
    print(f"  Camera height: {camera_height:.1f}m")
    print(f"  Grid resolution: {args.grid_res}m")
    print(f"{'='*60}\n")

    # Generate local height and semantic maps
    result = compute_local_maps(scene, capture_x, capture_y, capture_size, args.grid_res)

    out_dir = args.save_dir or os.environ.get("VIPLANNER_DATA_DIR") or str(Path.cwd() / "traversability_output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save local maps
    np.save(os.path.join(out_dir, "height.npy"), result["height"])
    if result.get("sem_mask") is not None:
        np.save(os.path.join(out_dir, "sem_mask.npy"), result["sem_mask"])
    
    # Render top-down RGB view for the SAME region
    if result.get("bounds") is not None:
        rgb_path = os.path.join(out_dir, "top_down_rgb.png")
        # Pass the sampling center and size to RGB rendering
        render_top_down_rgb_local(sim, scene, capture_x, capture_y, capture_size, camera_height, rgb_path, resolution=512)
    
    # Generate visualizations
    try:
        import cv2
        
        # Height map PNG
        hg = result.get("height")
        if hg is not None:
            import numpy as _np
            hg_np = _np.array(hg)
            finite = _np.isfinite(hg_np)
            if finite.any():
                hmin = _np.nanmin(hg_np[finite])
                hmax = _np.nanmax(hg_np[finite])
                denom = (hmax - hmin) if (hmax > hmin) else 1.0
                norm = _np.clip((hg_np - hmin) / denom, 0.0, 1.0)
                norm[~finite] = 0.0
                height_img = (norm * 255).astype(_np.uint8)
                cv2.imwrite(os.path.join(out_dir, "height.png"), height_img)
        
        # Semantic mask PNG
        if result.get("sem_mask") is not None:
            sm = (result["sem_mask"] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, "sem_mask.png"), sm)
            np.save(os.path.join(out_dir, "sem_mask.npy"), result["sem_mask"])
    except Exception as e:
        print(f"[WARNING] Failed to generate PNG files: {e}")
        pass
    
    # Create success flag file
    success_flag_path = os.path.join(out_dir, "SUCCESS.txt")
    with open(success_flag_path, "w") as f:
        f.write("=== Height & Semantic Data Generation Success ===\n")
        f.write(f"Timestamp: {args.seed}\n")
        f.write(f"Scene: {args.scene}\n")
        f.write(f"Grid resolution: {args.grid_res}m\n")
        f.write(f"Data shape: {result['height'].shape}\n")
        f.write(f"\n--- Generated Files ---\n")
        f.write("✓ height.npy - Surface height map (meters)\n")
        if result.get("sem_mask") is not None:
            f.write("✓ sem_mask.npy - Semantic floor mask (0/1)\n")
        else:
            f.write("✗ sem_mask.npy - Not generated (no semantic mesh)\n")
        f.write("✓ height.png - Height map visualization\n")
        if result.get("sem_mask") is not None:
            f.write("✓ sem_mask.png - Semantic mask visualization\n")
        if result.get("warp_mesh_global") is not None:
            f.write("✓ warp_mesh_global.npy - Global geometry mesh data\n")
        if result.get("warp_mesh_sem") is not None:
            f.write("✓ warp_mesh_sem.npy - Semantic floor-only mesh data\n")
        f.write(f"\n--- Status ---\n")
        f.write(f"Used semantics: {result['used_semantics']}\n")
        if not result['used_semantics']:
            f.write("Note: Semantic filtering not applied. Check VIPLANNER_SEMANTIC_MAP environment variable.\n")
        else:
            f.write("Semantic mask successfully generated from floor-only warp mesh.\n")
    
    print("\n" + "="*60)
    print(f"[SUCCESS] All data saved to: {out_dir}")
    print("="*60)
    print(f"✓ height.npy        - Height map")
    if result.get("sem_mask") is not None:
        print(f"✓ sem_mask.npy      - Semantic floor mask")
    if result.get("warp_mesh_global") is not None:
        print(f"✓ warp_mesh_global.npy - Global geometry mesh")
    if result.get("warp_mesh_sem") is not None:
        print(f"✓ warp_mesh_sem.npy    - Semantic floor mesh")
    print(f"✓ SUCCESS.txt       - Generation report")
    print("="*60)
    print(f"used_semantics={result['used_semantics']}  |  shape={result['height'].shape}")
    if not result['used_semantics']:
        print("[HINT] Semantics not applied. Ensure VIPLANNER_SEMANTIC_MAP is set.")
    print("="*60 + "\n")

    app.close()


if __name__ == "__main__":
    main()
