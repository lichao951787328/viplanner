#!/usr/bin/env python3
"""
Build traversability grid with ROTATED view (random yaw angle)

Key features:
- Fixed pitch = 90° (top-down view)
- Random yaw angle (rotation around Z-axis)
- Sample in LOCAL rotated coordinate frame
- Transform local cells to GLOBAL coordinates for raycast
- Generate: height map, semantic mask, RGB image (all in rotated view)

Outputs:
  - height.npy: Height map in rotated view
  - sem_mask.npy: Semantic traversability mask
  - top_down_rgb.png: RGB image from rotated camera
  - camera_pose.npy: Camera pose (position + orientation)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# Ensure the omni.viplanner package is importable
_EXT_ROOT = Path(__file__).resolve().parent.parent / "extension"
_OMNI_VIPLANNER_PKG = (_EXT_ROOT / "omni.viplanner").as_posix()
if _OMNI_VIPLANNER_PKG not in sys.path:
    sys.path.insert(0, _OMNI_VIPLANNER_PKG)
_EXT_ROOT_POSIX = _EXT_ROOT.as_posix()
if _EXT_ROOT_POSIX not in sys.path:
    sys.path.append(_EXT_ROOT_POSIX)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traversability grid with rotated view")
    p.add_argument("--scene", default="warehouse", choices=["warehouse", "carla", "matterport"], help="Scene")
    p.add_argument("--grid_res", type=float, default=0.1, help="Grid resolution (meters)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (None=random yaw)")
    p.add_argument("--save_dir", type=str, default=None, help="Output directory")
    
    # Sampling region parameters
    p.add_argument("--offset_x", type=float, default=-7.0, help="X offset from scene center (meters)")
    p.add_argument("--offset_y", type=float, default=0.0, help="Y offset from scene center (meters)")
    p.add_argument("--capture_size", type=float, default=8.0, help="Size of capture region (meters)")
    p.add_argument("--camera_height", type=float, default=8.0, help="Camera height above ground (meters)")
    
    # Rotation parameters
    p.add_argument("--yaw_deg", type=float, default=None, help="Yaw angle in degrees (None=random)")
    p.add_argument("--yaw_min", type=float, default=0.0, help="Min yaw angle for random (degrees)")
    p.add_argument("--yaw_max", type=float, default=360.0, help="Max yaw angle for random (degrees)")
    
    # Isaac app args
    AppLauncher.add_app_launcher_args(p)
    return p


def make_scene(scene_name: str):
    """Create Isaac Sim scene"""
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
    spec.loader.exec_module(mod)
    SceneCfg = getattr(mod, "TerrainSceneCfg")

    scene_cfg = SceneCfg(1, env_spacing=1.0)
    scene_cfg.robot = None
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None
    if hasattr(scene_cfg, "depth_camera"):
        scene_cfg.depth_camera = None

    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return scene, sim


def rotation_matrix_z(yaw_rad):
    """
    Create 2D rotation matrix for Z-axis rotation
    
    Args:
        yaw_rad: Yaw angle in radians
    
    Returns:
        2x2 rotation matrix
    """
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

def rot_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (x, y, z, w)
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Quaternion as (x, y, z, w)
    """
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qx, qy, qz, qw])

def render_rotated_rgb(sim, scene, center_x, center_y, camera_height, yaw_deg, output_path, resolution=512):
    """
    Render RGB image from rotated camera view
    
    Args:
        sim: SimulationContext
        scene: InteractiveScene
        center_x, center_y: Camera position in XY plane
        camera_height: Camera height (Z coordinate)
        yaw_deg: Yaw angle in degrees (rotation around Z-axis)
        output_path: Output image path
        resolution: Image resolution
    """
    try:
        if not hasattr(scene, 'sensors') or 'semantic_camera' not in scene.sensors:
            print("[WARNING] No semantic_camera found in scene")
            return False
        
        camera = scene.sensors['semantic_camera']
        
        if 'rgb' not in camera.data.output:
            print("[WARNING] semantic_camera does not have RGB data")
            return False
        
        print(f"[INFO] Rendering RGB from rotated camera...")
        print(f"  Position: ({center_x:.1f}, {center_y:.1f}, {camera_height:.1f})")
        print(f"  Yaw angle: {yaw_deg:.1f}°")
        
        import torch
        from isaaclab.utils import math as math_utils
        
        # Camera position
        position = torch.tensor([[center_x, center_y, camera_height]], dtype=torch.float32, device="cuda:0")
        
        # Camera orientation: pitch=0°, roll=90° (look down), yaw=specified
        # Roll controls looking down, Yaw controls rotation around Z-axis
        pitch_angle = torch.tensor([90.0], dtype=torch.float32, device="cpu")
        # roll_angle = torch.tensor([yaw_deg], dtype=torch.float32, device="cpu")
        yaw_angle = torch.tensor([yaw_deg], dtype=torch.float32, device="cpu")
        
        # R_x = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(np.deg2rad(roll_angle.item())), -np.sin(np.deg2rad(roll_angle.item()))],
        #     [0, np.sin(np.deg2rad(roll_angle.item())), np.cos(np.deg2rad(roll_angle.item()))]
        # ])
        
        R_y = np.array([
            [np.cos(np.deg2rad(pitch_angle.item())), 0, np.sin(np.deg2rad(pitch_angle.item()))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(pitch_angle.item())), 0, np.cos(np.deg2rad(pitch_angle.item()))]
        ])
        
        R_z = np.array([
            [np.cos(np.deg2rad(yaw_angle.item())), -np.sin(np.deg2rad(yaw_angle.item())), 0],
            [np.sin(np.deg2rad(yaw_angle.item())), np.cos(np.deg2rad(yaw_angle.item())), 0],
            [0, 0, 1]
        ])
        
        # Convert numpy quaternion (qx,qy,qz,qw) -> torch (w,x,y,z), batched and moved to GPU
        q_np = rot_to_quaternion(R_z @ R_y)
        # rot_to_quaternion returns [qx, qy, qz, qw]
        qx, qy, qz, qw = float(q_np[0]), float(q_np[1]), float(q_np[2]), float(q_np[3])
        orientation = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32, device="cuda:0")
        
        # print(f"[DEBUG] Camera orientation: pitch={pitch_angle.item():.1f}°, roll={roll_angle.item():.1f}°, yaw={yaw_angle.item():.1f}°")
        
        # Set camera pose
        camera.set_world_poses(
            positions=position,
            orientations=orientation,
            env_ids=torch.tensor([0], device="cuda:0"),
            convention="world",
        )
        
        # Render
        scene.write_data_to_sim()
        for _ in range(10):
            sim.render()
        scene.update(sim.get_physics_dt())
        
        # Get RGB data
        rgb_data = camera.data.output["rgb"][0]
        rgb_array = rgb_data.cpu().numpy()
        
        # Convert to uint8
        if rgb_array.dtype in [np.float32, np.float64]:
            rgb_array = (np.clip(rgb_array, 0, 1) * 255).astype(np.uint8)
        
        # Extract RGB channels
        if rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
            rgb_img = rgb_array[:, :, :3]
        else:
            rgb_img = rgb_array
        
        # Resize if needed
        if rgb_img.shape[0] != resolution or rgb_img.shape[1] != resolution:
            from PIL import Image
            rgb_img = np.array(Image.fromarray(rgb_img).resize((resolution, resolution)))
        
        # Save
        from PIL import Image
        Image.fromarray(rgb_img).save(output_path)
        print(f"[INFO] ✓ Saved rotated RGB to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Failed to render RGB: {e}")
        import traceback
        traceback.print_exc()
        return False


def compute_rotated_maps(scene, center_x, center_y, size, grid_res, yaw_deg):
    """
    Generate height and semantic maps in ROTATED local coordinate frame
    
    Workflow:
    1. Define local grid in rotated coordinate system (size x size square)
    2. For each local cell (i, j):
       - Compute local coordinates (x_local, y_local)
       - Transform to global coordinates using rotation matrix
       - Perform raycast in global frame
    3. Store results in local grid (preserving rotated view)
    
    Args:
        scene: InteractiveScene
        center_x, center_y: Center position in global frame
        size: Size of capture region (meters)
        grid_res: Grid resolution (meters)
        yaw_deg: Yaw angle in degrees
    
    Returns:
        dict with height_map, sem_mask, etc.
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

    # Setup terrain analysis
    tac = TerrainAnalysisCfg()
    tac.grid_resolution = grid_res
    tac.semantic_cost_mapping = None

    ta = TerrainAnalysis(tac, scene)
    ta._setup_raycaster()

    # Compute rotation matrix
    yaw_rad = np.deg2rad(yaw_deg)
    rot_matrix = rotation_matrix_z(yaw_rad)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Computing maps in ROTATED view")
    print(f"{'='*60}")
    print(f"  Center (global): ({center_x:.1f}, {center_y:.1f})")
    print(f"  Yaw angle: {yaw_deg:.1f}° ({yaw_rad:.3f} rad)")
    print(f"  Size: {size:.1f}m x {size:.1f}m")
    print(f"  Grid resolution: {grid_res}m")
    
    # Create LOCAL grid (in rotated frame)
    num_points = int(np.round(size / grid_res)) + 1
    
    # Local coordinates: centered at origin, ranging from -size/2 to +size/2
    half_size = size / 2.0
    xs_local = np.linspace(-half_size, half_size, num_points)
    ys_local = np.linspace(-half_size, half_size, num_points)
    X_local, Y_local = np.meshgrid(xs_local, ys_local, indexing='ij')
    
    print(f"  Local grid: {num_points}x{num_points} points")
    print(f"  Local bounds: x=[{-half_size:.2f}, {half_size:.2f}], y=[{-half_size:.2f}, {half_size:.2f}]")
    
    # Transform local points to GLOBAL coordinates
    # For each (x_local, y_local), compute (x_global, y_global):
    #   [x_global]   [x_center]       [x_local]
    #   [y_global] = [y_center] + R * [y_local]
    
    local_points = np.stack([X_local.flatten(), Y_local.flatten()], axis=1)  # (N, 2)
    global_points_center = local_points @ rot_matrix.T  # Rotate: (N, 2) @ (2, 2)
    global_points_center[:, 0] += center_x  # Translate X
    global_points_center[:, 1] += center_y  # Translate Y
    
    # X_global = global_points[:, 0].reshape(X_local.shape)
    # Y_global = global_points[:, 1].reshape(Y_local.shape)
    
    # print(f"  Global bounds: x=[{X_global.min():.2f}, {global_points_center.max():.2f}], y=[{Y_global.min():.2f}, {Y_global.max():.2f}]")
    
    
    # -----------新修改的在这
    # --- [STRATEGY] 5-Point Pattern Generation ---
    # Offset delta: 1/4 of grid resolution to stay inside the cell but away from center
    delta = grid_res * 0.25
    
    # Base Global Points: (N, 2)
    # We want to create (N*5, 2)
    
    # Define offsets in GLOBAL frame? 
    # No, it's safer to define in LOCAL frame and rotate, but for small delta, 
    # fixed global offsets work fine for jittering. 
    # Let's simply apply offsets to the computed global centers.
    
    n_pixels = global_points_center.shape[0]
    
    # shape: (N, 5, 2)
    expanded_points = np.zeros((n_pixels, 5, 2), dtype=np.float32)
    
    # 1. Center
    expanded_points[:, 0, :] = global_points_center
    # 2. Right (+x)
    expanded_points[:, 1, :] = global_points_center + np.array([delta, 0])
    # 3. Left (-x)
    expanded_points[:, 2, :] = global_points_center + np.array([-delta, 0])
    # 4. Up (+y)
    expanded_points[:, 3, :] = global_points_center + np.array([0, delta])
    # 5. Down (-y)
    expanded_points[:, 4, :] = global_points_center + np.array([0, -delta])
    
    # Flatten to (N*5, 2) for batch raycast
    batch_global_points = expanded_points.reshape(-1, 2)
    
    # Raycast parameters
    raycast_start_height = 2.0
    raycast_max_dist = 2.2
    # hit_threshold = raycast_start_height - 0.1 # Valid hit must be below this Z
    
    # Prepare Tensor 为每个采样点生成一组“从空中往下打”的射线起点和方向，用于后续的raycast（射线投射）操作，常用于地形高度或障碍物检测
    grid_points_down = torch.from_numpy(
        np.column_stack([
            batch_global_points[:, 0],
            batch_global_points[:, 1],
            np.ones(len(batch_global_points)) * raycast_start_height
        ])
    ).float().to(ta.device)
    # 这一步把所有采样点的XY坐标和统一的Z高度拼成一个 (N, 3) 的三维坐标数组，表示每条射线的起点。
    direction_down = torch.zeros_like(grid_points_down)
    direction_down[:, 2] = -1.0
    
    # ---------------------------------------------
    # Step 1: Ground Raycast (Multi-Sampled)
    # ---------------------------------------------
    print(f"\n[DEBUG] Step 1: Raycast Ground (Batch size: {len(batch_global_points)})")
    
    # 未击中的点返回的为inf,见网页https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/utils/warp/ops.html#raycast_mesh
    ground_hit = raycast_mesh(
        ray_starts=grid_points_down.unsqueeze(0),
        ray_directions=direction_down.unsqueeze(0),
        max_dist=1e6,
        mesh=ta._warp_mesh,
    )[0].squeeze(0)
    
    # 打印 ground_hit 中大于 raycast_max_dist 的坐标
    ground_hit_np = ground_hit.cpu().numpy()
    mask_gt = ground_hit_np[:, 2] > raycast_max_dist
    if np.any(mask_gt):
        print(f"[DEBUG] ground_hit > raycast_max_dist ({raycast_max_dist}):")
        idxs = np.where(mask_gt)[0]
        for idx in idxs:
            x, y, z = ground_hit_np[idx]
            print(f"  idx={idx}: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
    else:
        print(f"[DEBUG] No ground_hit z > raycast_max_dist ({raycast_max_dist})")
    
    # Raw Z values (N*5, )
    raw_z = ground_hit[:, 2].cpu().numpy()
    
    # 检查raw_z每一列是否全大于2.9，如果是则打印该列编号
    z_grouped = raw_z.reshape(n_pixels, 5)
    for col in range(z_grouped.shape[1]):
        if np.all(z_grouped[:, col] > raycast_max_dist):
            print(f"[WARNING] All values in column {col} of raw_z are > {raycast_max_dist}")

    # 1. Reshape back to (N, 5)
    z_grouped = raw_z.reshape(n_pixels, 5)
    
    # 2. Filter Misses: Set misses to NaN
    # Miss condition: value > raycast_max_dist (e.g. returns 3.0)
    mask_miss = z_grouped > raycast_max_dist
    z_grouped_filtered = z_grouped.copy()
    z_grouped_filtered[mask_miss] = np.nan
    
    # 统计 z_grouped_filtered 为 nan 的点并打印其坐标
    # nan_mask = np.isnan(z_grouped_filtered)
    # nan_indices = np.argwhere(nan_mask)
    # if nan_indices.size > 0:
    #     print(f"[INFO] Number of NaN points in z_grouped_filtered: {nan_indices.shape[0]}")
    #     for idx in nan_indices:
    #         pixel_idx, sample_idx = idx
    #         x_local, y_local = local_points[pixel_idx]
    #         x_global, y_global = global_points_center[pixel_idx]
    #         print(f"  NaN at pixel={pixel_idx}, sample={sample_idx}: local=({x_local:.2f}, {y_local:.2f}), global=({x_global:.2f}, {y_global:.2f})")
    # else:
    #     print("[INFO] No NaN points found in z_grouped_filtered.")
    # 3. Statistical Aggregation: NanMedian
    # Ignores NaNs. If [0.0, 3.0, 0.0, 3.0, 0.0] -> Median([0,0,0]) = 0.0 (Correct!)
    # If all NaNs -> returns NaN
    ground_height_agg = np.nanmedian(z_grouped_filtered, axis=1)
    
    # 统计 ground_height_agg 为 nan 的点并打印其坐标
    # nan_mask = np.isnan(ground_height_agg)
    # if np.any(nan_mask):
    #     nan_indices = np.where(nan_mask)[0]
    #     print(f"[INFO] Number of NaN ground points: {len(nan_indices)}")
    #     for idx in nan_indices:
    #         x_local, y_local = local_points[idx]
    #         x_global, y_global = global_points_center[idx]
    #         print(f"  NaN at grid idx={idx}: local=({x_local:.2f}, {y_local:.2f}), global=({x_global:.2f}, {y_global:.2f})")
    # else:
    #     print("[INFO] No NaN ground points found.")
            
    # Reshape to grid
    height_map = ground_height_agg.reshape(X_local.shape)
    
    # # === 生成临时二值图像: 有数据为白, 无数据为黑 ===
    # # height_map: np.ndarray, shape=(H, W)
    # # 有数据: np.isfinite(height_map)
    # binary_img = np.zeros_like(height_map, dtype=np.uint8)
    # binary_img[np.isfinite(height_map)] = 255
    # # binary_img 现在是一个二值图像, 有数据为255(白), 无数据为0(黑)
    # import cv2
    # cv2.imwrite(os.path.join(out_dir, "height_valid_mask.png"), binary_img)
    
    valid_count = np.sum(np.isfinite(ground_height_agg))
    print(f"[DEBUG]   Valid Ground Points (after median filtering): {valid_count}/{n_pixels} ({100*valid_count/n_pixels:.1f}%)")

    # ---------------------------------------------
    # Step 2: Obstacle Check (Multi-Sampled)
    # ---------------------------------------------
    print(f"\n[DEBUG] Step 2: Check for obstacles")
    
    has_obstacle_agg = np.zeros(n_pixels, dtype=bool)
    
    if hasattr(ta, "_warp_mesh_obstacles") and ta._warp_mesh_obstacles is not None:
        obstacle_hit = raycast_mesh(
            ray_starts=grid_points_down.unsqueeze(0),
            ray_directions=direction_down.unsqueeze(0),
            max_dist=raycast_max_dist,
            mesh=ta._warp_mesh_obstacles,
        )[0].squeeze(0)
        
        obs_z_raw = obstacle_hit[:, 2].cpu().numpy()
        obs_z_grouped = obs_z_raw.reshape(n_pixels, 5)
        
        # Determine valid obstacle hits
        # Hit is valid if Z < raycast_max_dist
        is_hit_matrix = (obs_z_grouped < raycast_max_dist) & np.isfinite(obs_z_grouped)
        
        # Aggregation: Conservative "OR"
        # If ANY of the 5 rays hit an obstacle, the cell is an obstacle.
        # This prevents "thin obstacle" tunneling.
        
        has_obstacle_agg = np.any(is_hit_matrix, axis=1)
        
        print(f"[DEBUG]   Obstacles detected: {has_obstacle_agg.sum()}")
    
    # Final Traversability Logic
    # 1. Ground must be valid (not NaN)
    # 2. Must not have obstacle
    # is_traversable = np.isfinite(ground_height_agg) & ~has_obstacle_agg
    is_traversable = ~has_obstacle_agg
    
    sem_mask = is_traversable.astype(np.uint8).reshape(X_local.shape)
    
    return {
        "height": height_map,
        "sem_mask": sem_mask,
        "center": (center_x, center_y),
        "yaw_deg": yaw_deg,
        "grid_res": grid_res,
        "size": size,
        "rotation_matrix": rot_matrix,
        "used_semantics": True,
    }
    
    
    
    # # Raycast parameters
    # raycast_start_height = 3.0
    # raycast_max_dist = 3.1
    
    # print(f"\n[DEBUG] Step 1: Raycast DOWN to find ground height")
    # print(f"  start_height={raycast_start_height}m, max_dist={raycast_max_dist}m")
    
    # # Single-sample raycast (can add multi-sampling if needed)
    # grid_points_down = torch.from_numpy(
    #     np.column_stack([
    #         X_global.flatten(),
    #         Y_global.flatten(),
    #         np.ones_like(X_global.flatten()) * raycast_start_height
    #     ])
    # ).float().to(ta.device)
    
    # direction_down = torch.zeros_like(grid_points_down)
    # direction_down[:, 2] = -1.0  # Point down
    
    # ground_hit = raycast_mesh(
    #     ray_starts=grid_points_down.unsqueeze(0),
    #     ray_directions=direction_down.unsqueeze(0),
    #     max_dist=raycast_max_dist,
    #     mesh=ta._warp_mesh,
    # )[0].squeeze(0)
    
    # ground_height = ground_hit[:, 2].cpu().numpy()
    
    # # Debug
    # valid_ground = ground_height[np.isfinite(ground_height)]
    # if len(valid_ground) > 0:
    #     print(f"[DEBUG]   Ground found: {len(valid_ground)}/{len(ground_height)} points ({100*len(valid_ground)/len(ground_height):.1f}%)")
    #     print(f"[DEBUG]   Ground height range: [{valid_ground.min():.2f}, {valid_ground.max():.2f}] meters")
    # else:
    #     print(f"[WARNING]   No ground found!")
    #     return {
    #         "height": np.full(X_local.shape, np.nan),
    #         "sem_mask": np.zeros(X_local.shape, dtype=np.uint8),
    #         "center": (center_x, center_y),
    #         "yaw_deg": yaw_deg,
    #         "grid_res": grid_res,
    #         "used_semantics": False,
    #     }
    
    # # Reshape to local grid
    # height_map = ground_height.reshape(X_local.shape)
    
    # # Step 2: Check for obstacles
    # print(f"\n[DEBUG] Step 2: Check for obstacles")
    
    # has_obstacle = None
    # if hasattr(ta, "_warp_mesh_obstacles") and ta._warp_mesh_obstacles is not None:
    #     grid_points_down_obstacle = torch.from_numpy(
    #         np.column_stack([
    #             X_global.flatten(),
    #             Y_global.flatten(),
    #             np.ones_like(X_global.flatten()) * raycast_start_height
    #         ])
    #     ).float().to(ta.device)
        
    #     direction_down_obstacle = torch.zeros_like(grid_points_down_obstacle)
    #     direction_down_obstacle[:, 2] = -1.0
        
    #     obstacle_hit_result = raycast_mesh(
    #         ray_starts=grid_points_down_obstacle.unsqueeze(0),
    #         ray_directions=direction_down_obstacle.unsqueeze(0),
    #         max_dist=raycast_max_dist,
    #         mesh=ta._warp_mesh_obstacles,
    #     )[0].squeeze(0)
        
    #     has_obstacle = torch.isfinite(obstacle_hit_result[:, 2]).cpu().numpy()
    #     print(f"[DEBUG]   Obstacles detected: {has_obstacle.sum()}/{len(has_obstacle)} points ({100*has_obstacle.sum()/len(has_obstacle):.1f}%)")
        
    #     # Traversability
    #     is_traversable = np.isfinite(ground_height) & ~has_obstacle
    # else:
    #     print(f"[WARNING]   No obstacle mesh available")
    #     is_traversable = np.isfinite(ground_height)
    
    # # Compute semantic mask
    # sem_mask = is_traversable.astype(np.uint8).reshape(X_local.shape)
    # print(f"[DEBUG]   Traversable points: {sem_mask.sum()}/{sem_mask.size} ({100*sem_mask.sum()/sem_mask.size:.1f}%)")
    
    # return {
    #     "height": height_map,
    #     "sem_mask": sem_mask,
    #     "center": (center_x, center_y),
    #     "yaw_deg": yaw_deg,
    #     "grid_res": grid_res,
    #     "size": size,
    #     "rotation_matrix": rot_matrix,
    #     "used_semantics": True,
    # }


# def compute_rotated_maps(scene, center_x, center_y, size, grid_res, yaw_deg):
#     """
#     Generate height and semantic maps in ROTATED local coordinate frame
#     (Robust Version: Uses 5-point multi-sampling to fix raycast leaks)
#     """
#     from importlib import util as importlib_util
#     from pathlib import Path as _Path
#     import types as _types
#     import sys as _sys
#     import numpy as np
#     import torch
#     from isaaclab.utils.warp import raycast_mesh
#     from isaaclab.utils.math import rotation_matrix_z

#     # --- Module Loading Logic (Keep unchanged) ---
#     _collectors_dir = _Path(__file__).resolve().parent.parent / "extension" / "omni.viplanner" / "omni" / "viplanner" / "collectors"
#     if "omni.viplanner.collectors" not in _sys.modules:
#         _pkg_collectors = _types.ModuleType("omni.viplanner.collectors")
#         _pkg_collectors.__path__ = [_collectors_dir.as_posix()]
#         _sys.modules["omni.viplanner.collectors"] = _pkg_collectors

#     _cfg_spec = importlib_util.spec_from_file_location("omni.viplanner.collectors.terrain_analysis_cfg", (_collectors_dir / "terrain_analysis_cfg.py").as_posix())
#     assert _cfg_spec and _cfg_spec.loader
#     _cfg_mod = importlib_util.module_from_spec(_cfg_spec)
#     _sys.modules[_cfg_spec.name] = _cfg_mod
#     _cfg_spec.loader.exec_module(_cfg_mod)
#     TerrainAnalysisCfg = getattr(_cfg_mod, "TerrainAnalysisCfg")

#     _ta_spec = importlib_util.spec_from_file_location("omni.viplanner.collectors.terrain_analysis_myself", (_collectors_dir / "terrain_analysis_myself.py").as_posix())
#     assert _ta_spec and _ta_spec.loader
#     _ta_mod = importlib_util.module_from_spec(_ta_spec)
#     _sys.modules[_ta_spec.name] = _ta_mod
#     _ta_spec.loader.exec_module(_ta_mod)
#     TerrainAnalysis = getattr(_ta_mod, "TerrainAnalysis")
#     # ---------------------------------------------

#     # Setup terrain analysis
#     tac = TerrainAnalysisCfg()
#     tac.grid_resolution = grid_res
#     tac.semantic_cost_mapping = None

#     ta = TerrainAnalysis(tac, scene)
#     ta._setup_raycaster()

#     # Compute rotation matrix
#     yaw_rad = np.deg2rad(yaw_deg)
#     rot_matrix = rotation_matrix_z(yaw_rad)
    
#     print(f"\n{'='*60}")
#     print(f"[INFO] Computing maps (Robust Multi-Sampling x5)")
#     print(f"{'='*60}")
    
#     # Create LOCAL grid
#     num_points = int(np.round(size / grid_res)) + 1
#     half_size = size / 2.0
#     xs_local = np.linspace(-half_size, half_size, num_points)
#     ys_local = np.linspace(-half_size, half_size, num_points)
#     X_local, Y_local = np.meshgrid(xs_local, ys_local, indexing='ij')
#     original_shape = X_local.shape
    
#     # Transform to GLOBAL (Central Points)
#     local_points = np.stack([X_local.flatten(), Y_local.flatten()], axis=1) # (N, 2)
#     global_points_center = local_points @ rot_matrix.T
#     global_points_center[:, 0] += center_x
#     global_points_center[:, 1] += center_y
    
#     # --- [STRATEGY] 5-Point Pattern Generation ---
#     # Offset delta: 1/4 of grid resolution to stay inside the cell but away from center
#     delta = grid_res * 0.25
    
#     # Base Global Points: (N, 2)
#     # We want to create (N*5, 2)
    
#     # Define offsets in GLOBAL frame? 
#     # No, it's safer to define in LOCAL frame and rotate, but for small delta, 
#     # fixed global offsets work fine for jittering. 
#     # Let's simply apply offsets to the computed global centers.
    
#     n_pixels = global_points_center.shape[0]
    
#     # shape: (N, 5, 2)
#     expanded_points = np.zeros((n_pixels, 5, 2), dtype=np.float32)
    
#     # 1. Center
#     expanded_points[:, 0, :] = global_points_center
#     # 2. Right (+x)
#     expanded_points[:, 1, :] = global_points_center + np.array([delta, 0])
#     # 3. Left (-x)
#     expanded_points[:, 2, :] = global_points_center + np.array([-delta, 0])
#     # 4. Up (+y)
#     expanded_points[:, 3, :] = global_points_center + np.array([0, delta])
#     # 5. Down (-y)
#     expanded_points[:, 4, :] = global_points_center + np.array([0, -delta])
    
#     # Flatten to (N*5, 2) for batch raycast
#     batch_global_points = expanded_points.reshape(-1, 2)
    
#     # Raycast parameters
#     raycast_start_height = 3.0
#     raycast_max_dist = 3.1
#     hit_threshold = raycast_start_height - 0.1 # Valid hit must be below this Z
    
#     # Prepare Tensor 为每个采样点生成一组“从空中往下打”的射线起点和方向，用于后续的raycast（射线投射）操作，常用于地形高度或障碍物检测
#     grid_points_down = torch.from_numpy(
#         np.column_stack([
#             batch_global_points[:, 0],
#             batch_global_points[:, 1],
#             np.ones(len(batch_global_points)) * raycast_start_height
#         ])
#     ).float().to(ta.device)
#     # 这一步把所有采样点的XY坐标和统一的Z高度拼成一个 (N, 3) 的三维坐标数组，表示每条射线的起点。
#     direction_down = torch.zeros_like(grid_points_down)
#     direction_down[:, 2] = -1.0
    
#     # ---------------------------------------------
#     # Step 1: Ground Raycast (Multi-Sampled)
#     # ---------------------------------------------
#     print(f"\n[DEBUG] Step 1: Raycast Ground (Batch size: {len(batch_global_points)})")
    
#     ground_hit = raycast_mesh(
#         ray_starts=grid_points_down.unsqueeze(0),
#         ray_directions=direction_down.unsqueeze(0),
#         max_dist=raycast_max_dist,
#         mesh=ta._warp_mesh,
#     )[0].squeeze(0)
    
#     # Raw Z values (N*5, )
#     raw_z = ground_hit[:, 2].cpu().numpy()
    
#     # 1. Reshape back to (N, 5)
#     z_grouped = raw_z.reshape(n_pixels, 5)
    
#     # 2. Filter Misses: Set misses to NaN
#     # Miss condition: value > hit_threshold (e.g. returns 3.0)
#     mask_miss = z_grouped > hit_threshold
#     z_grouped_filtered = z_grouped.copy()
#     z_grouped_filtered[mask_miss] = np.nan
    
#     # 3. Statistical Aggregation: NanMedian
#     # Ignores NaNs. If [0.0, 3.0, 0.0, 3.0, 0.0] -> Median([0,0,0]) = 0.0 (Correct!)
#     # If all NaNs -> returns NaN
#     ground_height_agg = np.nanmedian(z_grouped_filtered, axis=1)
    
#     # Reshape to grid
#     height_map = ground_height_agg.reshape(original_shape)
    
#     valid_count = np.sum(np.isfinite(ground_height_agg))
#     print(f"[DEBUG]   Valid Ground Points (after median filtering): {valid_count}/{n_pixels} ({100*valid_count/n_pixels:.1f}%)")

#     # ---------------------------------------------
#     # Step 2: Obstacle Check (Multi-Sampled)
#     # ---------------------------------------------
#     print(f"\n[DEBUG] Step 2: Check for obstacles")
    
#     has_obstacle_agg = np.zeros(n_pixels, dtype=bool)
    
#     if hasattr(ta, "_warp_mesh_obstacles") and ta._warp_mesh_obstacles is not None:
#         obstacle_hit = raycast_mesh(
#             ray_starts=grid_points_down.unsqueeze(0),
#             ray_directions=direction_down.unsqueeze(0),
#             max_dist=raycast_max_dist,
#             mesh=ta._warp_mesh_obstacles,
#         )[0].squeeze(0)
        
#         obs_z_raw = obstacle_hit[:, 2].cpu().numpy()
#         obs_z_grouped = obs_z_raw.reshape(n_pixels, 5)
        
#         # Determine valid obstacle hits
#         # Hit is valid if Z < hit_threshold
#         is_hit_matrix = (obs_z_grouped < hit_threshold) & np.isfinite(obs_z_grouped)
        
#         # Aggregation: Conservative "OR"
#         # If ANY of the 5 rays hit an obstacle, the cell is an obstacle.
#         # This prevents "thin obstacle" tunneling.
#         has_obstacle_agg = np.any(is_hit_matrix, axis=1)
        
#         print(f"[DEBUG]   Obstacles detected: {has_obstacle_agg.sum()}")
    
#     # Final Traversability Logic
#     # 1. Ground must be valid (not NaN)
#     # 2. Must not have obstacle
#     is_traversable = np.isfinite(ground_height_agg) & ~has_obstacle_agg
    
#     sem_mask = is_traversable.astype(np.uint8).reshape(original_shape)
    
#     return {
#         "height": height_map,
#         "sem_mask": sem_mask,
#         "center": (center_x, center_y),
#         "yaw_deg": yaw_deg,
#         "grid_res": grid_res,
#         "size": size,
#         "rotation_matrix": rot_matrix,
#         "used_semantics": True,
#     }


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Determine yaw angle
    if args.yaw_deg is not None:
        yaw_deg = args.yaw_deg
        print(f"[INFO] Using specified yaw angle: {yaw_deg:.1f}°")
    else:
        yaw_deg = np.random.uniform(args.yaw_min, args.yaw_max)
        print(f"[INFO] Using random yaw angle: {yaw_deg:.1f}° (range: [{args.yaw_min:.1f}°, {args.yaw_max:.1f}°])")
    
    os.environ.setdefault("VIPLANNER_DISABLE_PEOPLE", "1")
    app = AppLauncher(args).app
    
    scene, sim = make_scene(args.scene)
    
    # Camera position
    capture_x = 0.0 + args.offset_x
    capture_y = 0.0 + args.offset_y
    
    print(f"\n{'='*60}")
    print(f"[INFO] ROTATED VIEW Data Collection")
    print(f"{'='*60}")
    print(f"  Scene: {args.scene}")
    print(f"  Center: ({capture_x:.1f}, {capture_y:.1f})")
    print(f"  Size: {args.capture_size}m x {args.capture_size}m")
    print(f"  Camera height: {args.camera_height}m")
    print(f"  Yaw angle: {yaw_deg:.1f}°")
    print(f"  Grid resolution: {args.grid_res}m")
    print(f"{'='*60}\n")
    
    # Generate maps
    result = compute_rotated_maps(scene, capture_x, capture_y, args.capture_size, args.grid_res, yaw_deg)
    
    # Setup output directory
    out_dir = args.save_dir or os.environ.get("VIPLANNER_DATA_DIR") or str(Path.cwd() / "traversability_rotated_output")
    os.makedirs(out_dir, exist_ok=True)
    
    
    
    # Save data
    np.save(os.path.join(out_dir, "height.npy"), result["height"])
    np.save(os.path.join(out_dir, "sem_mask.npy"), result["sem_mask"])
    
    # Save camera pose
    camera_pose = {
        "position": np.array([capture_x, capture_y, args.camera_height]),
        "yaw_deg": yaw_deg,
        "pitch_deg": 0.0,
        "roll_deg": 90.0,
        "rotation_matrix": result["rotation_matrix"]
    }
    np.save(os.path.join(out_dir, "camera_pose.npy"), camera_pose)
    
    # Render RGB
    rgb_path = os.path.join(out_dir, "top_down_rgb.png")
    render_rotated_rgb(sim, scene, capture_x, capture_y, args.camera_height, yaw_deg, rgb_path, resolution=512)
    
    # Generate visualizations
    try:
        import cv2
        
        # Save binary mask: 255 where height is finite, 0 where not
        binary_mask = np.zeros_like(result["height"], dtype=np.uint8)
        binary_mask[np.isfinite(result["height"])] = 255
        cv2.imwrite(os.path.join(out_dir, "height_valid_mask.png"), binary_mask)
        
        # Height PNG
        hg = result["height"]
        finite = np.isfinite(hg)
        if finite.any():
            hmin = np.nanmin(hg[finite])
            hmax = np.nanmax(hg[finite])
            denom = (hmax - hmin) if (hmax > hmin) else 1.0
            norm = np.clip((hg - hmin) / denom, 0.0, 1.0)
            norm[~finite] = 0.0
            height_img = (norm * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, "height.png"), height_img)
        
        # Semantic mask PNG
        sm = (result["sem_mask"] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "sem_mask.png"), sm)
    except Exception as e:
        print(f"[WARNING] Failed to generate PNG: {e}")
    
    # Success report
    success_path = os.path.join(out_dir, "SUCCESS.txt")
    with open(success_path, "w") as f:
        f.write("=== Rotated View Data Generation Success ===\n")
        f.write(f"Timestamp: {args.seed}\n")
        f.write(f"Scene: {args.scene}\n")
        f.write(f"Center (global): ({capture_x:.1f}, {capture_y:.1f})\n")
        f.write(f"Yaw angle: {yaw_deg:.1f}°\n")
        f.write(f"Grid resolution: {args.grid_res}m\n")
        f.write(f"Data shape: {result['height'].shape}\n")
        f.write(f"\n--- Generated Files ---\n")
        f.write("✓ height.npy - Height map (rotated view)\n")
        f.write("✓ sem_mask.npy - Semantic mask (rotated view)\n")
        f.write("✓ camera_pose.npy - Camera pose info\n")
        f.write("✓ top_down_rgb.png - RGB image (rotated view)\n")
        f.write("✓ height.png - Height visualization\n")
        f.write("✓ sem_mask.png - Semantic visualization\n")
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Data saved to: {out_dir}")
    print("="*60)
    print(f"✓ height.npy - Height map (rotated view)")
    print(f"✓ sem_mask.npy - Semantic mask (rotated view)")
    print(f"✓ camera_pose.npy - Camera pose")
    print(f"✓ top_down_rgb.png - RGB image")
    print(f"✓ Yaw angle: {yaw_deg:.1f}°")
    print("="*60 + "\n")
    
    app.close()


if __name__ == "__main__":
    main()
