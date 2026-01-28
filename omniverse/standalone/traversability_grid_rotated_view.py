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
 

# check_indoor 表示是否接受室内检查，仅针对室外场景有效
def compute_rotated_maps(scene, center_x, center_y, size, grid_res, yaw_deg, check_indoor=True, terrain_analyzer=None):
    # [优化] 如果外部传入了 TA 实例，直接使用，跳过所有初始化开销
    if terrain_analyzer is not None:
        ta = terrain_analyzer
        # 确保 raycaster 已设置 (由于我们在第一步加了缓存检查，这里调用是安全的且快速的)
        ta._setup_raycaster()
        
        # 引入必要的库 (因为没走下面的 import 流程)
        import numpy as np
        import torch
        from isaaclab.utils.warp import raycast_mesh
        
        # 注意：这里需要确保 ta 所在的 device 和当前的 scene device 一致
    else:
        # === 原有的初始化逻辑 (当作为独立脚本运行时走这里) ===
        from importlib import util as importlib_util
        from pathlib import Path as _Path
        import types as _types
        import sys as _sys
        import numpy as np
        import torch
        from isaaclab.utils.warp import raycast_mesh

        _collectors_dir = _Path(__file__).resolve().parent.parent / "extension" / "omni.viplanner" / "omni" / "viplanner" / "collectors"
        # ... (省略原本的 import 代码) ...
        
        # 动态加载 TerrainAnalysis
        # ... (省略原本的 import 代码) ...
        TerrainAnalysis = getattr(_ta_mod, "TerrainAnalysis")
        TerrainAnalysisCfg = getattr(_cfg_mod, "TerrainAnalysisCfg")

        # Setup terrain analysis
        tac = TerrainAnalysisCfg()
        tac.grid_resolution = grid_res
        # tac.semantic_cost_mapping = None

        ta = TerrainAnalysis(tac, scene)
        ta._setup_raycaster()
    
    # Compute rotation matrix
    
    # [注意] 如果上面用了 terrain_analyzer，需要确保 numpy 被导入了
    if 'np' not in locals(): import numpy as np
    
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
    
    local_points = np.stack([X_local.flatten(), Y_local.flatten()], axis=1)  # (N, 2)
    global_points_center = local_points @ rot_matrix.T  # Rotate: (N, 2) @ (2, 2)
    global_points_center[:, 0] += center_x  # Translate X
    global_points_center[:, 1] += center_y  # Translate Y
    delta = grid_res * 0.25    
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
    
    # ---------------------------------------------
    # 0. 准备 Raycast 参数
    # ---------------------------------------------
    # 这里的 raycast_start_height 是发射源的高度。
    # 确保这个高度高于你场景中的大部分地面，否则射线从地下往下打会直接 miss。
    # 比如设为 2.0 或更高 (根据你的场景需求)
    raycast_start_height = 2.0  
    # 最大探测距离。如果 start=2.0, max_dist=2.2，说明只能探测到 Z=-0.2 以上的地面
    raycast_max_dist = 2.5      
    
    # 生成射线起点 (N*5, 3)
    grid_points_down = torch.from_numpy(
        np.column_stack([
            batch_global_points[:, 0],
            batch_global_points[:, 1],
            np.ones(len(batch_global_points)) * raycast_start_height
        ])
    ).float().to(ta.device)
    
    # 生成射线方向 (向下)
    direction_down = torch.zeros_like(grid_points_down)
    direction_down[:, 2] = -1.0
    
    # ---------------------------------------------
    # Step 1: 地面高度检测 (Raycast Ground)
    # ---------------------------------------------
    print(f"\n[DEBUG] Step 1: Raycast Ground (Batch size: {len(batch_global_points)})")
    
    # 向下发射射线，获取击中点
    ground_hit = raycast_mesh(
        ray_starts=grid_points_down.unsqueeze(0),
        ray_directions=direction_down.unsqueeze(0),
        max_dist=1e6, # 先用大距离拿到点，后续再根据 max_dist 过滤 NaN
        mesh=ta._warp_mesh,
    )[0].squeeze(0) # shape (N*5, 3)

    # ---------------------------------------------
    # Step 1.1: 室内点检测与过滤 (Indoor Check)
    # ---------------------------------------------
    is_indoor_mask = None
    
    if check_indoor:
        # [逻辑修正]: 基于"击中点"来判断是否在室内，而不是基于半空中的"发射点"
        # 1. 克隆击中点坐标
        check_origins = ground_hit.clone()
        
        # 2. 只有“有效击中”的点才需要检查室内 (如果是 inf 无穷远，说明没地，不用查)
        #    这里简单起见，我们对所有点做检查，但 check_if_indoor 内部最好能处理 inf 的情况
        #    将检查点从地面抬高 0.5m (模拟机器人高度)，进行四周探测
        check_origins[:, 2] += 0.5 
        
        # 3. 调用室内检测函数 (注意：使用原始 mesh)
        #    max_dist 设为室内墙壁判定的阈值 (如 3.0m - 5.0m)
        is_indoor_mask = ta.check_if_indoor(check_origins, max_dist=5.0) 
        
        num_indoor = is_indoor_mask.sum().item()
        if num_indoor > 0:
            print(f"[DEBUG] Detected {num_indoor} indoor points to be filtered.")

    # ---------------------------------------------
    # Step 1.2: 处理高度图数据
    # ---------------------------------------------
    # 获取 Z 轴高度 (N*5, )
    raw_z = ground_hit[:, 2].cpu().numpy()
    
    # [关键过滤]: 如果是室内点，将其高度设为 NaN
    if is_indoor_mask is not None:
        indoor_np = is_indoor_mask.cpu().numpy()
        raw_z[indoor_np] = np.nan

    # 数据重塑 (N, 5)
    z_grouped = raw_z.reshape(n_pixels, 5)
    
    # 过滤未击中的点 (Misses): 如果 Z > raycast_start_height + buffer 或者距离过远
    # 注意：raycast_mesh 返回的是世界坐标 Z。
    # 简单的判定：如果 Z 坐标过低(打穿了) 或者 距离发射点过远，都视为无效
    # 这里沿用你的逻辑：判断距离是否超过阈值
    # 计算距离: start_z - hit_z
    dists = raycast_start_height - z_grouped
    mask_miss = (dists > raycast_max_dist) | (dists < 0) # 距离过大 或 在发射点上方
    
    z_grouped_filtered = z_grouped.copy()
    z_grouped_filtered[mask_miss] = np.nan
    
    # 取中位数得到最终高度
    ground_height_agg = np.nanmedian(z_grouped_filtered, axis=1)
    height_map = ground_height_agg.reshape(X_local.shape)
    
    valid_count = np.sum(np.isfinite(ground_height_agg))
    print(f"[DEBUG] Valid Ground Points: {valid_count}/{n_pixels}")

    # ---------------------------------------------
    # Step 2: 障碍物检测 (Obstacle Check)
    # ---------------------------------------------
    print(f"\n[DEBUG] Step 2: Check for obstacles")
    
    has_obstacle_agg = np.zeros(n_pixels, dtype=bool)
    
    # 初始化击中矩阵 (False 表示没障碍)
    is_hit_matrix = np.zeros((n_pixels, 5), dtype=bool)

    # 2.1 物理障碍物检测 (针对 _warp_mesh_obstacles)
    if hasattr(ta, "_warp_mesh_obstacles") and ta._warp_mesh_obstacles is not None:
        obstacle_hit = raycast_mesh(
            ray_starts=grid_points_down.unsqueeze(0),
            ray_directions=direction_down.unsqueeze(0),
            max_dist=raycast_max_dist, # 障碍物也只检测在这个距离内的
            mesh=ta._warp_mesh_obstacles,
        )[0].squeeze(0)
        
        obs_z_raw = obstacle_hit[:, 2].cpu().numpy()
        obs_z_grouped = obs_z_raw.reshape(n_pixels, 5)
        
        # 如果击中点的距离在范围内，且是有限值，则视为障碍物
        obs_dists = raycast_start_height - obs_z_grouped
        is_hit_matrix = (obs_dists < raycast_max_dist) & (obs_dists > 0) & np.isfinite(obs_z_grouped)

    # 2.2 [关键逻辑] 融合室内检测结果
    # 如果该点被判定为室内，也将其视为障碍物(不可通行)
    if is_indoor_mask is not None:
        indoor_reshaped = is_indoor_mask.cpu().numpy().reshape(n_pixels, 5)
        # 取并集：物理障碍物 OR 室内点 = 不可通行
        is_hit_matrix = is_hit_matrix | indoor_reshaped

    # 聚合：5个采样点中只要有一个是障碍物，该网格即为障碍物
    has_obstacle_agg = np.any(is_hit_matrix, axis=1)
    
    print(f"[DEBUG] Obstacles detected: {has_obstacle_agg.sum()}")
    
    # ---------------------------------------------
    # Step 3: 生成语义掩码
    # ---------------------------------------------
    # 可通行 = (地面高度有效) AND (没有障碍物)
    is_traversable = np.isfinite(ground_height_agg) & ~has_obstacle_agg
    
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
    result = compute_rotated_maps(scene, capture_x, capture_y, args.capture_size, True, args.grid_res, yaw_deg)
    
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
