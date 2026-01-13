# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect Training Data for ViPlanner (myself variant)
- Uses ViewpointSampling_myself and ViewpointSamplingCfg_myself
- Minimal changes from data_collect.py
"""

import argparse
import sys
import importlib
import importlib.util
import types
from pathlib import Path

# --- 1. 路径与模块设置 (Global Setup) ---
# 确保在任何情况下（导入或运行）都能找到 omni.viplanner
_EXT_ROOT = Path(__file__).resolve().parent.parent / "extension"
_OMNI_VIPLANNER_PKG = (_EXT_ROOT / "omni.viplanner").as_posix()
if _OMNI_VIPLANNER_PKG not in sys.path:
    sys.path.insert(0, _OMNI_VIPLANNER_PKG)
_EXT_ROOT_POSIX = _EXT_ROOT.as_posix()
if _EXT_ROOT_POSIX not in sys.path:
    sys.path.append(_EXT_ROOT_POSIX)


# --- 2. 动态加载自定义模块 (Library Loading) ---
# 将这部分逻辑移到全局，以便其他脚本 import 该文件时能获取到 ViewpointSamplingCfg 类
_COLLECTORS_DIR = (
    Path(__file__).resolve().parent.parent
    / "extension"
    / "omni.viplanner"
    / "omni"
    / "viplanner"
    / "collectors"
)

# 构造一个轻量级“包”来承载 myself 模块
_PKG_NAME = "ovp_collectors_myself"
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [str(_COLLECTORS_DIR)]
    sys.modules[_PKG_NAME] = _pkg

# 尝试加载模块。如果因为缺少仿真环境(Isaac Sim context)导致加载失败，
# 捕获异常以防止 import 整个文件时崩溃，但在 main 中使用时会再次抛出。
try:
    _vp_mod = importlib.import_module(f"{_PKG_NAME}.viewpoint_sampling_myself")
    _vp_cfg_mod = importlib.import_module(f"{_PKG_NAME}.viewpoint_sampling_cfg_myself")
    
    # 暴露给外部调用的类
    ViewpointSampling = _vp_mod.ViewpointSampling
    ViewpointSamplingCfg = _vp_cfg_mod.ViewpointSamplingCfg
except Exception as e:
    # 这种情况通常发生在 ViewpointSampling 内部直接 import 了 omni.isaac.core
    # 而当前环境还没有启动 SimulationApp 时。
    # 我们允许这种情况发生，以便外部脚本至少能 import 这个文件本身而不报错。
    # 如果外部脚本真的需要使用这些类，它们需要确保环境已就绪。
    ViewpointSampling = None
    ViewpointSamplingCfg = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def main():
    """Main function to start the data collection in different environments (myself)."""
    
    # --- 3. 参数解析 (Execution Only) ---
    # 放在 main 内部，只有直接运行此脚本时才会执行
    from isaaclab.app import AppLauncher
    
    parser = argparse.ArgumentParser(description="Data collection for ViPlanner (myself).")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
    parser.add_argument(
        "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
    )
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save dataset (overrides default)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for viewpoint sampling")

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    
    # parse the arguments
    args_cli = parser.parse_args()
    
    # 强制启用相机
    args_cli.enable_cameras = True
    
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # --- 4. 仿真环境依赖导入 (Simulation Context) ---
    # 此时 App 已经启动，可以安全导入 IsaacLab 和 Omni 核心模块
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.timer import Timer
    
    # 检查前面的动态导入是否成功，如果不成功（可能因为之前没启动App），现在重试或报错
    global ViewpointSampling, ViewpointSamplingCfg
    if ViewpointSampling is None or ViewpointSamplingCfg is None:
        if _IMPORT_ERROR:
             print(f"[INFO] Initial import failed ({_IMPORT_ERROR}), retrying now that SimulationApp is running...")
        # 再次尝试导入，这次如果失败就是真的失败了
        _vp_mod = importlib.import_module(f"{_PKG_NAME}.viewpoint_sampling_myself")
        _vp_cfg_mod = importlib.import_module(f"{_PKG_NAME}.viewpoint_sampling_cfg_myself")
        ViewpointSampling = _vp_mod.ViewpointSampling
        ViewpointSamplingCfg = _vp_cfg_mod.ViewpointSamplingCfg

    # setup sampling config
    cfg = ViewpointSamplingCfg()
    # 指定用于地形分析的射线传感器
    cfg.terrain_analysis.raycaster_sensor = "depth_camera"

    # Disable optional people insertion and semantics to avoid extra dependencies
    import os as _os
    _os.environ.setdefault("VIPLANNER_DISABLE_PEOPLE", "1")
    # Do not disable semantics by default; let user/environment decide
    _os.environ.setdefault("VIPLANNER_DISABLE_SEMANTICS", "0")

    # Ensure package context so relative imports inside config modules work
    import types as _types
    # Correct package root locations under the extension folder
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
    # Create a lightweight config package stub to satisfy relative imports like '..viplanner'
    if "omni.viplanner.config" not in sys.modules:
        _ovc = _types.ModuleType("omni.viplanner.config")
        _ovc.__path__ = [_pkg_config_dir]
        sys.modules["omni.viplanner.config"] = _ovc

    # create environment cfg and modify the collector config depending on the environment
    if args_cli.scene == "matterport":
        # Load config module directly by file path to avoid running config/__init__.py
        _cfg_dir = _EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config"
        _mp_path = (_cfg_dir / "matterport_cfg.py").as_posix()
        _spec = importlib.util.spec_from_file_location("omni.viplanner.config.matterport_cfg", _mp_path)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec and _spec.loader, "Failed to construct spec for matterport_cfg.py"
        _spec.loader.exec_module(_mod)
        MatterportTerrainSceneCfg = getattr(_mod, "TerrainSceneCfg")
        scene_cfg = MatterportTerrainSceneCfg(1, env_spacing=1.0)
        # 简化运行：临时关闭语义代价过滤
        cfg.terrain_analysis.semantic_cost_mapping = None
    elif args_cli.scene == "carla":
        _cfg_dir = _EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config"
        _carla_path = (_cfg_dir / "carla_cfg.py").as_posix()
        _spec = importlib.util.spec_from_file_location("omni.viplanner.config.carla_cfg", _carla_path)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec and _spec.loader, "Failed to construct spec for carla_cfg.py"
        _spec.loader.exec_module(_mod)
        CarlaTerrainSceneCfg = getattr(_mod, "TerrainSceneCfg")
        scene_cfg = CarlaTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        scene_cfg.terrain.groundplane = False
        cfg.terrain_analysis.semantic_cost_mapping = None
        cfg.terrain_analysis.grid_resolution = 1.0
        cfg.terrain_analysis.sample_points = 10000
        cfg.terrain_analysis.dim_limiter_prim = "Road_Sidewalk"
    elif args_cli.scene == "warehouse":
        _cfg_dir = _EXT_ROOT / "omni.viplanner" / "omni" / "viplanner" / "config"
        _wh_path = (_cfg_dir / "warehouse_cfg.py").as_posix()
        _spec = importlib.util.spec_from_file_location("omni.viplanner.config.warehouse_cfg", _wh_path)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec and _spec.loader, "Failed to construct spec for warehouse_cfg.py"
        _spec.loader.exec_module(_mod)
        WarehouseTerrainSceneCfg = getattr(_mod, "TerrainSceneCfg")
        scene_cfg = WarehouseTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        # Enable semantic costs using Carla-style categories
        try:
            from omni.viplanner.config import CarlaSemanticCostMapping

            cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
        except Exception:
            cfg.terrain_analysis.semantic_cost_mapping = None
        # Use full terrain extent
        cfg.terrain_analysis.dim_limiter_prim = None
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # remove elements not necessary for the data collection
    scene_cfg.robot = None
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None

    # redirect camera prim paths when no robot base exists
    if args_cli.scene in ("warehouse", "carla"):
        scene_cfg.depth_camera.prim_path = "{ENV_REGEX_NS}/depth_cam"
        scene_cfg.semantic_camera.prim_path = "{ENV_REGEX_NS}/sem_cam"
    else:
        scene_cfg.depth_camera.prim_path = "/World/matterport"
        scene_cfg.semantic_camera.prim_path = "/World/matterport"

    # configure cameras/annotators
    cfg.cameras = {
        "depth_camera": "distance_to_image_plane",
        "semantic_camera": "semantic_segmentation",
    }

    # optional RGB
    if args_cli.scene == "matterport" and hasattr(scene_cfg, "rgb_camera"):
        scene_cfg.rgb_camera.prim_path = "/World/rgb_camera"
        cfg.cameras["rgb_camera"] = "rgb"
    elif "rgb" in scene_cfg.semantic_camera.data_types:
        cfg.cameras["semantic_camera"] = "rgb"

    # Build sim
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene manager: ", scene)
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    # Verify that terrain meshes are present on the stage
    try:
        from omni.viplanner.collectors.utils import get_all_meshes

        mesh_prims, mesh_names = get_all_meshes(scene.terrain.cfg.prim_path)
        preview = ", ".join(mesh_names[:5])
        print(f"[INFO] Terrain root: '{scene.terrain.cfg.prim_path}', meshes={len(mesh_prims)}: {preview}...")
        if len(mesh_prims) == 0:
            print(
                "[WARNING] No meshes detected under terrain root. If warehouse assets are missing, set \n"
                "  VIPLANNER_WAREHOUSE_USD to a valid USD (e.g., env/turtlebot3_simpleroom_use_bk.usd) and re-run."
            )
    except Exception as e:
        print(f"[WARNING] Mesh presence check skipped: {e}")

    # override save path from CLI or env var
    if args_cli.save_dir:
        cfg.save_path = args_cli.save_dir
    else:
        import os as _os
        cfg.save_path = _os.environ.get("VIPLANNER_DATA_DIR", None)

    # init sampler
    explorer = ViewpointSampling(cfg, scene)
    print("[INFO]: Setup complete (myself)...")

    # sample and render
    samples = explorer.sample_viewpoints(args_cli.num_samples, seed=args_cli.seed)
    explorer.render_viewpoints(samples)
    print("[INFO]: Viewpoints sampled (myself).")

    if not args_cli.headless:
        print("Rendering will continue to render the environment and visualize the last camera positions.")
        sim_dt = sim.get_physics_dt()
        while simulation_app.is_running():
            sim.render()
            explorer.scene.update(sim_dt)
    
    simulation_app.close()


if __name__ == "__main__":
    # 只有当作为主脚本运行时，才执行 main()
    # 这样其他脚本 import 这个文件时，不会触发 argparse 和 AppLauncher
    main()