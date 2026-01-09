# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""


import os
import sys

def nuclear_patch():
    # 1. 基础路径定义
    conda_base = os.environ.get('CONDA_PREFIX')
    vi_ext_root = os.path.expanduser("~/VLN/viplanner/omniverse/extension")
    # 推断仓库根目录（本脚本位于 <repo>/omniverse/standalone/）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # 2. 将仓库根与顶层包加入 sys.path，确保能导入 viplanner 顶层包
    for p in (repo_root, os.path.join(repo_root, "viplanner")):
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    # 3. 强行将插件根目录塞入 sys.path 最前端
    custom_plugins = ["omni.viplanner", "omni.isaac.matterport"]
    for plugin in custom_plugins:
        p = os.path.join(vi_ext_root, plugin)
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    # 4. 【核心】强行注入命名空间（解决 No module named 'omni.xxx'）
    import omni
    # 确保 omni 模块可以从多个地方加载
    if not hasattr(omni, "__path__"):
        omni.__path__ = []
        
    for plugin in custom_plugins:
        my_omni_dir = os.path.join(vi_ext_root, plugin, "omni")
        if os.path.exists(my_omni_dir) and my_omni_dir not in omni.__path__:
            omni.__path__.insert(0, my_omni_dir)
            
    # 5. 针对 'omni.isaac' 命名空间的二次注入
    try:
        import omni.isaac
        matterport_isaac_dir = os.path.join(vi_ext_root, "omni.isaac.matterport", "omni", "isaac")
        if os.path.exists(matterport_isaac_dir) and matterport_isaac_dir not in omni.isaac.__path__:
            omni.isaac.__path__.insert(0, matterport_isaac_dir)
    except:
        pass

    # 6. 注入 isaacsim 扩展搜索路径（exts / extscache / extsDeprecated）
    if conda_base:
        isaacsim_root = os.path.join(conda_base, 'lib', 'python3.11', 'site-packages', 'isaacsim')
        for sub in ('exts', 'extscache', 'extsDeprecated'):
            p = os.path.join(isaacsim_root, sub)
            if os.path.exists(p) and p not in sys.path:
                sys.path.append(p)

        # 7. 解决CXXABI版本问题：优先加载Isaac Sim自带的libstdc++等动态库
        kit_lib = os.path.join(isaacsim_root, 'kit', 'lib')
        old_ld = os.environ.get('LD_LIBRARY_PATH', '')
        def _prepend_ld(path):
            os.environ['LD_LIBRARY_PATH'] = path + (":" + old_ld if old_ld else "")
            print(f"[INFO] 已前置到LD_LIBRARY_PATH: {path}")
        # 优先尝试kit/lib，如果不存在或不包含libstdc++则回退到CONDA_PREFIX/lib
        if os.path.isdir(kit_lib):
            try:
                # 检查是否存在libstdc++文件
                has_libstdcpp = any(name.startswith('libstdc++') for name in os.listdir(kit_lib))
            except Exception:
                has_libstdcpp = False
            if has_libstdcpp:
                _prepend_ld(kit_lib)
            else:
                conda_lib = os.path.join(conda_base, 'lib') if conda_base else None
                if conda_lib and os.path.isdir(conda_lib):
                    _prepend_ld(conda_lib)
                    print(f"[INFO] kit/lib缺少libstdc++，已回退到CONDA lib: {conda_lib}")
                else:
                    print("[WARN] 未找到合适的libstdc++目录，请手动设置LD_LIBRARY_PATH")
        else:
            # kit/lib不存在，直接回退到CONDA lib
            conda_lib = os.path.join(conda_base, 'lib') if conda_base else None
            if conda_lib and os.path.isdir(conda_lib):
                _prepend_ld(conda_lib)
                print(f"[INFO] 未找到kit/lib，已回退到CONDA lib: {conda_lib}")
            else:
                print("[WARN] 未找到kit/lib且CONDA lib不可用，请手动设置LD_LIBRARY_PATH")

nuclear_patch()
print("[SUCCESS] 命名空间与路径已完成全手动强行挂载")

# 2. 激活 Isaac Sim 环境（Pip 模式必需）
try:
    import isaacsim
except ImportError:
    print("[Error] 找不到 isaacsim 包，请确认在 conda 环境中运行")

# isaaclab
import argparse

viplanner_ext_base = os.path.expanduser("~/VLN/viplanner/omniverse/extension")
if viplanner_ext_base not in sys.path:
    sys.path.append(viplanner_ext_base)
    # 也要把子目录加进去，确保 omni.viplanner 能被找到
    sys.path.append(os.path.join(viplanner_ext_base, "omni.viplanner"))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--model_dir", default=None, type=str, help="Path to model directory.")

# add applauncher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True
# Default to UI mode unless '--headless' is explicitly provided
try:
    if "--headless" not in sys.argv:
        args_cli.headless = False
except Exception:
    pass

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[INFO] Headless mode: {getattr(args_cli, 'headless', None)}")

"""Rest everything follows."""
# Ensure required Isaac extensions are enabled before importing modules
try:
    import omni.kit.app
    _ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for _ext in ("omni.isaac.core", "omni.isaac.ui", "omni.isaac.debug_draw", "isaacsim.util.debug_draw"):
        try:
            if not _ext_mgr.is_extension_enabled(_ext):
                _ext_mgr.set_extension_enabled_immediate(_ext, True)
        except Exception:
            pass
except Exception as _e:
    print(f"[Warn] Could not enable Isaac core extensions: {_e}")

import torch

# Import omni.isaac API with fallback to isaacsim legacy namespace
try:
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.objects import VisualCuboid
except ModuleNotFoundError:
    # Fallback for environments where omni.isaac is not available but extsDeprecated is present
    import isaacsim.core.utils.prims as prim_utils
    try:
        from isaacsim.core.api.objects import VisualCuboid
    except Exception:
        VisualCuboid = None
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.viplanner.viplanner.viplanner_algo import VIPlannerAlgo
from pxr import UsdGeom

"""
Main
"""


def main():
    """Imports all legged robots supported in IsaacLab and applies zero actions."""

    # create environment cfg
    if args_cli.scene == "matterport":
        env_cfg = ViPlannerMatterportCfg(seed=1234)
        goal_pos = torch.tensor([8.0, -13.5, 1.0])
    elif args_cli.scene == "carla":
        env_cfg = ViPlannerCarlaCfg(seed=1234)
        goal_pos = torch.tensor([137, 111.0, 1.0])
    elif args_cli.scene == "warehouse":
        env_cfg = ViPlannerWarehouseCfg(seed=1234)
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # create environment
    env = ManagerBasedRLEnv(env_cfg)

    # adjust the intrinsics of the camera
    depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # Make sure that groundplane is invisible
    if args_cli.scene == "carla":
        assert (
            prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
        )

    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    # Diagnostics: verify cameras and image tensors
    try:
        sensor_names = list(env.scene.sensors.keys())
        print(f"[INFO] Scene sensors: {sensor_names}")
        if "depth_camera" not in sensor_names or "semantic_camera" not in sensor_names:
            print("[ERROR] Required cameras missing (depth_camera/semantic_camera).")
        dep = obs.get("planner_image", {}).get("depth_measurement", None)
        sem = obs.get("planner_image", {}).get("semantic_measurement", None)
        if dep is not None and sem is not None:
            print(f"[INFO] Depth shape: {tuple(dep.shape)}, Semantic shape: {tuple(sem.shape)}")
            dep_min = torch.nan_to_num(dep).min().item()
            dep_max = torch.nan_to_num(dep).max().item()
            sem_min = torch.nan_to_num(sem).min().item()
            sem_max = torch.nan_to_num(sem).max().item()
            print(f"[INFO] Depth range: [{dep_min:.3f}, {dep_max:.3f}], Semantic range: [{sem_min:.3f}, {sem_max:.3f}]")
        else:
            print("[ERROR] Planner image tensors missing in observations.")
    except Exception as _e:
        print(f"[WARN] Camera diagnostics failed: {_e}")

    # Optional: place goal ~2m ahead in camera forward when explicitly requested.
    # Enable by: export VIPLANNER_FORWARD_GOAL=1
    if os.environ.get("VIPLANNER_FORWARD_GOAL", "").lower() in ("1", "true", "yes"):
        try:
            cam_pos = obs["planner_transform"]["cam_position"].clone()
            cam_quat = obs["planner_transform"]["cam_orientation"].clone()
            forward_cam = torch.tensor([2.0, 0.0, 0.0], device=cam_pos.device, dtype=cam_pos.dtype)
            forward_world = math_utils.quat_apply(cam_quat, forward_cam)
            # Flatten shapes like (1, 3) to (3,) to avoid indexing errors
            if isinstance(forward_world, torch.Tensor):
                forward_world = forward_world.reshape(-1)
            cam_pos = cam_pos.reshape(-1)
            if forward_world is not None and forward_world.numel() == 3 and cam_pos.numel() == 3:
                # 显式构造三维目标，避免索引边界错误
                goal_pos = torch.tensor(
                    [cam_pos[0] + forward_world[0], cam_pos[1] + forward_world[1], 1.0],
                    device=cam_pos.device,
                    dtype=cam_pos.dtype,
                )
            else:
                print("[WARN] Forward-goal placement skipped due to invalid shape; using default goal.")
        except Exception as _e:
            print(f"[WARN] Forward-goal placement failed ({_e}); using default goal.")

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    # pause the simulator
    # env.sim.pause()

    # load viplanner
    if args_cli.model_dir is None:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
        _default_model_dir = os.path.join(_repo_root, "viplanner_model")
        if os.path.isdir(_default_model_dir):
            args_cli.model_dir = _default_model_dir
        else:
            raise RuntimeError(
                "Model directory not provided. Pass --model_dir or place model files under viplanner_model/."
            )

    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=env.device)

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # One-time probe: draw a simple green line/point to validate DebugDraw
    try:
        if getattr(viplanner, "draw", None) is not None:
            viplanner.draw.clear_lines()
            viplanner.draw.clear_points()
            _p0 = [0.0, 0.0, 0.1]
            _p1 = [0.5, 0.0, 0.1]
            viplanner.draw.draw_lines([_p0], [_p1], [(0.4, 1.0, 0.1, 1.0)], [5.0])
            viplanner.draw.draw_points([_p1], [(0.4, 1.0, 0.1, 1.0)], [5.0])
            print("[INFO] DebugDraw probe line rendered.")
        else:
            print("[WARN] DebugDraw interface unavailable (likely headless or extension disabled).")
    except Exception as _e:
        print(f"[WARN] DebugDraw probe failed: {_e}")

    # initial paths
    _, paths, fear = viplanner.plan_dual(
        obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goals
    )

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue

            obs = env.step(action=paths.view(paths.shape[0], -1))[0]

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
        if torch.any(
            torch.norm(obs["planner_transform"]["cam_position"] - goals)
            > viplanner.train_config.data_cfg[0].max_goal_distance
        ):
            print(
                f"[WARNING]: Max goal distance is {viplanner.train_config.data_cfg[0].max_goal_distance} but goal is {torch.norm(obs['planner_transform']['cam_position'] - goals)} away from camera position! Please select new goal!"
            )
            env.sim.pause()
            continue

        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # draw path
        viplanner.debug_draw(paths, fear, goals)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
