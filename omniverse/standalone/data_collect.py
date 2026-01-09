# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect Training Data for ViPlanner
"""

"""Launch Isaac Sim Simulator first."""

# 说明（中文）：
# 本脚本用于在 IsaacLab/Isaac Sim 中生成训练数据（深度/语义/RGB 图像及相机内外参）。
# 主要流程：解析命令行 -> 启动仿真 App -> 创建场景配置 -> 采样相机视点 -> 渲染并保存数据 -> 可选的 UI 循环展示。

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
# 命令行解析器：定义数据采集的基本参数（环境数量、场景类型、样本数量、保存路径、随机种子）
parser = argparse.ArgumentParser(description="Data collection for ViPlanner.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
parser.add_argument("--save_dir", type=str, default=None, help="Directory to save dataset (overrides default)")
# 随机序列生成方式：指定随机种子以确保采样可复现。
parser.add_argument("--seed", type=int, default=1, help="Random seed for viewpoint sampling")

# append AppLauncher cli args
# 将 IsaacLab App 的通用参数（如设备、体验文件、是否无头等）追加到解析器中，便于通过统一接口启动仿真。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# 解析命令行参数，得到 args_cli。
args_cli = parser.parse_args()
# 强制启用相机渲染（确保采集管线有相机数据）。
args_cli.enable_cameras = True
# launch omniverse app
# 根据参数启动 Omniverse/IsaacLab 应用，返回一个 App 句柄。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
# 说明：下面导入的模块用于场景构建、仿真控制、计时日志以及视点采样与渲染。
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer
from omni.viplanner.collectors import ViewpointSampling, ViewpointSamplingCfg
from omni.viplanner.config import (
    CarlaSemanticCostMapping,
    MatterportSemanticCostMapping,
)
from omni.viplanner.config.carla_cfg import TerrainSceneCfg as CarlaTerrainSceneCfg
from omni.viplanner.config.matterport_cfg import (
    TerrainSceneCfg as MatterportTerrainSceneCfg,
)
from omni.viplanner.config.warehouse_cfg import (
    TerrainSceneCfg as WarehouseTerrainSceneCfg,
)


def main():
    """Main function to start the data collection in different environments."""
    # 创建视点采样的配置对象（控制渲染通道、地形分析、采样数量/范围等）。
    # setup sampling config
    cfg = ViewpointSamplingCfg()
    # 指定用于地形分析的射线传感器（这里使用深度相机作为高度/障碍评估的参考）。
    cfg.terrain_analysis.raycaster_sensor = "depth_camera"

    # create environment cfg and modify the collector config depending on the environment
    # 根据 --scene 选择具体场景的配置，并按场景特性调整采样与语义代价映射。
    if args_cli.scene == "matterport":
        # NOTE: only one env possible as the prims for the cameras cannot be initialized with the env regex
        # Matterport 场景：仅允许 1 个环境，因为其相机 prim 初始化不支持 env 正则模式。
        scene_cfg = MatterportTerrainSceneCfg(1, env_spacing=1.0)
        # overwrite semantic cost mapping and adjust parameters based on larger map
        # 指定适用于 Matterport 的语义代价映射（用于后续代价图生成/分析）。
        cfg.terrain_analysis.semantic_cost_mapping = MatterportSemanticCostMapping()
    elif args_cli.scene == "carla":
        # Carla 场景：可开启多个并行环境。
        scene_cfg = CarlaTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        # Carla 使用自己的地面，不需要默认 GroundPlane，关闭以避免干扰。
        scene_cfg.terrain.groundplane = False
        # overwrite semantic cost mapping and adjust parameters based on larger map
        # 指定适用于 Carla 的语义代价映射，并放宽栅格与采样规模以匹配更大地图范围。
        cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
        cfg.terrain_analysis.grid_resolution = 1.0
        cfg.terrain_analysis.sample_points = 10000
        # limit space to be within the road network
        # 限制采样空间在道路/人行道网络内，避免采样到不相关区域。
        cfg.terrain_analysis.dim_limiter_prim = "Road_Sidewalk"
    elif args_cli.scene == "warehouse":
        # 仓库场景：可开启多个并行环境，默认语义代价映射沿用 Carla（可根据需要替换）。
        scene_cfg = WarehouseTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        # overwrite semantic cost mapping
        cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
        # limit space to be within the road network
        # 以墙体网格名字作为空间限制（避免采样超出房间结构）。
        cfg.terrain_analysis.dim_limiter_prim = "Section"  # name of the meshes of the walls
    else:
        # 未支持的场景类型，抛出错误。
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # remove elements not necessary for the data collection
    # 数据采集不需要机器人/接触/高度扫描器，移除以降低开销与复杂度。
    scene_cfg.robot = None
    
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None

    # change the path to the semantic cameras as the robot base frame does not exist anymore
    # 由于不再实例化机器人底座，需要重定向相机的 prim 路径；
    # 对于多环境，使用 {ENV_REGEX_NS} 以匹配每个环境的命名空间。
    if args_cli.scene == "warehouse" or args_cli.scene == "carla":
        scene_cfg.depth_camera.prim_path = "{ENV_REGEX_NS}/depth_cam"
        scene_cfg.semantic_camera.prim_path = "{ENV_REGEX_NS}/sem_cam"
    else:
        scene_cfg.depth_camera.prim_path = "/World/matterport"
        scene_cfg.semantic_camera.prim_path = "/World/matterport"
    # 配置需要渲染的相机通道：深度图（到像平面距离）与语义分割。
    cfg.cameras = {
        "depth_camera": "distance_to_image_plane",
        "semantic_camera": "semantic_segmentation",
    }

    # adustments if also RGB images should be rendered
    # 如果需要渲染 RGB：
    # - 在 matterport 场景且存在独立 rgb_camera 配置时，设置其 prim 路径并启用 rgb 渲染；
    # - 否则若语义相机的 data_types 含 "rgb"，则切换该相机渲染为 rgb。
    if args_cli.scene == "matterport" and hasattr(scene_cfg, "rgb_camera"):
        scene_cfg.rgb_camera.prim_path = "/World/rgb_camera"
        cfg.cameras["rgb_camera"] = "rgb"
    elif "rgb" in scene_cfg.semantic_camera.data_types:
        cfg.cameras["semantic_camera"] = "rgb"

    # Load kit helper
    # 构建仿真上下文：创建仿真配置并初始化仿真控制器。
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    # generate scene
    # 创建交互场景，并计时输出创建耗时；随后重置仿真以开始渲染与物理。
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene manager: ", scene)
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    # override save path from CLI or environment variable if provided
    # 优先使用命令行的保存目录；否则尝试从环境变量 VIPLANNER_DATA_DIR 取默认保存路径。
    if args_cli.save_dir:
        cfg.save_path = args_cli.save_dir
    else:
        import os as _os
        cfg.save_path = _os.environ.get("VIPLANNER_DATA_DIR", None)

    # 初始化视点采样器：基于配置与场景，生成相机位姿并进行渲染与落盘。
    explorer = ViewpointSampling(cfg, scene)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # sample and render viewpoints
    # 按指定数量与随机种子采样视点，并渲染对应的相机数据到保存目录。
    samples = explorer.sample_viewpoints(args_cli.num_samples, seed=args_cli.seed)
    explorer.render_viewpoints(samples)
    print("[INFO]: Viewpoints sampled.")

    if not args_cli.headless:
        # 在非无头模式下，继续渲染以在 UI 中可视化最后的相机位置。
        print("Rendering will continue to render the environment and visualize the last camera positions.")
        # Define simulation stepping
        # 获取物理步长，并在 UI 循环中进行渲染与场景更新。
        sim_dt = sim.get_physics_dt()
        # Simulation loop
        while simulation_app.is_running():
            # Perform step
            sim.render()
            # Update buffers
            explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    # 入口：执行主流程，结束后关闭仿真应用。
    main()
    # Close the simulator
    simulation_app.close()
