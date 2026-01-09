# 逐行注释版：warehouse_cfg.py
# 说明：这是副本文件，仅用于学习，不参与运行。

# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)  # 版权信息
# Author: Pascal Roth                                           # 作者
# All rights reserved.                                          # 版权声明
#
# SPDX-License-Identifier: BSD-3-Clause                         # 许可证

import os  # 导入操作系统相关工具

import isaaclab.sim as sim_utils  # IsaacLab 仿真工具（光源、地面、相机等）
from isaaclab.assets import AssetBaseCfg  # 基础资产配置类（如光源）
from isaaclab.scene import InteractiveSceneCfg  # 交互式场景配置基类
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns  # 传感器配置与采样模式
from isaaclab.utils import configclass  # 装饰器：将类作为配置类使用
from omni.viplanner.utils import UnRealImporterCfg  # 自定义的 USD 地形导入配置

from ..viplanner import DATA_DIR  # 数据目录（仓库中的资源路径）
from .base_cfg import ViPlannerBaseCfg  # 环境基类配置

##
# 预定义配置
##
# isort: off  # 排序工具忽略
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # 机器人资产配置（ANYMAL-C）

##
# 场景定义
##

@configclass  # 声明为配置类，可与 IsaacLab 的配置系统协作
class TerrainSceneCfg(InteractiveSceneCfg):  # 地形场景配置（含机器人）
    """Configuration for the terrain scene with a legged robot."""  # 文档串

    # 地形（从 USD 导入）
    terrain = UnRealImporterCfg(  # 使用自定义导入器读取地形网格
        prim_path="/World/Warehouse",  # 地形在场景中的基础路径
        physics_material=sim_utils.RigidBodyMaterialCfg(  # 物理材质（摩擦、弹性等）
            friction_combine_mode="multiply",  # 摩擦组合模式
            restitution_combine_mode="multiply",  # 恢复系数组合模式
            static_friction=1.0,  # 静摩擦
            dynamic_friction=1.0,  # 动摩擦
        ),
        # 可通过环境变量覆盖 USD 路径，以便调试或使用本地收集的资产
        usd_path=os.environ.get(
            "VIPLANNER_WAREHOUSE_USD",  # 环境变量：指定仓库主 USD 文件路径
            os.path.join(DATA_DIR, "warehouse", "warehouse_new.usd"),  # 默认内置路径
        ),
        groundplane=True,  # 创建不可见地平面作为兜底碰撞
        # 语义映射：仅当显式提供时启用，避免缺失文件导致错误
        sem_mesh_to_class_map=(
            os.environ.get("VIPLANNER_SEMANTIC_MAP")  # 提供 keyword_mapping.yml 时启用
            if os.environ.get("VIPLANNER_SEMANTIC_MAP") and not os.environ.get("VIPLANNER_DISABLE_SEMANTICS")
            else None  # 未提供时禁用
        ),
        people_config_file=(
            None  # 若禁用人物注入
            if os.environ.get("VIPLANNER_DISABLE_PEOPLE")
            else os.path.join(DATA_DIR, "warehouse", "people_cfg.yml")  # 默认人物配置（可选）
        ),
        axis_up="Z",  # USD 文件的上轴设置（Z 向上）
    )

    # 机器人配置（替换 prim 路径到环境实例）
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # 机器人在每个环境的路径
    robot.init_state.pos = (5.0, 5.5, 0.6)  # 初始位置（x, y, z）
    robot.init_state.rot = (0.5253, 0.0, 0.0, 0.8509)  # 初始旋转（四元数）

    # 传感器：高度扫描（射线）
    height_scanner = RayCasterCfg(  # 射线扫描器配置
        prim_path="{ENV_REGEX_NS}/Robot/base",  # 挂载在机器人底座
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 相对偏移
        ray_alignment="yaw",  # 射线方向与偏航轴对齐（替代已弃用选项）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格采样密度与范围
        debug_vis=True,  # 调试可视化
        mesh_prim_paths=["/World/GroundPlane"],  # 参与高度扫描的网格路径（兜底地面）
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, debug_vis=False)  # 接触力

    # 光照设置
    light = AssetBaseCfg(  # 光照资产配置
        prim_path="/World/light",  # 光源 prim 路径
        spawn=sim_utils.DistantLightCfg(  # 平行光（远光源）
            color=(1.0, 1.0, 1.0),  # 颜色
            intensity=1000.0,  # 光强（可根据黑暗程度调整）
        ),
    )

    # 相机：深度
    depth_camera = CameraCfg(  # 深度相机配置
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera",  # 相机 prim 路径
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),  # 安装位姿
        spawn=sim_utils.PinholeCameraCfg(),  # 小孔相机模型
        width=848,  # 分辨率宽
        height=480,  # 分辨率高
        data_types=["distance_to_image_plane"],  # 输出深度到像平面距离
    )

    # 相机：语义（以及可选 RGB）
    semantic_camera = CameraCfg(  # 语义相机配置
        prim_path="{ENV_REGEX_NS}/Robot/base/semantic_camera",  # 相机 prim 路径
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),  # 安装位姿
        spawn=sim_utils.PinholeCameraCfg(),  # 小孔相机模型
        width=1280,  # 分辨率宽
        height=720,  # 分辨率高
        data_types=["semantic_segmentation", "rgb"],  # 输出语义分割与 RGB 图像
        colorize_semantic_segmentation=False,  # 由管线自行重着色（便于自定义颜色）
    )

##
# 环境配置
##

@configclass  # 环境整体配置类
class ViPlannerWarehouseCfg(ViPlannerBaseCfg):  # 速度跟踪任务的环境配置
    """Configuration for the locomotion velocity-tracking environment."""  # 文档串

    # 场景设置：一个环境实例，环境间距 1.0
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0)  # 单环境部署

    def __post_init__(self):  # 初始化结束时的钩子
        """Post initialization."""  # 文档串
        super().__post_init__()  # 调用父类后处理
        # 调整查看器相机（眼睛位置与注视点）
        self.viewer.eye = (5, 12, 5)  # 观察位置
        self.viewer.lookat = (5, 0, 0.0)  # 注视点
