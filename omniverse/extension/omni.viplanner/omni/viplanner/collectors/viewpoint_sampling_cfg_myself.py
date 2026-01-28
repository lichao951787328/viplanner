'''
Author: lichao951787328 951787328@qq.com
Date: 2025-12-31 14:52:20
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-16 14:10:25
FilePath: /viplanner/omniverse/extension/omni.viplanner/omni/viplanner/collectors/viewpoint_sampling_cfg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .terrain_analysis_cfg import TerrainAnalysisCfg


@configclass
class ViewpointSamplingCfg:
    """Configuration for the viewpoint sampling.

    视点采样（viewpoint sampling）的配置项。
    该配置用于数据采集阶段：基于地形分析结果，从可行区域随机采样相机位姿，
    并为指定的相机导出对应的标注（例如深度、语义分割等）。
    """

    terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis.

    用于地形分析（TerrainAnalysis）的传感器/相机名称。
    - 默认使用 `camera_0` 作为射线投射（raycast）的参考传感器。
    - 这会影响高度图构建、样本点合法性校验等步骤。
    - 生效: 是（用于 TerrainAnalysis）
    """

    verify_viewpoint_clearance: bool = False  # 是否在渲染前验证视点中心净空
    clearance_eps: float = 0.02  # 地面上方起点的小偏移，避免自碰撞

    # dict of cameras and corresponding annotators
    cameras: dict[str, str] = {
        "camera_0": "distance_to_image_plane",
        "camera_1": "semantic_segmentation",
    }
    """Dict of cameras and corresponding annotators to use for the viewpoint sampling.

    需要参与渲染/标注导出的相机与其对应的“标注器”类型。
    - key: 场景中相机的名称（例如 `camera_0`、`camera_1`）。
    - value: 要导出的标注类型（annotator）。常见示例：
      - "distance_to_image_plane": Z 深度（到成像平面的距离，单位米，后续可用 `depth_scale` 缩放）。
      - "semantic_segmentation": 语义分割（每像素类别索引/颜色等）。
      - 也可以扩展为其他 annotator（例如 RGB）。
    - 生效: 是（用于渲染与落盘时选择通道/目录）
    """
    depth_scale: float = 1000.0
    """Scaling factor for the depth values.

    深度缩放因子。用于将渲染得到的深度（通常为米）缩放到目标单位（例如毫米）。
    - 例如设为 1000.0 时：depth_mm = depth_m * 1000。
    - 仅对深度标注生效，对语义等其他通道无影响。
    - 生效: 是（保存深度 PNG/NPY 时应用）
    """

    # sampling 实际没有使用到
    sample_points: int = 10000
    """Number of random points to sample.

    采样候选视点（落在可通行区域）数量上限。实际有效数量受一系列过滤器影响：
    - 网格内外/墙体过滤、与障碍物的安全距离过滤、语义代价过滤等。
    - 最终有效点数将不超过该值。
    - 生效: 否（当前流程未使用该顶层字段；实际使用的是 TerrainAnalysisCfg.sample_points）
    """
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution

    相机姿态采样的欧拉角范围（单位：度），按均匀分布随机选取：
    - x_angle_range: 绕 x 轴（pitch/俯仰）范围。
    - y_angle_range: 绕 y 轴（yaw/偏航）范围。注：在 Isaac 约定下，负角度表示“向下看”。
    - 通常结合具体相机安装姿态进行小范围微扰，避免出现不合理的极端视角。
    - 生效: 是（采样视点时用于姿态扰动）
    """
    height: float = 0.5
    """Height to use for the random points.

    随机视点的相机高度（相对于地面高度网格），单位米。
    - 实际采样流程会先求地面高度，再在其上方 `height` 处放置相机中心。
    - 生效: 否（当前实现未使用；相机高度由 TerrainAnalysisCfg.robot_height 决定）
    """

    # SAVING
    save_path: str | None = None
    """Directory to save the viewpoint samples, camera intrinsics and rendered images to.

    输出保存目录：用于保存视点采样结果、相机内参/外参以及渲染得到的图像。
    - 若为 None，则默认使用场景/资源所在目录（实现依赖调用方约定）。
    - 建议显式设置，便于批量数据管理与复现。
    - 生效: 是（控制缓存与输出位置）
    """

    # debug
    debug_viz: bool = True
    """Whether to visualize the sampled points and orientations.

    是否在仿真界面中可视化采样出的相机位置与朝向，便于调参与排错。
    - Headless 模式下可能不可用；若使用 USD/BasisCurves 等兜底可视化，效果以实现为准。
    - 生效: 是（控制是否绘制绿色箭头）
    """
