# VIPlanner 使用总览（数据、训练、部署）

本指南面向在 Isaac Sim 5.x + IsaacLab 2.x 环境下使用本仓库（VIPlanner）的人，涵盖数据采集、模型训练与推理部署的整体流程与关键命令。

## 架构概览
- 场景和资产：`omniverse/extension/omni.viplanner/config/*` 定义各场景（如仓库），`utils/unreal_importer.py` 负责将 USD 资产导入为可碰撞的地形与网格。
- 传感与观测：`omniverse/extension/.../mdp/observations.py` 定义相机与语义分割等观测通道；`warehouse_cfg.py` 配置相机/光照。
- 控制与规划：`viplanner/viplanner_algo.py` 负责整合感知与路径规划；`mdp/commands/path_follower_command_generator.py` 通过纯跟随控制产生底盘线速度与角速度。
- 语义分割：`Mask2Former` 子模块含语义分割训练与推理脚本，用于生成语义观测（可选）。
- 演示与采集：`omniverse/standalone/viplanner_demo.py` 运行单场景 demo；`omniverse/standalone/data_collect.py` 可用于数据采集。

## 环境与运行前准备
- IsaacLab 启动脚本位于你的 IsaacLab 仓库目录：`./isaaclab.sh -p <python脚本>`。
- 建议设置以下环境变量以确保本地资产与场景加载正常：
  - `VIPLANNER_PROPS_DIR=/home/eai/VLN/viplanner/env/Collected_warehouse/Props`
  - `VIPLANNER_AUTOLINK_PROPS=1`（将绝对路径 `/home/eai/VLN/Props` 自动指向已收集资产）
  - `VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd`（或简易场景）
  - 可选：`VIPLANNER_SEMANTIC_MAP=/path/to/keyword_mapping.yml`（若你有语义关键字映射）

## 数据采集
- 目标：采集相机图像（深度/语义）、位姿、轨迹等，用于后续训练感知与规划模型。
- 入口：`omniverse/standalone/data_collect.py` 或自定义采集脚本（也可在 `viplanner_demo.py` 中插入保存逻辑）。
- 采集内容：
  - 传感器输出（`observations.py`）定义了 `distance_to_image_plane` 与 `semantic_segmentation` 等。
  - 元信息（标签映射、类别颜色）来自 `viplanner/config/viplanner_sem_meta.py` 与你的 `keyword_mapping.yml`（若启用）。
- 样例命令：
  ```bash
  export VIPLANNER_PROPS_DIR=/home/eai/VLN/viplanner/env/Collected_warehouse/Props
  export VIPLANNER_AUTOLINK_PROPS=1
  export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd
  cd /home/eai/VLN/Viplanner_isaaclab/IsaacLab
  ./isaaclab.sh -p /home/eai/VLN/viplanner/omniverse/standalone/data_collect.py
  ```
- 数据组织建议：
  - 以场景/日期为上层目录，将图像（深度/语义/RGB）、位姿/轨迹、类别映射、相机内参保存为结构化子目录（如 `images/`, `labels/`, `poses/`, `intrinsics.json`）。

## 语义分割训练（可选）
- 组件：`Mask2Former/` 内含常见数据集配置与训练脚本。
- 入门：参考 `Mask2Former/README.md` 与 `GETTING_STARTED.md`。
- 数据准备：将采集到的语义标签整理为与 `configs/*` 兼容的格式（Cityscapes/COCO 等）；若自定义数据，建议按 COCO 格式组织。
- 常见命令（示例）：
  ```bash
  cd /home/eai/VLN/viplanner/Mask2Former
  # 安装依赖
  pip install -r requirements.txt
  # 训练（示例，具体以你的数据与配置为准）
  python train_net.py --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
    --num-gpus 2 SOLVER.IMS_PER_BATCH 16 OUTPUT_DIR ./output_coco
  # 推理
  python predict.py --config-file <your-config> --input <images-dir> --output <pred-dir> --opts MODEL.WEIGHTS <weights.pth>
  ```
- 与仿真结合：将训练好的语义模型生成的类别映射/颜色用于 `observations.py` 的重着色，或直接使用仿真引擎的语义通道。

## 路径规划与跟随训练/调参
- 路径跟随控制器：`mdp/commands/path_follower_command_generator.py` 与其配置 `path_follower_command_generator_cfg.py`。
- 关键参数：
  - `lookAheadDistance`：前视距离；越大转向越平滑，但易忽略近端障碍。
  - `dynamic_lookahead` 与 `min_points_within_lookahead`：基于路径点密度的自适应前视。
  - `maxSpeed`/`maxAccel`/`yawRateGain` 等：影响速度与转向响应；末端 `stopDisThre`/`slowDwnDisThre` 决定减速与停靠行为。
- 调参流程：从较小 `maxSpeed`（如 0.5）、适中 `lookAheadDistance`（0.8–1.5m）开始，逐步增大；在复杂场景开启 `dynamic_lookahead` 并提升 `min_points_within_lookahead`（≥5）。
- 如果你有学习型规划器/编码器（如 `viplanner/plannernet`），参考 `viplanner/train.py` 与对应数据/模型配置，先以采集数据训练，再在仿真中替换为推理模式。

## 部署与推理
- 演示脚本：`omniverse/standalone/viplanner_demo.py`。
- 使用：
  ```bash
  export VIPLANNER_PROPS_DIR=/home/eai/VLN/viplanner/env/Collected_warehouse/Props
  export VIPLANNER_AUTOLINK_PROPS=1
  export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd
  cd /home/eai/VLN/Viplanner_isaaclab/IsaacLab
  ./isaaclab.sh -p /home/eai/VLN/viplanner/omniverse/standalone/viplanner_demo.py --model_dir /home/eai/VLN/viplanner/viplanner_model
  ```
- Headless 模式：若你在无图形环境运行，图形可视化会被禁用，日志中可能出现 Graph Visualization 警告，可忽略。

## 常见问题与建议
- 资产缺失：通过 `VIPLANNER_PROPS_DIR` + `VIPLANNER_AUTOLINK_PROPS=1` 解决绝对路径引用；或在 Isaac Sim 中用 “Collect Assets/Flatten USD”。
- 语义类别不匹配：已对未知类别（如 bracket）做了 `static` 回退；建议后续补充 `keyword_mapping.yml` 提升可行走区域识别。
- 黑屏/昏暗：调整 `warehouse_cfg.py` 中的光源强度或添加 `DomeLight`；将 viewer 对准机器人 prim（`/World/envs/env_0/Robot`）。

---
后续我将按模块为你生成“逐行注释”的讲解文档，避免直接修改源码：
- 第一批：`warehouse_cfg.py`、`unreal_importer.py`、`path_follower_command_generator.py`、`observations.py`、`viplanner_demo.py`。
- 你可以告诉我优先级，我将按文件逐步生成带行级注释的副本（放在 `docs/annotated/`）。
