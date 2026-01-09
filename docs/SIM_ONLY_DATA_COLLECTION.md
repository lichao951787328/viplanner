# 仿真环境数据采集（仅仿真场景）

本文档说明如何在**仅有仿真场景**（无现实设备）的条件下，快速生成训练所需的数据（深度、语义、RGB、相机内外参、视点采样等）。

## 一、前置条件
- 已安装并可运行 Isaac Sim 5.x 与 IsaacLab 2.x（使用 `isaaclab.sh` 启动）。
- 本仓库已按迁移指引完成运行（可运行 `viplanner_demo.py`）。
- 若使用自收集的仓库场景，请准备好 `Collected_warehouse`（或使用简易场景）。

## 二、环境变量（资产加载）
建议设置以下变量，保证 USD 场景及其依赖资产能正确解析：
```bash
export VIPLANNER_PROPS_DIR=/home/eai/VLN/viplanner/env/Collected_warehouse/Props
export VIPLANNER_AUTOLINK_PROPS=1
export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/Collected_warehouse/warehouse.usd
```
- 无 `keyword_mapping.yml` 时无需设置语义映射；代码已默认禁用，避免报错。
- 如需自定义输出目录，可设置 `VIPLANNER_DATA_DIR`（也可用 CLI 参数 `--save_dir`）。

## 三、快速开始（Warehouse 场景）
```bash
cd /home/eai/VLN/Viplanner_isaaclab/IsaacLab
./isaaclab.sh -p /home/eai/VLN/viplanner/omniverse/standalone/data_collect.py \
  --scene warehouse \
  --num_envs 4 \
  --num_samples 5000 \
  --save_dir /home/eai/VLN/viplanner/datasets/warehouse_sim \
  --seed 42 \
  --headless
```
- `--num_envs`：并行相机数量，受 GPU/内存影响；2–8 之间较稳。
- `--num_samples`：采样视点总数；更大意味着更多渲染轮次。
- `--save_dir`：输出数据集目录；也可以不传，使用 `VIPLANNER_DATA_DIR` 或默认目录策略。
- `--seed`：固定随机种子，便于复现。
- `--headless`：无界面渲染（速度更快，适合服务器）。

## 四、快速开始（简易场景）
如无完整仓库资产，可使用简易 USD：
```bash
export VIPLANNER_WAREHOUSE_USD=/home/eai/VLN/viplanner/env/turtlebot3_simpleroom_use_bk.usd
cd /home/eai/VLN/Viplanner_isaaclab/IsaacLab
./isaaclab.sh -p /home/eai/VLN/viplanner/omniverse/standalone/data_collect.py \
  --scene warehouse \
  --num_envs 2 \
  --num_samples 1000 \
  --save_dir /home/eai/VLN/viplanner/datasets/simple_room \
  --seed 1 \
  --headless
```

## 五、采集内容与输出结构
数据由 `ViewpointSampling` 管线生成，主要包含：
- 视点采样：在可行走区域上采样相机位姿与姿态范围（俯仰/横滚）。
- 渲染输出：
  - `depth/`：深度图（`.png` + 对应 `.npy` 浮点米制值）
  - `semantics/`：语义图（颜色编码，未知类回退为 `static`）或 `rgb/`（若选择输出 RGB）
  - `intrinsics.txt`：相机内参（按 ROS 投影矩阵 3x4 存储，逐相机顺序）
  - `camera_extrinsic.txt`：相机外参/位姿（x y z qx qy qz qw）
  - `viewpoints_seed{seed}_samples{N}.pkl`：采样的相机位姿缓存

若使用 `--save_dir`，输出会保存在该目录；否则：
- 若 `VIPLANNER_DATA_DIR` 被设置，则保存到该目录；
- 否则按默认策略在地形的 USD 路径下创建子目录（适合临时测试）。

## 六、配置细节（采样与渲染）
- 脚本位置：`omniverse/standalone/data_collect.py`
- 关键配置：`omni.viplanner.collectors.ViewpointSamplingCfg`
  - `terrain_analysis.raycaster_sensor` 默认使用 `depth_camera`
  - `cameras`：
    - Warehouse/Carla 场景：`{"depth_camera": "distance_to_image_plane", "semantic_camera": "semantic_segmentation"}`
    - 若启用 RGB，会输出到 `rgb/`（语义/RGB二选一绑定在 `semantic_camera`）
  - `depth_scale`：深度缩放为 1000（米→毫米）用于 PNG 保存；`.npy` 保留米制浮点
  - `x_angle_range`/`y_angle_range`：相机姿态采样范围（度）
  - `height`：采样高度（米）
  - `save_path`：由 `--save_dir` 或 `VIPLANNER_DATA_DIR` 覆盖

## 七、性能与质量建议
- Headless 渲染更快；`--num_envs` 根据 GPU VRAM 调整。
- 语义映射文件缺失时已自动禁用；若你有 `keyword_mapping.yml`，可通过 `VIPLANNER_SEMANTIC_MAP` 启用，提升“地面/障碍”识别。
- 光照过暗可在 `warehouse_cfg.py` 中调高 `DistantLightCfg.intensity`，或添加 DomeLight。

## 八、故障排查
- 资产缺失：确保 `VIPLANNER_PROPS_DIR` + `VIPLANNER_AUTOLINK_PROPS=1` 指向 `Collected_warehouse/Props`；或在 Isaac Sim 中 `Collect Assets/Flatten USD`。
- 黑屏/空图：确认 `--headless` 与 GPU 驱动正常；可临时减少 `--num_envs`。
- 颜色类 KeyError：已对未知类别做了 `static` 回退；可后续提供 `keyword_mapping.yml` 细化映射。

## 九、下一步
- 你可以直接用上述数据训练语义分割（可选）或用于下游规划网络的监督。
- 若需要，我可以为 `ViewpointSampling` 增加更详细的过滤/采样策略（如只采样通道中心、避让障碍边界等），以及自定义保存元数据（如场景名称、采样参数快照）。
