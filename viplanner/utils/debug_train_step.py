# %% [Cell 1] 基础设置与导入
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trainer_dataset import CollectData
from plannernet.autoencoder_myself_cubic_dj import AutoEncoderGrid
from traj_cost_opt.traj_cost_myself_cubic import TrajCost # 确保这里能导入 BatchTrajCost
# 注意：如果 BatchTrajCost 定义在 test_trainer.py 里，你需要把它复制到一个单独的文件
# 或者为了调试，我们把 BatchTrajCost 的核心逻辑简单模拟一下，或者你直接引用 test_trainer
# 这里假设你已经把 BatchTrajCost 移到了 traj_cost_myself_cubic.py 或直接在这里定义

# 为了方便，这里重新定义一下简单的 BatchTrajCost (或者你从 test_trainer 复制过来)
# 如果你的 test_trainer.py 里有 BatchTrajCost 类，建议把它挪到单独的 .py 文件里方便引用
# 这里我假设你已经在 test_trainer.py 同级或 traj_cost_opt 里有了
try:
    from test_trainer import BatchTrajCost, MapProcessor
except ImportError:
    # 如果导入失败，请把 test_trainer.py 里的 BatchTrajCost 和 MapProcessor 类复制到这里
    print("请确保 BatchTrajCost 和 MapProcessor 可以被导入，或者直接粘贴到这个 Cell 里")

# 配置
class Config:
    data_root = "/home/lichao/viplanner/rotated_out/carla/samples"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_dist = 4.0
    step_size = 1.0
    sub_step_size = 0.2
    batch_size = 2
    
    # Loss 权重
    w_obs = 5.0
    w_goal = 1.2
    w_motion = 12.0
    w_guide = 0.2
    fear_ahead_dist = 2.0

cfg = Config()
print(f"Device: {cfg.device}")

# %% [Cell 2] 初始化模型、优化器和工具类
# 1. 模型
model = AutoEncoderGrid(
    encoder_channel=64, 
    max_dist=cfg.max_dist, 
    step_size=cfg.step_size
).to(cfg.device)

# 2. 优化器
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 3. Cost 计算器
batch_traj_cost = BatchTrajCost(
    gpu_id=0 if torch.cuda.is_available() else "cpu",
    w_obs=cfg.w_obs,
    w_motion=cfg.w_motion,
    w_goal=cfg.w_goal,
    w_guide=cfg.w_guide,
    obstalce_thread=cfg.fear_ahead_dist
)

# 4. 地图处理器
map_proc = MapProcessor()

print("Model, Optimizer, CostFunction Ready.")

# %% [Cell 3] 数据加载
dataset = CollectData(
    root_dir=cfg.data_root,
    mode='train',
    config={"enable_map_noise": True} # 开启噪声，模拟真实训练
)
dataset.set_stage(0) # 设置课程学习阶段

from torch.utils.data import DataLoader
# batch_size=2 方便对比
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
data_iter = iter(loader)
print("DataLoader Ready.")

# %% [Cell 4] 【核心调试循环】反复运行此 Cell 进行单步训练
# ==========================================
# 1. 获取数据
try:
    batch_data = next(data_iter)
except StopIteration:
    data_iter = iter(loader)
    batch_data = next(data_iter)

map_tensor, goal_tensor, dist_tensor, path_pixel, path_dist = batch_data

# ==========================================
# 2. 数据预处理 (完全复刻 train_pipeline)
# Map: [B, 1, 80, 80]
raw_np = map_tensor.squeeze(1).numpy()
smooth_cost_np = map_proc.process_batch(raw_np)
smooth_cost_tensor = torch.from_numpy(smooth_cost_np).float().to(cfg.device)

# Resize to 80x80 (网络输入)
net_input = torch.nn.functional.interpolate(
    smooth_cost_tensor, size=(80, 80), mode='bilinear', align_corners=False
).to(cfg.device)

# Goal: Pixels -> Meters
map_res = 0.1
src_h, src_w = map_tensor.shape[-2], map_tensor.shape[-1]
target_h, target_w = 80, 80
scale_r = target_h / src_h
scale_c = target_w / src_w
center_row, center_col = target_h / 2.0, target_w / 2.0

goal_col = goal_tensor[:, 0] * scale_c
goal_row = goal_tensor[:, 1] * scale_r

goal_meters = torch.zeros((cfg.batch_size, 2), dtype=torch.float32)
goal_meters[:, 0] = (center_row - goal_row) * map_res # x
goal_meters[:, 1] = (center_col - goal_col) * map_res # y
goal_meters = goal_meters.to(cfg.device)

# Goal Normalization
goal_norm = goal_meters.clone()
goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist

print(f"goal_meters: {goal_meters}")
print(f"goal_norm: {goal_norm}")
# ==========================================
# 3. 前向传播 & 轨迹生成
optimizer.zero_grad()

# A. 网络输出关键点
preds, pred_fear, mask = model(net_input, goal_norm)

print(f"preds shape: {preds.shape}")
print(f"preds: {preds}")
# B. 生成密集轨迹 (TrajGenerator)
# 这是计算 Loss 的关键，如果这里生成的轨迹很乱，Loss 就会很大
waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
    preds, step=cfg.sub_step_size, mask=mask
)
print(f"Generated waypoints shape: {waypoints.shape}")
print(f"Generated waypoints: {waypoints}")
# ==========================================
# 4. 计算 Loss
loss, l_traj, l_motion, l_fear = batch_traj_cost.CostofTraj_Batch(
    waypoints=waypoints,
    goal=goal_meters,
    fear=pred_fear,
    log_step=0, # epoch
    ahead_dist=cfg.fear_ahead_dist,
    batch_maps=net_input,
    sub_step_size=cfg.sub_step_size,
    mask=mask,
    distance=dist_tensor.to(cfg.device),
    path_pixel_raw_tensor=path_pixel.to(cfg.device),
    path_dist_tensor=path_dist.to(cfg.device)
)

# ==========================================
# 5. 反向传播
loss.backward()
optimizer.step()

print(f"Total Loss: {loss.item():.4f}")
print(f"  |-- Traj (Obs+Goal): {l_traj.item():.4f}")
print(f"  |-- Motion: {l_motion.item():.4f}")
print(f"  |-- Fear: {l_fear.item():.4f}")

# ==========================================
# 6. 可视化 (Visual Check)
# 画出 Cost Map, 预测的关键点，生成的密集轨迹
for i in range(min(cfg.batch_size, 2)):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # A. 画 Cost Map (网络看到的地图)
    cost_map_vis = net_input[i, 0].detach().cpu().numpy()
    ax.imshow(cost_map_vis, cmap='jet', origin='upper', extent=[0, 80, 80, 0])
    
    # B. 辅助函数: Meters -> Pixels
    def to_pix(pts_m):
        pts_m = pts_m.detach().cpu().numpy()
        if pts_m.ndim == 1: pts_m = pts_m[None, :]
        r = center_row - pts_m[:, 0] / map_res
        c = center_col - pts_m[:, 1] / map_res
        return c, r # x=col, y=row
    
    # C. 画预测的关键点 (Sparse)
    kp_c, kp_r = to_pix(preds[i])
    ax.plot(kp_c, kp_r, 'bs', markersize=6, label='Pred Keys')
    
    # D. 画生成的密集轨迹 (Dense - 用于计算 Loss)
    # 过滤掉 mask 掉的部分
    wp = waypoints[i]
    if mask is not None:
         # 简单估算有效长度
         valid_cnt = (~mask[i]).sum().item()
         valid_len = int(valid_cnt / cfg.sub_step_size) + 1
         wp = wp[:valid_len]
         
    wp_c, wp_r = to_pix(wp)
    ax.plot(wp_c, wp_r, 'w.-', linewidth=1, markersize=2, label='Dense Traj')
    
    # E. 画起点终点
    g_c, g_r = to_pix(goal_meters[i])
    ax.scatter(g_c, g_r, c='red', marker='*', s=150, zorder=10, label='Goal')
    ax.scatter(center_col, center_row, c='green', s=100, zorder=10, label='Start')
    
    ax.legend(loc='upper right')
    ax.set_title(f"Sample {i} | Loss: {loss.item():.2f}")
    plt.show()# %% [Cell 1] 基础设置与导入
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trainer_dataset import CollectData
from plannernet.autoencoder_myself_cubic_dj import AutoEncoderGrid
from traj_cost_opt.traj_cost_myself_cubic import TrajCost # 确保这里能导入 BatchTrajCost
# 注意：如果 BatchTrajCost 定义在 test_trainer.py 里，你需要把它复制到一个单独的文件
# 或者为了调试，我们把 BatchTrajCost 的核心逻辑简单模拟一下，或者你直接引用 test_trainer
# 这里假设你已经把 BatchTrajCost 移到了 traj_cost_myself_cubic.py 或直接在这里定义

# 为了方便，这里重新定义一下简单的 BatchTrajCost (或者你从 test_trainer 复制过来)
# 如果你的 test_trainer.py 里有 BatchTrajCost 类，建议把它挪到单独的 .py 文件里方便引用
# 这里我假设你已经在 test_trainer.py 同级或 traj_cost_opt 里有了
try:
    from test_trainer import BatchTrajCost, MapProcessor
except ImportError:
    # 如果导入失败，请把 test_trainer.py 里的 BatchTrajCost 和 MapProcessor 类复制到这里
    print("请确保 BatchTrajCost 和 MapProcessor 可以被导入，或者直接粘贴到这个 Cell 里")

# 配置
class Config:
    data_root = "/home/eai/VLN/viplanner/rotated_out/carla/samples"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_dist = 4.0
    step_size = 1.0
    sub_step_size = 0.2
    batch_size = 2
    
    # Loss 权重
    w_obs = 5.0
    w_goal = 1.2
    w_motion = 12.0
    w_guide = 1.0
    fear_ahead_dist = 2.0

cfg = Config()
print(f"Device: {cfg.device}")

# %% [Cell 2] 初始化模型、优化器和工具类
# 1. 模型
model = AutoEncoderGrid(
    encoder_channel=64, 
    max_dist=cfg.max_dist, 
    step_size=cfg.step_size
).to(cfg.device)

# 2. 优化器
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 3. Cost 计算器
batch_traj_cost = BatchTrajCost(
    gpu_id=0 if torch.cuda.is_available() else "cpu",
    w_obs=cfg.w_obs,
    w_motion=cfg.w_motion,
    w_goal=cfg.w_goal,
    w_guide=cfg.w_guide,
    obstalce_thread=cfg.fear_ahead_dist
)

# 4. 地图处理器
map_proc = MapProcessor()

print("Model, Optimizer, CostFunction Ready.")

# %% [Cell 3] 数据加载
# %% [Cell 3] 数据加载器准备
# 必须确保 Dataset 的配置与训练时一致，否则生成的路径可能不对
dataset = CollectData(
    root_dir=cfg.data_root,
    mode='train',
    config={"enable_map_noise": True} 
)
# 设置课程阶段 (Stage 0 通常包含引导路径)
dataset.set_stage(0)

from torch.utils.data import DataLoader
# batch_size=2，方便在 VS Code 右侧看两张图的对比
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# 创建迭代器，模拟 enumerate(train_loader)
data_iter = iter(loader)
print("DataLoader Ready. 每次运行 Cell 4 将抽取一个新的 Batch。")


# %% [Cell 4] 【核心调试循环】运行此 Cell 模拟一次训练迭代
# ==========================================
# 1. 获取数据 (模拟 for batch_idx, (...) in enumerate(loader))
try:
    # 关键修改：这里必须显式解包 5 个变量，与 Dataset.__getitem__ 返回值一一对应
    batch_data = next(data_iter)
    map_tensor, goal_tensor, dist_tensor, path_pixel_raw_tensor, path_dist_tensor = batch_data
except StopIteration:
    # 如果数据取完了，重新创建一个迭代器
    data_iter = iter(loader)
    batch_data = next(data_iter)
    map_tensor, goal_tensor, dist_tensor, path_pixel_raw_tensor, path_dist_tensor = batch_data

print(f"Got Batch Data:")
print(f"  - Map shape: {map_tensor.shape}")
print(f"  - Goal shape: {goal_tensor.shape}")
print(f"  - Guide Path shape: {path_pixel_raw_tensor.shape}") # 确认这里有数据 [B, N, 2]

# ==========================================
# 2. 数据预处理 (与 train_pipeline 保持一致)
# Map: [B, 1, 80, 80]
# 注意：debug 模式下如果是单进程，Tensor 可能在 CPU，需要 .to(device)
raw_np = map_tensor.squeeze(1).numpy()
smooth_cost_np = map_proc.process_batch(raw_np)
smooth_cost_tensor = torch.from_numpy(smooth_cost_np).float().to(cfg.device)

# Resize to 80x80 (网络输入)
net_input = torch.nn.functional.interpolate(
    smooth_cost_tensor, size=(80, 80), mode='bilinear', align_corners=False
).to(cfg.device)

# Goal: Pixels -> Meters (物理坐标转换)
map_res = 0.1
src_h, src_w = map_tensor.shape[-2], map_tensor.shape[-1]
target_h, target_w = 80, 80
scale_r = target_h / src_h
scale_c = target_w / src_w
center_row, center_col = target_h / 2.0, target_w / 2.0

# 假设 goal_tensor 是像素坐标 [col, row]
goal_col = goal_tensor[:, 0] * scale_c
goal_row = goal_tensor[:, 1] * scale_r

goal_meters = torch.zeros((cfg.batch_size, 2), dtype=torch.float32)
goal_meters[:, 0] = (center_row - goal_row) * map_res # x (meters)
goal_meters[:, 1] = (center_col - goal_col) * map_res # y (meters)
goal_meters = goal_meters.to(cfg.device)

# Goal Normalization
goal_norm = goal_meters.clone()
goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist

# ==========================================
# 3. 前向传播 & 轨迹生成
optimizer.zero_grad()

# A. 网络输出
preds, pred_fear, mask = model(net_input, goal_norm)

# B. 生成密集轨迹 (TrajGenerator)
waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
    preds, step=cfg.sub_step_size, mask=mask
)

# ==========================================
# 4. 计算 Loss (关键：传入 Guide Path Tensors)
# 这里必须把解包出来的 path_pixel_raw_tensor 传进去，否则 Guide Loss 无法计算
loss, l_traj, l_motion, l_fear = batch_traj_cost.CostofTraj_Batch(
    waypoints=waypoints,
    goal=goal_meters,
    fear=pred_fear,
    log_step=0, # epoch
    ahead_dist=cfg.fear_ahead_dist,
    batch_maps=net_input,
    sub_step_size=cfg.sub_step_size,
    mask=mask,
    distance=dist_tensor.to(cfg.device),
    path_pixel_raw_tensor=path_pixel_raw_tensor.to(cfg.device), # <--- 传入
    path_dist_tensor=path_dist_tensor.to(cfg.device)             # <--- 传入
)

# ==========================================
# 5. 反向传播
loss.backward()
optimizer.step()

# 打印详细 Loss
print("-" * 30)
print(f"Total Loss: {loss.item():.4f}")
print(f"  |-- Traj (Obs+Goal+Guide): {l_traj.item():.4f}") # 这里的 l_traj 包含了 Guide Loss
print(f"  |-- Motion: {l_motion.item():.4f}")
print(f"  |-- Fear: {l_fear.item():.4f}")
print("-" * 30)

# ==========================================
# 6. 可视化 (Visual Check)
# 这一步对于检查 "Guide Path 是否正确" 非常重要
for i in range(min(cfg.batch_size, 2)):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # A. 画 Cost Map
    cost_map_vis = net_input[i, 0].detach().cpu().numpy()
    ax.imshow(cost_map_vis, cmap='jet', origin='upper', extent=[0, 80, 80, 0])
    
    # 辅助函数: Meters -> Pixels
    def to_pix(pts_m):
        pts_m = pts_m.detach().cpu().numpy()
        if pts_m.ndim == 1: pts_m = pts_m[None, :]
        r = center_row - pts_m[:, 0] / map_res
        c = center_col - pts_m[:, 1] / map_res
        return c, r # x=col, y=row
    
    # B. 画网络生成的密集轨迹 (White)
    wp = waypoints[i]
    if mask is not None:
         valid_cnt = (~mask[i]).sum().item()
         valid_len = int(valid_cnt / cfg.sub_step_size) + 1
         wp = wp[:valid_len]
    wp_c, wp_r = to_pix(wp)
    ax.plot(wp_c, wp_r, 'w.-', linewidth=1.5, markersize=3, label='Pred Traj')
    
    # C. 画 Ground Truth 引导路径 (Yellow/Cyan)
    # path_pixel_raw_tensor 是像素坐标，直接画
    gt_path = path_pixel_raw_tensor[i].numpy() 
    # 注意：DataSet 里返回的 path_pixel_raw 通常是 [col, row] 或 [row, col]
    # 取决于 trainer_dataset.py 里的实现。
    # cv2 习惯是 (x,y)，numpy 是 (row, col)。
    # 假设是 (col, row) (图像坐标 x, y)
    gt_c = gt_path[:, 0] # x
    gt_r = gt_path[:, 1] # y
    
    # 过滤掉 padding 的 0 (如果 path 是用 0 填充的)
    # 简单的过滤：不画 (0,0) 点，或者根据 path_dist_tensor 截断
    valid_len_gt = (path_dist_tensor[i] > 0).sum().item() # 粗略估计
    if valid_len_gt > 0:
        ax.plot(gt_c[:valid_len_gt], gt_r[:valid_len_gt], 'y--', linewidth=2, label='Guide Path (GT)')

    # D. 起点终点
    g_c, g_r = to_pix(goal_meters[i])
    ax.scatter(g_c, g_r, c='red', marker='*', s=150, zorder=10, label='Goal')
    ax.scatter(center_col, center_row, c='green', s=100, zorder=10, label='Start')
    
    ax.legend(loc='upper right')
    ax.set_title(f"Sample {i} | Traj Loss (Inc. Guide): {l_traj.item():.2f}")
    plt.show()