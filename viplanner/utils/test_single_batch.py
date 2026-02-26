# %% [Cell 1] 基础设置与导入
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 路径修复 (根据你的实际情况调整)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trainer_dataset import CollectData
from plannernet.autoencoder_myself_cubic_dj import AutoEncoderGrid
# 假设你已经把 BatchTrajCost 放在合适的位置，或者从 test_trainer 导入
try:
    from test_trainer import BatchTrajCost, MapProcessor
except ImportError:
    # 备用导入路径，根据你的实际文件结构
    from traj_cost_opt.traj_cost_myself_cubic import BatchTrajCost, MapProcessor

# --- 配置区域 ---
class Config:
    data_root = "/home/lichao/viplanner/rotated_out/carla/samples"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_dist = 4.0
    step_size = 1.0
    sub_step_size = 0.2
    
    # 这里的 Batch Size 不需要太大，2-4个样本足够测试过拟合
    batch_size = 16 
    
    # 训练参数
    learning_rate = 1e-3 # 测试时可以用稍大的 LR
    num_epochs = 500     # 跑 500 轮，通常足够让 Loss 收敛
    plot_interval = 100  # 每 100 轮画一次图
    
    # Loss 权重 (保持与你设定的一致)
    w_obs = 5.0
    w_goal = 5
    w_motion = 10.0
    w_guide = 0.2
    # w_obs = 0.0
    # w_goal = 0.0
    # w_motion = 0.0
    # w_guide = 10.0
    fear_ahead_dist = 2.0

cfg = Config()
print(f"Device: {cfg.device}")

# %% [Cell 2] 准备模型与固定数据
# 1. 初始化模型
model = AutoEncoderGrid(
    encoder_channel=64, 
    max_dist=cfg.max_dist, 
    step_size=cfg.step_size
).to(cfg.device)

# 2. 优化器
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

# 3. Cost 函数
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

# 5. 加载数据 (关键：只取一个 Batch)
# 注意：测试过拟合时，建议关闭 map_noise，保证输入是恒定的
dataset = CollectData(
    root_dir=cfg.data_root, 
    mode='train', 
    config={"enable_map_noise": False} 
)
dataset.set_stage(0)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# ---【关键步骤】获取并固定这一个 Batch 的数据 ---
data_iter = iter(loader)
fixed_batch = next(data_iter)

# 解包数据
map_tensor_raw, goal_tensor_raw, dist_tensor, path_pixel_raw, path_dist = fixed_batch
# === 【在这里插入排查代码，只运行一次】 ===
guide_data = path_pixel_raw.detach().cpu().numpy()
print(f"\n====== Data Inspection ======")
print(f"Max Value: {np.max(guide_data):.4f}")
print(f"Sample 0 Head: {guide_data[0, :3]}")
print(f"Sample 0 Tail: {guide_data[0, -3:]}")

if np.max(guide_data) > 10.0:
    print(">>> 结论：当前Batch是【像素坐标】(Pixels)！")
    print(">>> 原因：A*路径太短，触发了Dataset逻辑漏洞，未转换为米。")
    print(">>> 结果：Loss爆炸，轨迹射向边界。")
elif np.max(guide_data) > 0.1:
    print(">>> 结论：当前Batch是【物理坐标】(Meters)。")
    print(">>> 结果：如果是这种情况，轨迹应该能正常训练。")
else:
    print(">>> 结论：数据异常（全0或过小）。")
# ========================================
# print("goal_tensor_raw:", goal_tensor_raw)
# print("path_pixel_raw:", path_pixel_raw)
# print("path_dist:", path_dist)
max_val = path_pixel_raw.max().item()
# print(f"【排查】Guide Path 最大值: {max_val:.4f}")
# if max_val > 10.0:
#     print(">>> 警报！检测到 PIXEL 坐标 (数值过大)，Dataset 输出错误！")
# else:
#     print(">>> 正常。检测到 METER 坐标 (数值 < 10.0)。")

# print("Fixed Batch Data Loaded. Ready for Overfitting Test.")
# print(f"Map Shape: {map_tensor_raw.shape}")

# %% [Cell 3] 训练循环 (Overfitting Loop)

# 为了节省计算资源，我们把固定的预处理放到循环外
# 1. 预处理 Map
raw_np = map_tensor_raw.squeeze(1).numpy()
smooth_cost_np = map_proc.process_batch(raw_np)
smooth_cost_tensor = torch.from_numpy(smooth_cost_np).float().to(cfg.device)
# Resize
net_input = torch.nn.functional.interpolate(
    smooth_cost_tensor, size=(80, 80), mode='bilinear', align_corners=False
).to(cfg.device)

# 2. 预处理 Goal (Pixels -> Meters)
map_res = 0.1
target_h, target_w = 80, 80
src_h, src_w = map_tensor_raw.shape[-2], map_tensor_raw.shape[-1]
scale_r, scale_c = target_h / src_h, target_w / src_w
center_row, center_col = target_h / 2.0, target_w / 2.0

goal_col = goal_tensor_raw[:, 0] * scale_c
goal_row = goal_tensor_raw[:, 1] * scale_r

goal_meters = torch.zeros((cfg.batch_size, 2), dtype=torch.float32)
goal_meters[:, 0] = (center_row - goal_row) * map_res # x
goal_meters[:, 1] = (center_col - goal_col) * map_res # y
goal_meters = goal_meters.to(cfg.device)

goal_norm = goal_meters.clone()
goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist
goal_norm = goal_norm.to(cfg.device)

# 3. 移动其他 Tensor 到 GPU
dist_tensor = dist_tensor.to(cfg.device)
path_pixel_raw = path_pixel_raw.to(cfg.device)
path_dist = path_dist.to(cfg.device)

print("Starting Training Loop...")
loss_history = []

for epoch in range(1, cfg.num_epochs + 1):
    optimizer.zero_grad()
    
    # --- Forward ---
    preds, pred_fear, mask = model(net_input, goal_norm)
    
    # --- Trajectory Generation ---
    waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
        preds, step=cfg.sub_step_size, mask=mask
    )
    
    # --- Loss Calculation ---
    loss, l_traj, l_motion, l_fear = batch_traj_cost.CostofTraj_Batch(
        waypoints=waypoints,
        goal=goal_meters,
        fear=pred_fear,
        log_step=epoch, 
        ahead_dist=cfg.fear_ahead_dist,
        batch_maps=net_input,
        sub_step_size=cfg.sub_step_size,
        mask=mask,
        distance=dist_tensor,
        path_pixel_raw_tensor=path_pixel_raw,
        path_dist_tensor=path_dist
    )
    
    # --- Backward ---
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
    optimizer.step()
    
    loss_history.append(loss.item())
    
    # 简单的日志打印
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss: {loss.item():.5f} "
              f"(Traj: {l_traj.item():.3f}, Motion: {l_motion.item():.3f}, Fear: {l_fear.item():.3f})")

    # --- 可视化 (Visual Check) ---
    if epoch % cfg.plot_interval == 0 or epoch == cfg.num_epochs:
        print(f"Visualize at Epoch {epoch}...")
        
        # 创建包含所有样本的子图，每行最多4个
        num_samples = net_input.shape[0]
        ncols = min(4, num_samples)  # 每行最多4个
        nrows = (num_samples + ncols - 1) // ncols  # 向上取整
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        
        # 统一处理 axes 为二维数组
        if num_samples == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx in range(num_samples):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            # A. Cost Map
            cost_map_vis = net_input[idx, 0].detach().cpu().numpy()
            ax.imshow(cost_map_vis, cmap='jet', origin='upper', extent=[0, 80, 80, 0])
            
            # 坐标转换函数
            def to_pix(pts_m):
                pts_m = pts_m.detach().cpu().numpy()
                if pts_m.ndim == 1: pts_m = pts_m[None, :]
                r = center_row - pts_m[:, 0] / map_res
                c = center_col - pts_m[:, 1] / map_res
                return c, r # x, y

            # B. 预测轨迹 (White)
            wp = waypoints[idx]
            # 简单截断无效部分用于显示
            if mask is not None:
                valid_cnt = (~mask[idx]).sum().item()
                valid_len = int(valid_cnt / cfg.sub_step_size)
                wp = wp[:valid_len]
            
            wp_c, wp_r = to_pix(wp)
            ax.plot(wp_c, wp_r, 'w.-', linewidth=2, label='Pred')
            
            # C. Ground Truth Guide Path (Yellow)
            gt_path = path_pixel_raw[idx].cpu().numpy()
            # 缩放因子
            s_c = 80.0 / map_tensor_raw.shape[-1]
            s_r = 80.0 / map_tensor_raw.shape[-2]
            # 假设 gt_path 是 [x, y] (col, row)
            gt_c = gt_path[:, 0] * s_c
            gt_r = gt_path[:, 1] * s_r
            # 简单的去除 (0,0) 点
            valid_gt = (np.abs(gt_c) > 0.1) | (np.abs(gt_r) > 0.1)
            if valid_gt.any():
                ax.plot(gt_c[valid_gt], gt_r[valid_gt], 'y--', linewidth=1.5, label='GT')

            # D. Goal & Start
            g_c, g_r = to_pix(goal_meters[idx])
            ax.scatter(g_c, g_r, c='red', marker='*', s=150, zorder=10, label='Goal')
            ax.scatter(center_col, center_row, c='green', s=100, zorder=10, label='Start')
            
            ax.legend()
            ax.set_title(f"Sample {idx} | Loss: {loss.item():.4f}")
        
        # 隐藏多余的子图
        for idx in range(num_samples, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        fig.savefig(f'/home/lichao/viplanner/rotated_out/carla/overfit_results_guide/epoch_{epoch:04d}.png', 
                            dpi=100, bbox_inches='tight')
        plt.close(fig)

print("Overfitting Test Finished.")