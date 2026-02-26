# %% [Cell 1] 基础设置与导入
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import shutil

# 路径修复 (确保能导入您的模块)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trainer_dataset import CollectData
from plannernet.autoencoder_myself_cubic_dj import AutoEncoderGrid
# 假设您已经有了 BatchTrajCost，直接导入
try:
    from test_trainer import BatchTrajCost, MapProcessor
except ImportError:
    from traj_cost_opt.traj_cost_myself_cubic import BatchTrajCost, MapProcessor

# --- 配置区域 ---
class Config:
    data_root = "/home/lichao/viplanner/rotated_out/carla/samples"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 结果保存路径
    save_dir = "./noise_test_results"
    
    max_dist = 4.0
    step_size = 1.0
    sub_step_size = 0.2
    batch_size = 8       # 增大 Batch Size 以测试更多样本
    
    # 测试步数
    total_steps = 500     # 跑 50 个 Batch
    log_interval = 5     # 每 5 个 Batch 保存一次图片
    
    # Loss 权重
    w_obs = 5.0
    w_goal = 1.2
    w_motion = 12.0
    w_guide = 2.0         
    fear_ahead_dist = 2.0
    
    # --- 开启噪声配置 (重点) ---
    data_config = {
        "enable_map_noise": True,   # 开启地图噪点/腐蚀膨胀
        "enable_ghosting": True,    # 开启鬼影（假障碍物）
        "enable_blur": True,        # 开启模糊
        "ghost_prob": 0.6,          # 60% 概率出现鬼影
        "blur_prob": 0.5            # 50% 概率模糊
    }

cfg = Config()
print(f"Device: {cfg.device}")

# 清理并重建保存目录
if os.path.exists(cfg.save_dir):
    shutil.rmtree(cfg.save_dir)
os.makedirs(cfg.save_dir)
print(f"Save directory created: {cfg.save_dir}")

# %% [Cell 2] 初始化模型与数据
# 1. 模型
model = AutoEncoderGrid(
    encoder_channel=64, 
    max_dist=cfg.max_dist, 
    step_size=cfg.step_size
).to(cfg.device)

# 2. 优化器
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 3. Cost 计算器 (使用您现有的)
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

# 5. 数据集 (开启噪声)
dataset = CollectData(
    root_dir=cfg.data_root,
    mode='train',
    config=cfg.data_config 
)
dataset.set_stage(0) 

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
data_iter = iter(loader)

print("Setup Complete. Starting Noise Loop...")

# %% [Cell 3] 循环测试与可视化
loss_history = []

for step in range(1, cfg.total_steps + 1):
    # 1. 获取数据
    try:
        batch_data = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch_data = next(data_iter)
        
    map_tensor, goal_tensor, dist_tensor, path_pixel, path_dist = batch_data
    
    # ==========================================
    # 2. 数据预处理 (包含坐标转换)
    # ==========================================
    # Map (含噪声)
    raw_np = map_tensor.squeeze(1).numpy()
    smooth_cost_np = map_proc.process_batch(raw_np) # 变成 Cost Map
    net_input = torch.from_numpy(smooth_cost_np).float().to(cfg.device)
    
    # Resize 到 80x80
    if net_input.shape[-1] != 80:
        net_input = torch.nn.functional.interpolate(net_input, size=(80,80), mode='bilinear')

    # 坐标转换参数
    map_res = 0.1
    src_h = map_tensor.shape[-2]
    target_h = 80
    scale = target_h / src_h 
    center = target_h / 2.0
    
    # A. Goal: Pixels -> Meters
    goal_pix = goal_tensor.to(cfg.device) * scale
    goal_meters = torch.zeros_like(goal_pix)
    goal_meters[:, 0] = (center - goal_pix[:, 1]) * map_res # row -> x
    goal_meters[:, 1] = (center - goal_pix[:, 0]) * map_res # col -> y
    
    goal_norm = goal_meters.clone()
    goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist
    
    # B. Guide Path: Pixels -> Meters (关键：必须缩放并转换)
    path_pixel_scaled = path_pixel.clone().to(cfg.device) * scale
    guide_meters = torch.zeros_like(path_pixel_scaled)
    guide_meters[:, :, 0] = (center - path_pixel_scaled[:, :, 1]) * map_res # row -> x
    guide_meters[:, :, 1] = (center - path_pixel_scaled[:, :, 0]) * map_res # col -> y
    
    # ==========================================
    # 3. 前向传播
    # ==========================================
    optimizer.zero_grad()
    preds, pred_fear, mask = model(net_input, goal_norm)
    
    # 生成轨迹
    waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
        preds, step=cfg.sub_step_size, mask=mask
    )
    
    # ==========================================
    # 4. 计算 Loss (传入转换后的 guide_meters)
    # ==========================================
    loss, l_traj, l_motion, _ = batch_traj_cost.CostofTraj_Batch(
        waypoints=waypoints,
        goal=goal_meters,
        fear=pred_fear,
        log_step=step,
        ahead_dist=cfg.fear_ahead_dist,
        batch_maps=net_input,
        sub_step_size=cfg.sub_step_size,
        mask=mask,
        distance=dist_tensor.to(cfg.device),
        path_pixel_raw_tensor=guide_meters, # <--- 传入 Meter 单位的引导点
        path_dist_tensor=path_dist.to(cfg.device)
    )
    
    loss.backward()
    optimizer.step()
    
    # ==========================================
    # 5. 可视化保存
    # ==========================================
    if step % cfg.log_interval == 0:
        print(f"Step {step}/{cfg.total_steps} | Loss: {loss.item():.4f}")
        
        # 为了查看更丰富的情况，我们在一个图里画出 Batch 中的前 4 个样本
        num_vis = min(4, cfg.batch_size)
        fig, axes = plt.subplots(1, num_vis, figsize=(4*num_vis, 4))
        if num_vis == 1: axes = [axes]
        
        for i in range(num_vis):
            ax = axes[i]
            
            # 背景 Cost Map (包含噪声的)
            cmap = net_input[i, 0].detach().cpu().numpy()
            ax.imshow(cmap, cmap='jet', origin='upper', extent=[0, 80, 80, 0])
            
            # 辅助转换: Meters -> Pixels (80x80)
            def to_pix(pts_m):
                if isinstance(pts_m, torch.Tensor): pts_m = pts_m.detach().cpu().numpy()
                if pts_m.ndim == 1: pts_m = pts_m[None, :]
                r = center - pts_m[:, 0] / map_res
                c = center - pts_m[:, 1] / map_res
                return c, r # x, y
            
            # A. 画预测轨迹 (白色)
            wp = waypoints[i]
            if mask is not None:
                valid_len = int((~mask[i]).sum().item() / cfg.sub_step_size) + 1
                wp = wp[:valid_len]
            wp_c, wp_r = to_pix(wp)
            ax.plot(wp_c, wp_r, 'w.-', linewidth=2, label='Pred')
            
            # B. 画 GT 引导 (黄色虚线)
            gt = guide_meters[i]
            valid_gt = torch.norm(gt, dim=1) > 0.01 # 过滤无效点
            gt = gt[valid_gt]
            gt_c, gt_r = to_pix(gt)
            ax.plot(gt_c, gt_r, 'y--', linewidth=1.5, label='GT')
            
            # C. 起终点
            g_c, g_r = to_pix(goal_meters[i])
            ax.plot(g_c, g_r, 'r*', markersize=12)
            ax.plot(40, 40, 'go', markersize=8) 
            
            if i == 0: ax.legend() # 只在第一张图显示图例
            ax.set_title(f"Sample {i}")
            ax.axis('off')
            
        plt.suptitle(f"Step {step} | Loss: {loss.item():.4f} (Noise ON)")
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(cfg.save_dir, f"step_{step:04d}.png")
        plt.savefig(save_path)
        plt.close()

print(f"Test Finished. Results saved in: {cfg.save_dir}")