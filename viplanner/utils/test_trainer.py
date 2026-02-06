import os
import sys
import time
import cv2
import torch
import torch.optim as optim
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
import shutil
import pdb
import heapq
Debug = False

# ==========================================
# 1. 路径修复 (关键步骤)
# ==========================================

# 获取当前脚本 (test_trainer.py) 所在的目录 -> .../viplanner/utils
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录 (父目录) -> .../viplanner
parent_dir = os.path.dirname(current_dir)

# 将父目录加入 Python 搜索路径
# 这样 Python 就能看到 'plannernet', 'traj_cost_opt' 等文件夹了
sys.path.append(parent_dir)

print(f"[INFO] Running from: {current_dir}")
print(f"[INFO] Added parent path: {parent_dir}")

# ==========================================
# 2. 导入模块
# ==========================================
try:
    # A. 导入同级目录下的文件 (假设 trainer_dataset.py 也在 utils 里)
    from trainer_dataset import CollectData

    # B. 导入父目录下的模块 (现在可以直接从文件夹名开始了)
    # 对应: .../viplanner/plannernet/autoencoder_myself_cubic.py
    from plannernet.autoencoder_myself_cubic_dj import AutoEncoderGrid
    
    # 对应: .../viplanner/traj_cost_opt/traj_cost_myself_cubic.py
    from traj_cost_opt.traj_cost_myself_cubic import TrajCost


except ImportError as e:
    print(f"\n[Error] 导入失败: {e}")
    print("-" * 30)
    print("调试建议:")
    print("1. 确认 trainer_dataset.py 是否在 utils 文件夹内？")
    print(f"2. 确认 {parent_dir}/plannernet 文件夹是否存在？")
    print(f"3. 确认 {parent_dir}/traj_cost_opt 文件夹是否存在？")
    sys.exit(1)


# ==========================================
# 1. 配置类 (保留您的原始设计)
# ==========================================
class Config:
    def __init__(self):
        # 路径设置
        self.data_root = "/home/eai/VLN/viplanner/rotated_out/carla/samples"  # 数据路径
        self.save_dir = "/home/eai/VLN/viplanner/rotated_out/carla/checkpoints_v2"      # 模型保存路径
        self.log_dir = "/home/eai/VLN/viplanner/rotated_out/carla/logs_v2"              # 可视化结果保存路径
        
        # 训练超参数
        self.epochs = 125
        self.batch_size = 64    # 根据显存大小调整，取2先试试
        self.lr = 5e-4
        self.num_workers = 4
        
        # 地图相关
        self.map_size = 80
        self.max_dist = 4.0    # 预测最远距离
        self.step_size = 0.5    # 轨迹点间隔 (注意：您改成了 0.3)
        self.sub_step_size = 0.2  # 子轨迹点间隔, 用于计算更精细的成本
        self.resolution = 0.1   # 地图分辨率 (0.1m/pixel)
        
        # 损失权重 (传给 TrajCost)
        self.w_obs = 2        # 障碍物避让权重
        self.w_goal = 6.0       # 到达目标权重
        self.w_motion = 1     # 平滑/动态权重
        self.fear_ahead_dist = 2.0
        
        self.robot_width = 0.6     # 机器人宽度 (米)，用于计算左右边界点
        global Debug
        self.debug = Debug           # 是否开启调试模式 (打印更多信息，保存中间结果等)
        
        # 数据增强配置
        self.train_aug_config = {
            "enable_map_noise": True, 
            "enable_ghosting": True, 
            "enable_blur": True,
            "ghost_prob": 0.8,  # 您指定的高概率
            "blur_prob": 0.8
        }
        self.val_aug_config = {
            "enable_map_noise": False, 
            "enable_ghosting": False, 
            "enable_blur": False
        }

def dijkstra_length(cost_map, start_rc, goal_rc, res=0.1, occ_thresh=0.6):
    h, w = cost_map.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if cost_map[sr, sc] > occ_thresh or cost_map[gr, gc] > occ_thresh:
        return None

    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]

    nbrs = [
        (-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
        (-1, -1, res * np.sqrt(2)), (-1, 1, res * np.sqrt(2)),
        (1, -1, res * np.sqrt(2)), (1, 1, res * np.sqrt(2)),
    ]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d != dist[r, c]:
            continue
        if (r, c) == (gr, gc):
            return d
        for dr, dc, wgt in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and cost_map[nr, nc] <= occ_thresh:
                nd = d + wgt
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    return None
        
        
# ==========================================
# 2. 核心修复: 支持 Batch 的 Cost 计算器
# ==========================================
class BatchTrajCost(TrajCost):
    """
    继承自 TrajCost，重写核心函数以支持 Batch 地图输入。
    解决单张地图无法对应 Batch 数据的问题。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def Pos2IndNormal_Batch(self, points, H, W):
        """
        将物理坐标转换为归一化的采样坐标 [-1, 1] for grid_sample
        假设地图中心对应物理坐标 (0,0)
        """
        # 假设 resolution = 0.1, center = (H/2, W/2)
        # 坐标系: x向前(Row减), y向左(Col减)
        res = 0.1
        c_row, c_col = H / 2.0, W / 2.0
        
        px = points[..., 0]  # x (Front)
        py = points[..., 1]  # y (Left)
        
        # 物理 -> 像素坐标
        row_idx = c_row - (px / res)
        col_idx = c_col - (py / res)
        
        # 像素 -> 归一化 [-1, 1] (grid_sample 使用 x=col, y=row)
        norm_col = 2.0 * (col_idx / (W - 1)) - 1.0
        norm_row = 2.0 * (row_idx / (H - 1)) - 1.0
        
        # Stack 为 (x, y) 即 (col, row)
        grid = torch.stack((norm_col, norm_row), dim=-1)
        return torch.clamp(grid, -1.0, 1.0)

    def CostofTraj_Batch(self, waypoints, goal, fear, log_step, ahead_dist, batch_maps, sub_step_size, mask=None):
        """
        Args:
            waypoints: [B, Max_Dense_Len, 2] 已经插值并补齐的密集轨迹
            goal: [B, 2] 物理坐标目标点
            mask: [B, N_keypoints] 原始关键点的 Mask (True=Invalid/Padding)
            sub_step_size: 插值步长 (cfg.sub_step_size)，用于推算密集轨迹的有效长度
        """
        batch_size, max_dense_len, _ = waypoints.shape
        device = waypoints.device

        if torch.isnan(waypoints).any():
            print("[Error] Waypoints contain NaNs! Model outputs might be exploding.")
            # 返回一个带梯度的伪 Loss 防止程序直接 Crash，或者选择 raise Error
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # ============================================================
        # 1. 计算密集轨迹的 Valid Length 和 Mask
        # ============================================================
        if mask is not None:
            # A. 计算关键点的有效数量 (Keypoints Valid Count)
            # mask True 为无效，取反求和
            valid_key_cnt = (~mask).sum(dim=1).float() # [B]
            global Debug
            if Debug:
                print(f"[DEBUG] valid_key_cnt: {valid_key_cnt}")
                pdb.set_trace()
            # B. 推算插值后的有效数量 (Dense Valid Count)
            # mask不包含起点，所以这里不加1
            valid_dense_len = ((valid_key_cnt) / sub_step_size).long() + 1
            
            if Debug:
                print(f"[DEBUG] valid_dense_len: {valid_dense_len}")
            # 钳制一下，防止计算误差导致越界
            valid_dense_len = torch.clamp(valid_dense_len, min=2, max=max_dense_len)
            if Debug:
                print(f"[DEBUG] valid_dense_len after clamping: {valid_dense_len}")
            
            # C. 生成密集轨迹的 Mask (True = Invalid/Padding)
            # [B, Max_Dense_Len]
            idx_range = torch.arange(max_dense_len, device=device).expand(batch_size, max_dense_len)
            dense_mask = idx_range >= valid_dense_len.unsqueeze(1)
        else:
            valid_dense_len = torch.full((batch_size,), max_dense_len, device=device, dtype=torch.long)
            dense_mask = torch.zeros((batch_size, max_dense_len), dtype=torch.bool, device=device)
        # global Debug
        if Debug:
            print(f"[DEBUG] waypoints shape: {waypoints.shape}, dtype: {waypoints.dtype}")
            print(f"[DEBUG] waypoints:\n{waypoints}")
            print(f"[DEBUG] mask shape: {mask.shape if mask is not None else None}")
            print(f"[DEBUG] mask:\n{mask if mask is not None else None}")
            print(f"[DEBUG] valid_dense_len: {valid_dense_len}")
            print(f"[DEBUG] dense_mask shape: {dense_mask.shape}")
            print(f"[DEBUG] dense_mask:\n{dense_mask}")
            # import pdb;
            pdb.set_trace()

        # ============================================================
        # 2. Obstacle Loss (屏蔽 Padding 区域)
        # ============================================================
        # 获取每个点的 Cost [B, N]
        oloss_per_point = self._compute_oloss_batch(waypoints, batch_maps)
        dense_mask_3x = torch.cat([dense_mask, dense_mask, dense_mask], dim=0)
        
        if Debug:
            print(f"[DEBUG] oloss_per_point shape: {oloss_per_point.shape}")
            print(f"[DEBUG] oloss_per_point:\n{oloss_per_point}")
            pdb.set_trace()
        
        # 将 Mask 部分的 Cost 置为 0
        oloss_per_point = oloss_per_point.masked_fill(dense_mask_3x, 0.0)
        
        if Debug:
            print(f"[DEBUG] oloss_per_point after masking:\n{oloss_per_point}")
            pdb.set_trace()

        valid_len_3x = torch.cat([valid_dense_len, valid_dense_len, valid_dense_len], dim=0).float()
        
        # 求和并除以该轨迹的有效长度 (平均 Cost)
        # 避免除以 0，加个 epsilon
        oloss = torch.sum(oloss_per_point, dim=1) / (valid_len_3x + 1e-6)
        oloss = torch.mean(oloss)

        # ============================================================
        # 3. Goal Loss (取真正的最后一个有效点)
        # ============================================================
        # 不能直接取 waypoints[:, -1]，因为那里可能是补齐的 0
        # 我们利用 valid_dense_len 找到每个 Batch 真实的最后一个点的索引
        last_indices = (valid_dense_len - 1).clamp(min=0, max = max_dense_len - 1)  # [B]
        
        # 使用 gather 提取每个样本对应的终点
        # gather 需要 indices 维度匹配: [B, 1, 2]
        gather_idx = last_indices.view(batch_size, 1, 1).expand(-1, -1, 2)
        real_endpoints = torch.gather(waypoints, 1, gather_idx).squeeze(1)  # [B, 2]
        
        if Debug:
            print(f"[DEBUG] waypoints shape: {waypoints.shape}")
            print(f"[DEBUG] waypoints:\n{waypoints}")
            print(f"[DEBUG] last_indices: {last_indices}")
            print(f"[DEBUG] real_endpoints shape: {real_endpoints.shape}")
            print(f"[DEBUG] real_endpoints:\n{real_endpoints}")
            pdb.set_trace()
        
        gloss_M = torch.norm(goal[:, :2] - real_endpoints, dim=1)
        gloss = torch.mean(gloss_M ** 2.0)  # 乘以2是为了让数值更大一些，您可以根据实际情况调整

        # ============================================================
        # 4. Motion Loss (平滑度：基于动态有效长度计算)
        # ============================================================
        # 计算相邻点距离 (实际步长) [B, N-1]
        wp_ds = torch.norm(waypoints[:, 1:] - waypoints[:, :-1], dim=2)
        
        if Debug:
            print(f"[DEBUG] wp_ds shape: {wp_ds.shape}")
            print(f"[DEBUG] wp_ds:\n{wp_ds}")
            pdb.set_trace()
        # 计算"理想步长" (Ideal Step Size)
        # 理想情况下，从起点(0,0)到目标(goal)应该是均匀分布的直线
        # 理想步长 = (0到Goal的距离) / (有效段数)
        # 注意：这里假设 waypoints 起点大约是 (0,0)
        dist_to_goal = torch.norm(goal, dim=1)

        with torch.no_grad():
            cost_maps_np = batch_maps.detach().cpu().numpy()[:, 0]
            map_res = 0.1
            dist_list = []
            for i in range(batch_size):
                h, w = cost_maps_np[i].shape
                c_row = h // 2
                c_col = w // 2
                g_row = int(round(c_row - goal[i, 0].item() / map_res))
                g_col = int(round(c_col - goal[i, 1].item() / map_res))
                g_row = int(np.clip(g_row, 0, h - 1))
                g_col = int(np.clip(g_col, 0, w - 1))
                d = dijkstra_length(cost_maps_np[i], (c_row, c_col), (g_row, g_col), res=map_res)
                dist_list.append(d if d is not None else dist_to_goal[i].item())
            dist_to_goal = torch.tensor(dist_list, device=device, dtype=waypoints.dtype) 
        
        num_segments = (valid_dense_len - 1).float().clamp(min=1.0)
        if Debug:
            print(f"[DEBUG] dist_to_goal: {dist_to_goal}")
            print(f"[DEBUG] num_segments: {num_segments}")
            pdb.set_trace()
        ideal_step = (dist_to_goal / num_segments).unsqueeze(1) # [B, 1]
        if Debug:
            print(f"[DEBUG] ideal_step: {ideal_step}")
            pdb.set_trace()
        # 损失 = abs(实际步长 - 理想步长)
        mloss_per_step = torch.abs(wp_ds - ideal_step)
        if Debug:
            print(f"[DEBUG] mloss_per_step shape: {mloss_per_step.shape}")
            print(f"[DEBUG] mloss_per_step:\n{mloss_per_step}")
            pdb.set_trace()
        # Mask 掉无效的段 (dense_mask[:, 1:] 对应 N-1 个段)
        seg_mask = dense_mask[:, 1:]
        mloss_per_step = mloss_per_step.masked_fill(seg_mask, 0.0)
        if Debug:
            print(f"[DEBUG] mloss_per_step after masking:\n{mloss_per_step}")
            pdb.set_trace()
        # 平均
        mloss = torch.sum(mloss_per_step, dim=1) / num_segments
        mloss = torch.mean(mloss)

        # Total Loss
        trajectory_loss = self.w_obs * oloss + self.w_goal * gloss + self.w_motion * mloss

        # ============================================================
        # 5. Fear Loss (只看有效区域)
        # ============================================================
        # 计算累积距离
        goal_dists = torch.cumsum(wp_ds, dim=1) # [B, N-1]
        
        # 将无效区域的距离设为无穷大，确保不会被选入
        goal_dists = goal_dists.masked_fill(seg_mask, float('inf'))
        
        # 堆叠匹配 _compute_oloss_batch 的输出结构 [3B, N] -> 这里 oloss 是 N 个点
        # 注意: oloss_per_point 是 [B, N], 但 oloss_M 需要与 expanded points 对应
        # 这里我们需要复用 oloss_per_point 的逻辑，但要注意它已经是 [B, N] 了
        # 为了计算 fear，我们需要知道到底哪个点撞了。
        # 简单起见，我们直接复用上面的 oloss_per_point (它是 Center 采样的结果，实际上应该用 Left/Right/Center 的 max)
        
        # 重新调用一次 _compute_oloss_batch 拿到 [3B, N] 的原始数据更准确，
        # 或者直接利用上面的 oloss_per_point 近似 (如果只关心中心碰撞)。
        # 为了准确，我们复用 _compute_oloss_batch 的逻辑内部:
        
        # --- 重新获取含膨胀信息的 Cost ---
        raw_cost_expanded = self._compute_oloss_batch(waypoints, batch_maps) # [3B, N]
        
        # 将 mask 扩展到 3B
        dense_mask_expanded = torch.cat([dense_mask, dense_mask, dense_mask], dim=0) # [3B, N]
        
        # 1. 屏蔽无效点
        raw_cost_expanded = raw_cost_expanded.masked_fill(dense_mask_expanded, 0.0)
        if Debug:
            print(f"[DEBUG] raw_cost_expanded after masking:\n{raw_cost_expanded}")
            pdb.set_trace()
        
        # 2. 屏蔽超过预瞄距离(ahead_dist)的点
        # 需要把 goal_dists [B, N-1] 补齐到 [B, N] (第一点距离为0)
        zeros = torch.zeros((batch_size, 1), device=device)
        dists_full = torch.cat([zeros, goal_dists], dim=1) # [B, N]
        dists_expanded = torch.cat([dists_full, dists_full, dists_full], dim=0) # [3B, N]
        
        raw_cost_expanded[dists_expanded > ahead_dist] = 0.0
        if Debug:
            print(f"[DEBUG] raw_cost_expanded after ahead_dist masking:\n{raw_cost_expanded}")
            pdb.set_trace()
        
        # 3. 取最大值 (判断是否碰撞) [3B, N] -> [B, 3] -> [B, 1]
        # view 成 [3, B, N] 然后 max dim=2 (along path)
        max_cost_along_path, _ = torch.max(raw_cost_expanded.view(3, batch_size, -1), dim=2) # [3, B]
        if Debug:
            print(f"[DEBUG] max_cost_along_path shape: {max_cost_along_path.shape}")
            print(f"[DEBUG] max_cost_along_path:\n{max_cost_along_path}")
            pdb.set_trace()
        # 只要 Left/Center/Right 任意一个撞了就算撞
        is_collision = torch.any(max_cost_along_path > self.obstalce_thread, dim=0).float().unsqueeze(1) # [B, 1]
        if Debug:
            print(f"[DEBUG] is_collision shape: {is_collision.shape}")
            print(f"[DEBUG] is_collision:\n{is_collision}")
            pdb.set_trace()
        fear_loss = torch.nn.BCELoss()(fear, is_collision)

        return trajectory_loss + 0.1 * fear_loss, trajectory_loss, fear_loss
    def _compute_oloss_batch(self, waypoints, batch_maps):
        B, N, _ = waypoints.shape
        H, W = batch_maps.shape[2], batch_maps.shape[3]

        # 1. 计算切线和法线
        tangent = waypoints[:, 1:] - waypoints[:, :-1]
        
        tangent = torch.cat([tangent, tangent[:, -1:]], dim=1)
        tangent = tangent / (torch.norm(tangent, dim=2, keepdim=True) + 1e-6)
        normals = tangent[..., [1, 0]] * torch.tensor([-1, 1], device=waypoints.device)

        # 2. 膨胀 (Center, Left, Right)
        points_left = waypoints + normals * self.robot_width / 2
        points_right = waypoints - normals * self.robot_width / 2
        
        # 堆叠: [3*B, N, 2] -> 顺序: Right, Center, Left (需与 fear loss 逻辑对应)
        all_points = torch.cat([points_right, waypoints, points_left], dim=0) 
        
        if Debug:
            print(f"[DEBUG] all_points shape: {all_points.shape}")
            print(f"[DEBUG] points_left shape: {points_left.shape}")
            print(f"[DEBUG] points_right shape: {points_right.shape}")
            print(f"[DEBUG] waypoints (center) shape: {waypoints.shape}")
            pdb.set_trace()
            # 可视化膨胀点在地图上的位置
            for vis_idx in range(min(B, 2)):  # 只可视化前2个样本，避免过多弹窗
                # 获取当前样本的 cost map
                cost_map_vis = batch_maps[vis_idx, 0].cpu().numpy()
                
                # 提取当前样本的三组点 (Right, Center, Left)
                pts_right = points_right[vis_idx].detach().cpu().numpy()  # [N, 2]
                pts_center = waypoints[vis_idx].detach().cpu().numpy()     # [N, 2]
                pts_left = points_left[vis_idx].detach().cpu().numpy()     # [N, 2]
                
                # 物理坐标 -> 像素坐标转换
                res = 0.1
                center_row = H / 2.0
                center_col = W / 2.0
                def meters_to_pixels(pts):
                    px = pts[:, 0]  # x (meters)
                    py = pts[:, 1]  # y (meters)
                    row_idx = center_row - px / res
                    col_idx = center_col - py / res
                    return row_idx, col_idx
                r_row, r_col = meters_to_pixels(pts_right)
                c_row, c_col = meters_to_pixels(pts_center)
                l_row, l_col = meters_to_pixels(pts_left)
                # 创建可视化
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(cost_map_vis, cmap='jet', origin='upper', extent=[0, W, H, 0])
                # 绘制三组点
                ax.plot(r_col, r_row, 'b.-', markersize=3, linewidth=1, label='Right', alpha=0.7)
                ax.plot(c_col, c_row, 'g.-', markersize=3, linewidth=1, label='Center', alpha=0.7)
                ax.plot(l_col, l_row, 'r.-', markersize=3, linewidth=1, label='Left', alpha=0.7)
                # 标注起点
                ax.plot(center_col, center_row, 'mo', markersize=12, label='Start', 
                        markeredgecolor='white', markeredgewidth=1.5)
                ax.set_xlabel('Column (pixels)', fontsize=12)
                ax.set_ylabel('Row (pixels)', fontsize=12)
                ax.set_title(f'Expanded Points - Sample {vis_idx}', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                plt.show(block=True)
                plt.close()
            
        # 3. 归一化采样坐标
        grid = self.Pos2IndNormal_Batch(all_points, H, W) # [3*B, N, 2]
        
        # 4. 堆叠 Map: [3*B, 1, H, W]
        all_maps = torch.cat([batch_maps, batch_maps, batch_maps], dim=0)
        
        # 5. Grid Sample
        sampled_cost = torch.nn.functional.grid_sample(
            all_maps, 
            grid.unsqueeze(2), 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        ).squeeze(3).squeeze(1)  # [3B, N]
        
        return sampled_cost


# ==========================================
# 3. 地图处理器 (CPU端: 0/1 -> Smooth Cost)
# ==========================================
class MapProcessor:
    def __init__(self, res=0.1, sigma=2.0):
        self.res = res
        self.sigma = sigma
    
    def process_batch(self, raw_maps_numpy):
        """
        Raw: [B, H, W], 1=Free, 0=Obs
        Output: [B, 1, H, W], 0~1 (1=Obs), Smooth Cost
        """
        batch_size, h, w = raw_maps_numpy.shape
        out_maps = []
        
        for i in range(batch_size):
            grid = raw_maps_numpy[i]
            
            # 1. 反转 (1=Obs, 0=Free)
            binary_obs = (grid == 0).astype(np.float32)
            binary_free = (grid == 1).astype(np.float32)
            
            # 2. 距离变换
            dist = ndimage.distance_transform_edt(binary_free)
            dist_metric = dist * self.res
            
            # 3. 指数衰减 Cost = exp(-2.0 * dist)
            cost = np.exp(-3.0 * dist_metric)
            cost[binary_obs == 1] = 1.0  # 强制障碍物
            
            # 4. 平滑
            cost = gaussian_filter(cost, sigma=self.sigma)
            cost = np.clip(cost, 0.0, 1.0)
            
            out_maps.append(cost)
            
        return np.array(out_maps)[:, np.newaxis, :, :] # [B, 1, H, W]


# ==========================================
# 4. 主训练流程
# ==========================================

def get_curriculum_probs(epoch, max_warmup_epoch=100):
    """
    根据 epoch 计算当前课程难度概率 (线性过渡)
    
    Args:
        epoch: 当前轮数
        max_warmup_epoch: 达到最终难度的轮数 (比如第100轮达到最难)
    """
    # 计算进度 alpha: 0.0 (开始) -> 1.0 (结束)
    # min确保超过 warmup 轮数后，概率保持在最终状态，不再变化
    alpha = min(epoch / max_warmup_epoch, 1.0)
    
    # --- A. 任务阶段概率 (Step1 -> Step3) ---
    # 初始状态 (Epoch 0): 100% 简单
    start_steps = np.array([1.0, 0.0, 0.0])
    # 最终状态 (Epoch >= 100): 主要是困难，保留少量简单
    end_steps = np.array([0.1, 0.3, 0.6])
    
    # 线性插值公式: current = start + (end - start) * alpha
    curr_steps = start_steps + (end_steps - start_steps) * alpha
    
    # --- B. 距离概率 (Near -> Far) ---
    # 初始: 90% 近距离
    start_dist = np.array([0.9, 0.1])
    # 最终: 80% 远距离
    end_dist = np.array([0.2, 0.8])
    
    curr_dist = start_dist + (end_dist - start_dist) * alpha
    
    return {
        'step1': curr_steps[0], 
        'step2': curr_steps[1], 
        'step3': curr_steps[2],
        'p_near': curr_dist[0], 
        'p_far':  curr_dist[1]
    }

def align_map_size(map_tensor, target_size=(80, 80)):
    """
    检查并强制缩放地图到指定尺寸 (80, 80)
    map_tensor: [B, C, H, W]
    """
    if map_tensor.shape[-1] != target_size[1] or map_tensor.shape[-2] != target_size[0]:
        # print(f"[DEBUG] Resizing map from {map_tensor.shape[-2:]} to {target_size}")
        map_tensor = torch.nn.functional.interpolate(
            map_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
    return map_tensor

def train_pipeline():
    # 1. 初始化配置
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if os.path.exists(cfg.log_dir + '/tb'):
        shutil.rmtree(cfg.log_dir + '/tb')
    writer = SummaryWriter(log_dir=cfg.log_dir + '/tb')
    
    # --- A. 数据集准备 (使用 Config 参数) ---
    print("[INFO] Loading Dataset...")
    
    # 训练集
    train_dataset = CollectData(
        root_dir=cfg.data_root,
        mode='train',
        split_ratio=0.9,  # 保留您的设定
        safe_dist_threshold=2.0,
        config=cfg.train_aug_config  # 传入包含 ghost_prob=0.8 的配置
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # 验证集
    val_dataset = CollectData(
        root_dir=cfg.data_root,
        mode='val',
        split_ratio=0.9,
        safe_dist_threshold=2.0,
        config=cfg.val_aug_config
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- B. 模型初始化 ---
    model = AutoEncoderGrid(
        encoder_channel=64, 
        max_dist=cfg.max_dist, 
        step_size=cfg.step_size  # 保留您的设定 0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # --- C. 工具类初始化 ---
    # 使用 BatchTrajCost 替代原始 TrajCost
    batch_traj_cost = BatchTrajCost(
        gpu_id=0 if torch.cuda.is_available() else "cpu",
        w_obs=cfg.w_obs,
        w_motion=cfg.w_motion,
        w_goal=cfg.w_goal,
        obstalce_thread=cfg.fear_ahead_dist
    )
    map_proc = MapProcessor()

    # --- D. 训练循环 ---
    print("[INFO] Start Training Loop...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        start_time = time.time()
        model.train()
        train_loss_total = 0.0
        
        # 1. 课程学习更新
        probs = get_curriculum_probs(epoch, max_warmup_epoch=int(0.8 * cfg.epochs))
        
        # 打印当前概率 (可选，方便观察)
        if epoch % 2 == 0:
            print(f"[Curriculum] Ep {epoch}: "
                  f"Step1={probs['step1']:.2f}, Step3={probs['step3']:.2f}, "
                  f"Far={probs['p_far']:.2f}")

        # 更新数据集策略
        train_dataset.update_curriculum(epoch, probs)
        val_dataset.update_curriculum(epoch, probs)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (map_tensor, goal_tensor) in enumerate(pbar):
            # map_tensor: [B, 1, 80, 80] (1=Free)
            global Debug
            if Debug:
                # 获取当前 Batch 有多少个样本 (通常是 2)
                current_batch_size = map_tensor.shape[0]
                
                # --- 修正点 1: 遍历当前 Batch 的样本数，而不是 batch_idx ---
                for sample_idx in range(current_batch_size):
                    
                    # 1. 转换图像
                    # map_tensor: [B, 1, H, W] -> 取出 [H, W]
                    # 必须先 detach() 才能转 numpy
                    raw_map = map_tensor[sample_idx, 0].detach().cpu().numpy()
                    
                    # --- 修正点 2: 打印极值，确认数据是否正常 ---
                    # 如果 Max 是 1.0，Min 是 0.0，那显示出来就是 白(1) 和 黑(0)
                    # 为了让人眼看清楚，我们可以反转一下：障碍物(0)变白(255)，空地(1)变黑(0)
                    # 或者保持原样：空地(1)->255(白), 障碍(0)->0(黑)
                    
                    print(f"[Debug Vis] Batch {batch_idx} Sample {sample_idx}: Map Range [{raw_map.min():.2f}, {raw_map.max():.2f}]")
                    
                    map_img = (raw_map * 255).astype(np.uint8)
                    
                    # 如果你想让障碍物(0)显示为白色，背景(1)显示为黑色，取消下面这行的注释：
                    # map_img = 255 - map_img 
                    
                    map_img_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
                                        
                    # 2. 获取目标点
                    # 注意：goal_tensor 已经在之前的代码中被转换了吗？
                    # 如果 goal_tensor 还是像素坐标，直接用；如果是 meters，需要转换回像素
                    # 假设这里已经是像素坐标 (col, row)
                    goal_col = int(goal_tensor[sample_idx, 0].item())
                    goal_row = int(goal_tensor[sample_idx, 1].item())
                    
                    # 3. 画图
                    # 画终点 (红色实心圆)
                    cv2.circle(map_img_color, (goal_col, goal_row), 3, (0, 0, 255), -1)
                    
                    # 画起点 (绿色实心圆) - 假设地图中心是起点
                    center_x, center_y = 40, 40 # 80x80的一半
                    cv2.circle(map_img_color, (center_x, center_y), 2, (0, 255, 0), -1)
                    
                    # 4. 显示
                    win_name = f'Debug View' # 使用固定窗口名，防止弹出几百个窗口
                    cv2.imshow(win_name, map_img_color)
                    
                    print(f"    Goal Pixel: ({goal_col}, {goal_row})")
                    print("    Press Any Key to continue...")
                    
                    # --- 修正点 3: 这里会暂停，按任意键继续 ---
                    key = cv2.waitKey(0) 
                    if key == 27: # 按 ESC 键退出 Debug 模式
                        Debug = False
                        cv2.destroyAllWindows()
                        break
                
            # --- 核心：数据预处理 (Batch Map) ---
            raw_np = map_tensor.squeeze(1).numpy()
            # 转换为 Cost Map (1=Obs, Smooth)
            smooth_cost_np = map_proc.process_batch(raw_np)
            smooth_cost_tensor = torch.from_numpy(smooth_cost_np).float().to(device)
            
            # 准备网络输入 (Cost Map)
            net_input = align_map_size(smooth_cost_tensor, target_size=(80, 80)).to(device)
            
            # --- 像素坐标 -> 物理坐标转换 ---
            # goal_tensor 当前格式: [B, 2] 或 [B, 3]，其中前两维是像素坐标 (row_idx, col_idx)
            # 需要转换为物理坐标 (g_x, g_y) in meters

            # 地图参数
            map_res = 0.1  # 分辨率 (m/pixel)
            src_h, src_w = map_tensor.shape[-2], map_tensor.shape[-1]
            target_h, target_w = 80, 80
            scale_r = target_h / src_h
            scale_c = target_w / src_w
            center_row = target_h / 2.0
            center_col = target_w / 2.0

            # goal_tensor = (col, row)，先同步缩放到 80x80
            goal_col = goal_tensor[:, 0] * scale_c
            goal_row = goal_tensor[:, 1] * scale_r
            # print(f"[DEBUG] Goal Pixel (Batch): {goal_tensor.cpu().numpy()}")
            # x 对应 row，y 对应 col
            goal_meters = torch.zeros((goal_tensor.shape[0], 2), dtype=torch.float32)
            goal_meters[:, 0] = (center_row - goal_row) * map_res  # x (meters)
            goal_meters[:, 1] = (center_col - goal_col) * map_res  # y (meters)
            
            if cfg.debug:
                for debug_idx in range(goal_meters.shape[0]):
                    print(f"[DEBUG] Sample {debug_idx}: Goal Pixel=({goal_col[debug_idx]:.1f}, {goal_row[debug_idx]:.1f}), Goal Meters=({goal_meters[debug_idx, 0]:.2f}, {goal_meters[debug_idx, 1]:.2f})")
                                        
                    # 获取当前样本的 cost map
                    cost_map = net_input[debug_idx, 0].cpu().numpy()
                    
                    # 创建可视化图像
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(cost_map, cmap='jet', origin='upper', extent=[0, target_w, target_h, 0])
                    
                    # 标注终点
                    ax.plot(goal_col[debug_idx].item(), goal_row[debug_idx].item(), 'r*', markersize=20, label='Goal')
                    
                    # 标注起点 (中心)
                    ax.plot(center_col, center_row, 'go', markersize=10, label='Start')
                    
                    # 在图上添加文本注释
                    text_str = (
                        f"Goal Index: ({goal_col[debug_idx].item():.1f}, {goal_row[debug_idx].item():.1f})\n"
                        f"Goal Meters: ({goal_meters[debug_idx, 0].item():.2f}, {goal_meters[debug_idx, 1].item():.2f})"
                    )
                    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel('Column (pixels)')
                    ax.set_ylabel('Row (pixels)')
                    ax.set_title(f'Cost Map with Goal - Epoch {epoch}, Batch {batch_idx}, Sample {debug_idx}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show(block=True)
                    plt.close()
                
            
            # 补齐维度 [B, 3] 如果需要 theta，否则保持 [B, 2]
            # 如果你的模型需要 3维 goal (x, y, theta)，请确保维度匹配
            if goal_meters.shape[1] == 2 and net_input.ndim > 0:
                # 这种情况下通常不需要额外补 0，除非 TrajGenerator 报错
                pass

            goal_meters = goal_meters.to(device)
            
            goal_norm = goal_meters.clone()
            # print(f"[DEBUG] Goal Meters (Batch): {goal_meters.cpu().numpy()}")
            goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist 
            # print(f"[DEBUG] Goal Norm (Batch): {goal_norm.cpu().numpy()}")
            # goal_tensor = goal_tensor.to(device)
            
            optimizer.zero_grad()
            
            # --- 前向传播 ---
            preds, pred_fear, mask = model(net_input, goal_norm)
            
            # --- 轨迹生成 ---
            waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
                preds, step=cfg.sub_step_size, mask=mask
            )

            if cfg.debug:
                # 可视化预测关键点和密集轨迹
                print(f"[DEBUG] Visualizing Predictions - Epoch {epoch}, Batch {batch_idx}")
                print(f"    Preds (meters):\n{preds.shape}")
                print(f"    Waypoints (meters):\n{waypoints.shape}")
                print(f"Waypoints (meters):\n{waypoints.detach().cpu().numpy()}")
                for vis_idx in range(preds.shape[0]):
                    # 获取当前样本的 cost map
                    cost_map_vis = net_input[vis_idx, 0].cpu().numpy()
                    
                    # 创建可视化图像
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(cost_map_vis, cmap='jet', origin='upper', extent=[0, target_w, target_h, 0])
                    
                    # 1. 转换预测关键点坐标 (preds) 到像素坐标
                    pred_points = preds[vis_idx].detach().cpu().numpy()  # [N_keypoints, 2]
                    pred_x = pred_points[:, 0]  # 物理坐标 x (meters)
                    pred_y = pred_points[:, 1]  # 物理坐标 y (meters)
                    
                    # 物理坐标 -> 像素坐标
                    pred_row = center_row - pred_x / map_res
                    pred_col = center_col - pred_y / map_res
                    
                    # 过滤掉 mask=True 的点 (如果有 mask)
                    if mask is not None:
                        valid_mask = ~mask[vis_idx].cpu().numpy()  # [N_keypoints]
                        pred_row_valid = pred_row[valid_mask]
                        pred_col_valid = pred_col[valid_mask]
                    else:
                        pred_row_valid = pred_row
                        pred_col_valid = pred_col
                    
                    # 绘制预测关键点 (蓝色方块)
                    ax.plot(pred_col_valid, pred_row_valid, 'bs', markersize=8, 
                            label='Pred Keypoints', markerfacecolor='blue', markeredgecolor='white')
                    
                    # 2. 转换密集轨迹点坐标 (waypoints) 到像素坐标
                    waypoints_sample = waypoints[vis_idx].detach().cpu().numpy()  # [Max_Dense_Len, 2]
                    wp_x = waypoints_sample[:, 0]
                    wp_y = waypoints_sample[:, 1]
                    
                    wp_row = center_row - wp_x / map_res
                    wp_col = center_col - wp_y / map_res
                    
                    # 过滤掉 padding 的密集点
                    if mask is not None:
                        valid_key_cnt = (~mask[vis_idx]).sum().float().item()
                        valid_dense_len = int((valid_key_cnt / cfg.sub_step_size) + 1)
                        valid_dense_len = min(max(valid_dense_len, 2), waypoints_sample.shape[0])
                        wp_row_valid = wp_row[:valid_dense_len]
                        wp_col_valid = wp_col[:valid_dense_len]
                    else:
                        wp_row_valid = wp_row
                        wp_col_valid = wp_col
                    
                    # 绘制密集轨迹 (白色线条 + 小圆点)
                    ax.plot(wp_col_valid, wp_row_valid, 'w.-', linewidth=2, markersize=3, 
                            label='Waypoints (Dense)', alpha=0.8)
                    
                    # 3. 标注终点和起点
                    ax.plot(goal_col[vis_idx].item(), goal_row[vis_idx].item(), 'r*', 
                            markersize=20, label='Goal', markeredgecolor='yellow', markeredgewidth=1.5)
                    ax.plot(center_col, center_row, 'go', markersize=12, 
                            label='Start', markeredgecolor='white', markeredgewidth=1.5)
                    
                    # 4. 添加文本信息
                    text_str = (
                        f"Goal: ({goal_meters[vis_idx, 0].item():.2f}, {goal_meters[vis_idx, 1].item():.2f}) m\n"
                        f"Pred Keypoints: {len(pred_row_valid)}\n"
                        f"Dense Waypoints: {len(wp_row_valid)}"
                    )
                    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    ax.set_xlabel('Column (pixels)', fontsize=12)
                    ax.set_ylabel('Row (pixels)', fontsize=12)
                    ax.set_title(f'Trajectory Prediction - Epoch {epoch}, Batch {batch_idx}, Sample {vis_idx}', 
                                 fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    plt.show(block=True)
                    plt.close()
                    
            # --- 诊断日志：有效长度与终点距离 ---
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    if mask is not None:
                        valid_key_cnt = (~mask).sum(dim=1).float()
                        valid_dense_len = (valid_key_cnt / cfg.sub_step_size).long() + 1
                        valid_dense_len = torch.clamp(valid_dense_len, min=2, max=waypoints.shape[1])
                        last_idx = (valid_dense_len - 1).clamp(min=0, max=waypoints.shape[1] - 1)
                        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)
                        end_pts = torch.gather(waypoints, 1, gather_idx).squeeze(1)
                    else:
                        valid_key_cnt = torch.full(
                            (waypoints.shape[0],),
                            float(waypoints.shape[1]),
                            device=waypoints.device,
                        )
                        valid_dense_len = torch.full(
                            (waypoints.shape[0],),
                            waypoints.shape[1],
                            device=waypoints.device,
                            dtype=torch.long,
                        )
                        end_pts = waypoints[:, -1, :]

                    end_dist = torch.norm(goal_meters - end_pts, dim=1)
                    print(
                        f"[Diag] ep{epoch} b{batch_idx} "
                        f"key_cnt={valid_key_cnt.mean().item():.2f} "
                        f"dense_len={valid_dense_len.float().mean().item():.1f} "
                        f"end_dist={end_dist.mean().item():.2f}"
                    )
                    writer.add_scalar(
                        "Diag/EndDist",
                        end_dist.mean().item(),
                        epoch * len(train_loader) + batch_idx,
                    )
            
            # --- 计算 Loss (Batch) ---
            loss, l_traj, l_fear = batch_traj_cost.CostofTraj_Batch(
                waypoints=waypoints,
                goal=goal_meters,
                fear=pred_fear,
                log_step=epoch,
                ahead_dist=cfg.fear_ahead_dist,
                batch_maps=net_input,  # 传入与网络输入一致的 Batch Map
                sub_step_size=cfg.sub_step_size,
                mask=mask
            )
            
            loss.backward()
            
            writer.add_scalar('Loss/Train_Total', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Train_Traj', l_traj.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Train_Fear', l_fear.item(), epoch * len(train_loader) + batch_idx)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'traj': l_traj.item()})
            
            # 可视化 (每 50 batch 保存一张)
            if batch_idx % 50 == 0:
                visualize_debug(
                    cost_maps=net_input.detach().cpu().numpy(),
                    waypoints=waypoints,
                    preds=preds,          # <--- 新增
                    mask=mask,            # <--- 新增
                    goals=goal_meters,
                    epoch=epoch,
                    idx=batch_idx,
                    save_dir=f"{cfg.log_dir}/train",
                    res=cfg.resolution,
                    sub_step_size=cfg.sub_step_size  # <--- 新增，确保截断准确
                )

        # --- E. 验证循环 ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_idx, (map_tensor, goal_tensor) in enumerate(val_loader):
                # 同样的预处理流程
                raw_np = map_tensor.squeeze(1).numpy()
                smooth_cost_np = map_proc.process_batch(raw_np)
                smooth_cost_tensor = torch.from_numpy(smooth_cost_np).float().to(device)
                net_input = align_map_size(smooth_cost_tensor, target_size=(80, 80)).to(device)
                
                # 2. 【关键修改】坐标转换 (Pixels -> Meters)
                # 必须与训练集保持一致！
                map_res = 0.1
                src_h, src_w = map_tensor.shape[-2], map_tensor.shape[-1]
                target_h, target_w = 80, 80
                scale_r = target_h / src_h
                scale_c = target_w / src_w
                center_row = target_h / 2.0
                center_col = target_w / 2.0

                goal_col = goal_tensor[:, 0] * scale_c
                goal_row = goal_tensor[:, 1] * scale_r

                goal_meters = torch.zeros((goal_tensor.shape[0], 2), dtype=torch.float32)
                goal_meters[:, 0] = (center_row - goal_row) * map_res
                goal_meters[:, 1] = (center_col - goal_col) * map_res
                goal_meters = goal_meters.to(device)

                # 3. 【关键修改】归一化 (Meters -> Ratio)
                goal_norm = goal_meters.clone()
                goal_norm[:, :2] = goal_meters[:, :2] / cfg.max_dist
                
                preds, pred_fear, mask = model(net_input, goal_norm)
                
                waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
                    preds, step=cfg.sub_step_size, mask=mask
                )
                
                loss, _, _ = batch_traj_cost.CostofTraj_Batch(
                    waypoints=waypoints,
                    goal=goal_meters,
                    fear=pred_fear,
                    log_step=0,
                    ahead_dist=cfg.fear_ahead_dist,
                    batch_maps=net_input,
                    sub_step_size=cfg.sub_step_size,
                    mask=mask
                )
                val_loss_total += loss.item()
                
                if batch_idx % 20 == 0:
                    visualize_debug(
                        cost_maps=net_input.detach().cpu().numpy(),
                        waypoints=waypoints,
                        preds=preds,          # <--- 新增
                        mask=mask,            # <--- 新增
                        goals=goal_meters,
                        epoch=epoch,
                        idx=0,
                        save_dir=f"{cfg.log_dir}/val",
                        res=cfg.resolution,
                        sub_step_size=cfg.sub_step_size # <--- 新增
                    )
        
        # --- F. 统计与保存 ---
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{cfg.save_dir}/best_model.pt")
            print(">>> Best Model Saved!")
            
        if (epoch + 1) % 50 == 0:
             torch.save(model.state_dict(), f"{cfg.save_dir}/epoch_{epoch+1}.pt")

def visualize_debug(cost_maps, waypoints, preds, mask, goals, epoch, idx, save_dir, res=0.1, sub_step_size=0.2):
    """
    Args:
        cost_maps: Numpy [B, 1, H, W] or [B, H, W] -> Cost Map
        waypoints: Tensor [B, Dense_Len, 2] -> 密集的插值轨迹
        preds:     Tensor [B, Key_Len, 2]   -> 模型预测的稀疏关键点
        mask:      Tensor [B, Key_Len]      -> 关键点掩码 (True=Invalid)
        goals:     Tensor [B, 2]            -> 目标点 (物理坐标 meters)
        res:       float                    -> 地图分辨率
        sub_step_size: float                -> 用于计算密集轨迹有效长度 (必须与 Config 一致)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # === 1. 数据准备 (取 Batch 中的第 0 个样本) ===
    # 确保输入转为 numpy
    cmap = cost_maps[0, 0] if cost_maps.ndim == 4 else cost_maps[0] # [H, W]
    traj_dense = waypoints[0].detach().cpu().numpy() # [Dense_Len, 2]
    pred_keys = preds[0].detach().cpu().numpy()      # [Key_Len, 2]
    goal = goals[0].detach().cpu().numpy()           # [2]
    cur_mask = mask[0].detach().cpu().numpy() if mask is not None else None # [Key_Len]

    # 地图中心参数
    H, W = cmap.shape
    center_row = H / 2.0
    center_col = W / 2.0

    # === 2. 坐标转换辅助函数 ===
    def to_pixel(points_meter):
        """ 物理坐标 (x=前, y=左) -> 像素坐标 (row, col) """
        if points_meter.ndim == 1: points_meter = points_meter[None, :]
        pm_x = points_meter[:, 0]
        pm_y = points_meter[:, 1]
        
        # 转换公式: row = center - x/res, col = center - y/res
        rows = center_row - pm_x / res
        cols = center_col - pm_y / res
        return rows, cols

    # === 3. 处理有效性 (Masking) ===
    
    # A. 处理预测关键点 (Preds)
    if cur_mask is not None:
        # mask 为 True 表示无效 (Padding)，取反获得有效点
        valid_key_idx = ~cur_mask.astype(bool)
        pred_keys_valid = pred_keys[valid_key_idx]
        
        # B. 计算密集轨迹的有效长度
        # 逻辑必须与 CostofTraj_Batch 中的 valid_dense_len 计算一致
        n_valid_keys = np.sum(valid_key_idx)
        valid_dense_len = int(n_valid_keys / sub_step_size) + 1
        
        # 钳制长度，防止越界
        valid_dense_len = min(max(valid_dense_len, 2), len(traj_dense))
        traj_dense_valid = traj_dense[:valid_dense_len]
    else:
        # 如果没有 mask，假定全部有效
        pred_keys_valid = pred_keys
        traj_dense_valid = traj_dense

    # === 4. 绘图 ===
    plt.figure(figsize=(6, 6)) # 稍微大一点
    
    # 画地图
    plt.imshow(cmap, cmap='jet', origin='upper', extent=[0, W, H, 0]) # 注意 extent 确保坐标对齐
    
    # 转换坐标
    traj_r, traj_c = to_pixel(traj_dense_valid)
    pred_r, pred_c = to_pixel(pred_keys_valid)
    goal_r, goal_c = to_pixel(goal)
    
    # 1. 画密集轨迹 (白色细线)
    plt.plot(traj_c, traj_r, 'w-', linewidth=2, alpha=0.8, label='Dense Traj')
    
    # 2. 画预测关键点 (蓝色方块 - 明显一点)
    plt.plot(pred_c, pred_r, 'bs', markersize=6, markeredgecolor='white', label='Pred Keys')
    
    # 3. 画目标点 (红色五角星)
    plt.plot(goal_c, goal_r, 'r*', markersize=12, markeredgecolor='black', label='Goal')
    
    # 4. 画起点 (绿色圆点)
    plt.plot(center_col, center_row, 'go', markersize=8, markeredgecolor='white', label='Start')

    plt.legend(loc='upper right', fontsize='small')
    plt.title(f"Ep {epoch} Batch {idx} | ValidKeys: {len(pred_keys_valid)}")
    plt.tight_layout()
    
    # 保存
    plt.savefig(f"{save_dir}/debug_{epoch}_{idx}.png", dpi=100)
    plt.close()

if __name__ == "__main__":
    train_pipeline()