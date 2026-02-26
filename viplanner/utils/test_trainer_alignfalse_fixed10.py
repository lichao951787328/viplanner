import os
import sys
import time
import cv2
import torch
import torch.optim as optim
import numpy as np
# 在训练时打开，调试时关闭，显示图像
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
import shutil
import pdb
import heapq
Debug = False  # 是否开启调试模式 (打印更多信息，保存中间结果等)
USING_transformer = True  # 是否使用 Transformer 版本的 PlannerNet
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    # print("[INFO] OpenCV threading disabled explicitly.")
except Exception as e:
    print(f"[WARN] Failed to disable OpenCV threading: {e}")

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
    from trainer_dataset_fixed10 import CollectData

    # B. 导入父目录下的模块 (现在可以直接从文件夹名开始了)
    # 对应: .../viplanner/plannernet/autoencoder_myself_cubic.py
    from plannernet.autoencoder_myself_cubic_dj_fixed10 import AutoEncoderGrid
    from plannernet.planner_transformer_fixed10 import TransformerPlanner
    
    # 对应: .../viplanner/traj_cost_opt/traj_cost_myself_cubic.py
    from traj_cost_opt.traj_cost_myself_cubic_alignfalse import TrajCost


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
        # 路径设置
        # 获取当前文件所在目录的上上层目录
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # .../viplanner/utils
        project_root = os.path.dirname(os.path.dirname(current_file_dir))  # .../viplanner的上层
        rotated_out_dir = os.path.join(project_root, "rotated_out", "carla")
        
        self.data_root = os.path.join(rotated_out_dir, "samples")

        # A/B/C 实验开关（单次运行建议只选一个）
        # A: 现有 loss + 混合障碍风险
        # B: A + 单步进展约束 Lprog
        # C: B + 长度比约束 Llen
        self.exp_variant = "A"

        self.save_dir = os.path.join(rotated_out_dir, f"checkpoints_v2_alignfalse_fixed10_{self.exp_variant}")
        self.log_dir = os.path.join(rotated_out_dir, f"logs_v2_alignfalse_fixed10_{self.exp_variant}")
        # self.data_root = "/home/eai/VLN/viplanner/rotated_out/carla/samples"  # 数据路径
        # self.save_dir = "/home/eai/VLN/viplanner/rotated_out/carla/checkpoints_v2"      # 模型保存路径
        # self.log_dir = "/home/eai/VLN/viplanner/rotated_out/carla/logs_v2"              # 可视化结果保存路径
        
        # 训练超参数
        self.epochs = 10000
        self.batch_size = 128    # 根据显存大小调整，取2先试试
        self.lr = 5e-4
        self.num_workers = 4
        
        # 地图相关
        self.map_size = 80
        self.max_dist = 4.0    # 预测最远距离
        self.step_size = 1    # 轨迹点间隔
        self.sub_step_size = 0.1  # 子轨迹点间隔, 用于计算更精细的成本
        self.resolution = 0.1   # 地图分辨率 (0.1m/pixel)
        
        # 损失权重 (传给 TrajCost)
        self.w_obs = 5       # 障碍物避让权重
        self.w_goal = 5       # 到达目标权重
        self.w_motion = 14     # 平滑/动态权重
        self.w_guide = 2      # 引导点权重 (新添加)
        self.fear_ahead_dist = 2.0

        # --- A/B/C 损失参数 ---
        # A: 混合障碍风险
        self.obs_topk_ratio = 0.15
        self.obs_safe_threshold = 0.8
        self.obs_mean_alpha = 0.3
        self.obs_topk_beta = 0.4
        self.obs_max_gamma = 0.4
        self.obs_barrier_delta = 0.3

        # B: 进展约束
        self.w_prog = 1.0
        self.prog_margin = 0.02

        # C: 长度比约束
        self.w_len = 0.5
        self.len_ratio_target = 1.25
        
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
        self.exp_variant = "A"
        self.obs_topk_ratio = 0.15
        self.obs_safe_threshold = 0.8
        self.obs_mean_alpha = 0.3
        self.obs_topk_beta = 0.4
        self.obs_max_gamma = 0.4
        self.obs_barrier_delta = 0.1
        self.w_prog = 1.0
        self.prog_margin = 0.02
        self.w_len = 0.3
        self.len_ratio_target = 1.25
        
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
        # align_corners=False 对应“像素中心”映射：
        # x_norm = 2 * ((x + 0.5) / W) - 1
        # y_norm = 2 * ((y + 0.5) / H) - 1
        norm_col = 2.0 * ((col_idx + 0.5) / W) - 1.0
        norm_row = 2.0 * ((row_idx + 0.5) / H) - 1.0
        
        # Stack 为 (x, y) 即 (col, row)
        grid = torch.stack((norm_col, norm_row), dim=-1)
        return torch.clamp(grid, -1.0, 1.0)

    def CostofTraj_Batch(self, waypoints, goal, fear, log_step, ahead_dist, batch_maps, sub_step_size, mask=None, distance=None, path_pixel_raw_tensor=None, path_dist_tensor=None):
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
            dummy_loss = waypoints.sum() * 0.0 
            return dummy_loss, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            # 返回一个带梯度的伪 Loss 防止程序直接 Crash，或者选择 raise Error
            # return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # ============================================================
        # 1. 计算密集轨迹的 Valid Length 和 Mask
        # ============================================================
        if mask is not None:
            # A. 计算关键点的有效数量 (Keypoints Valid Count)
            # mask True 为无效，取反求和
            valid_key_cnt = (~mask).sum(dim=1).float() # [B]
            global Debug
            if Debug :
                print(f"[DEBUG] valid_key_cnt: {valid_key_cnt}")
                # pdb.set_trace()
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
            # pdb.set_trace()

        # ============================================================
        # 2. Obstacle Loss (屏蔽 Padding 区域)
        # ============================================================
        # 获取每个点的 Cost [B, N]
        oloss_per_point = self._compute_oloss_batch(waypoints, batch_maps)
        dense_mask_3x = torch.cat([dense_mask, dense_mask, dense_mask], dim=0)
        
        if Debug:
            print(f"[DEBUG] oloss_per_point shape: {oloss_per_point.shape}")
            print(f"[DEBUG] oloss_per_point:\n{oloss_per_point}")
            # pdb.set_trace()
        
        # 将 Mask 部分的 Cost 置为 0
        oloss_per_point = oloss_per_point.masked_fill(dense_mask_3x, 0.0)
        
        if Debug:
            print(f"[DEBUG] oloss_per_point after masking:\n{oloss_per_point}")
            # pdb.set_trace()

        valid_len_3x = torch.cat([valid_dense_len, valid_dense_len, valid_dense_len], dim=0).float()
        
        # A 组：混合障碍风险（mean + topk + max + barrier）
        valid_len_3x_int = valid_len_3x.long().clamp(min=1)
        n_rows = oloss_per_point.shape[0]
        mean_list, topk_list, max_list, barrier_list = [], [], [], []

        for i in range(n_rows):
            curr_len = int(valid_len_3x_int[i].item())
            valid_vals = oloss_per_point[i, :curr_len]

            curr_mean = torch.mean(valid_vals)
            curr_max = torch.max(valid_vals)

            topk_k = max(1, int(curr_len * self.obs_topk_ratio))
            topk_vals, _ = torch.topk(valid_vals, k=topk_k, largest=True)
            curr_topk = torch.mean(topk_vals)
            # valid_vals：轨迹上每个有效采样点的障碍代价值
            # self.obs_safe_threshold：你定义的“安全阈值”（比如 0.8）
            # valid_vals - threshold：超过阈值多少
            # relu(...)：只保留超过阈值的部分；低于阈值直接记 0（不罚）
            # ** 2：超得越多，惩罚增长更快（平方惩罚）
            # mean(...)：对这条轨迹所有有效点求平均
            # 所以它的含义是：
            # 只惩罚“危险区”点，且越危险惩罚越大；安全区不惩罚。
            # 直观看：
            # 若某点 cost = 0.6，阈值 0.8 → 惩罚 0
            # 若某点 cost = 1.0，阈值 0.8 → 惩罚 (1.0−0.8) 2=0.04
            # 若某点 cost = 1.5，阈值 0.8 → 惩罚 (1.5−0.8) 2=0.49（大很多）
            curr_barrier = torch.mean(torch.relu(valid_vals - self.obs_safe_threshold) ** 2)
            mean_list.append(curr_mean)
            topk_list.append(curr_topk)
            max_list.append(curr_max)
            barrier_list.append(curr_barrier)

        oloss_mean = torch.mean(torch.stack(mean_list))
        oloss_topk = torch.mean(torch.stack(topk_list))
        oloss_max = torch.mean(torch.stack(max_list))
        oloss_barrier = torch.mean(torch.stack(barrier_list))

        oloss = (
            self.obs_mean_alpha * oloss_mean
            + self.obs_topk_beta * oloss_topk
            + self.obs_max_gamma * oloss_max
            + self.obs_barrier_delta * oloss_barrier
        )

        # ============================================================
        # 3. Goal Loss (取真正的最后一个有效点)
        # ============================================================
        # 不能直接取 waypoints[:, -1]，因为那里可能是补齐的 0
        # 我们利用 valid_dense_len 找到每个 Batch 真实的最后一个点的索引
        last_indices = (valid_dense_len - 1).clamp(min=0, max=max_dense_len - 1)  # [B]
        
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
            # pdb.set_trace()
        
        gloss_M = torch.norm(goal[:, :2] - real_endpoints, dim=1)
        gloss = torch.mean(torch.log(gloss_M + 1.0))  # 乘以2是为了让数值更大一些，您可以根据实际情况调整

        # ============================================================
        # 4. Motion Loss (修正版：自适应均布)
        # ============================================================
        # 计算相邻点距离 (实际步长) [B, N-1]
        wp_ds = torch.norm(waypoints[:, 1:] - waypoints[:, :-1], dim=2)
        
        # Mask 掉无效的段 (dense_mask[:, 1:] 对应 N-1 个段)
        seg_mask = dense_mask[:, 1:]  # [B, N-1]
        if Debug:
            print(f"[DEBUG] wp_ds shape: {wp_ds.shape}")
            print(f"[DEBUG] wp_ds:\n{wp_ds}")
            print(f"[DEBUG] seg_mask shape: {seg_mask.shape}")
            print(f"[DEBUG] seg_mask:\n{seg_mask}")
            # pdb.set_trace()
            
        # --- 核心修改开始 ---
        
        # 1. 先把无效段的距离置为 0，防止影响求和
        wp_ds_masked = wp_ds.masked_fill(seg_mask, 0.0)
        if Debug:
            print(f"[DEBUG] wp_ds_masked shape: {wp_ds_masked.shape}")
            print(f"[DEBUG] wp_ds_masked:\n{wp_ds_masked}")
            # pdb.set_trace()
        
        # 2. 计算每条轨迹“当前实际”的平均步长 (Total Length / Valid Segments)
        # 注意：这里不用 dist_to_goal (直线距离)，而是用 sum(wp_ds) (曲线距离)
        # 这样无论路径多弯，我们只要求点之间是均匀的即可
        current_path_len = torch.sum(wp_ds_masked, dim=1) # [B]
        
        # 加上 1e-6 防止除以0
        num_segments = (valid_dense_len - 1).float().clamp(min=1.0)
        adaptive_mean_step = (current_path_len / num_segments).unsqueeze(1) # [B, 1]
        if Debug:
            print(f"[DEBUG] current_path_len: {current_path_len}")
            print(f"[DEBUG] num_segments: {num_segments}")
            print(f"[DEBUG] adaptive_mean_step: {adaptive_mean_step}")
            # pdb.set_trace()
            
        # 3. 计算方差损失：(每个步长 - 当前平均步长)^2
        # 这会让所有步长趋向于一致，而不限制总长度
        mloss_per_step = (wp_ds - adaptive_mean_step) ** 2
        
        # 4. 再次 Mask (确保无效区域不计入 Loss)
        mloss_per_step = mloss_per_step.masked_fill(seg_mask, 0.0)
        if Debug:
            print(f"[DEBUG] mloss_per_step shape: {mloss_per_step.shape}")
            print(f"[DEBUG] mloss_per_step:\n{mloss_per_step}")
            # pdb.set_trace()
            
        # 5. 求平均
        mloss = torch.sum(mloss_per_step, dim=1) / num_segments
        mloss = torch.mean(mloss)
        if Debug:
            print(f"[DEBUG] mloss: {mloss}")
            # pdb.set_trace()
            
        # --- 核心修改结束 ---
        # [可选] 增加一个微小的长度惩罚 (Length Regularization)
        # 防止轨迹为了均匀而故意画蛇添足绕大圈
        dist_to_goal = torch.norm(goal[:, :2], dim=1) 
        l_reg = torch.mean(current_path_len / (dist_to_goal + 1e-6)) 
        mloss += 0.1 * l_reg
        
        # ============================================================
        # 5. Guide Loss (基于比例对齐 - 纯 Tensor 实现)
        # ============================================================
        if Debug:
            print(f"[DEBUG] path_pixel_raw_tensor: {path_pixel_raw_tensor}")
            print(f"[DEBUG] path_dist_tensor: {path_dist_tensor}")
        
        
        # loss_guide = torch.tensor(0.0, device=device)
        # if path_pixel_raw_tensor is not None and path_dist_tensor is not None:
        #     # --- A. 准备预测轨迹的累积距离 (Meters) ---
        #     # 补回第一个点的距离 0, 得到 [B, N]
        #     zeros_col = torch.zeros((batch_size, 1), device=device)
        #     # [B, N] 类似于: 0.0, 0.05, 0.10, ..., 0.56
        #     wp_cum_dist = torch.cumsum(torch.cat([zeros_col, wp_ds_masked], dim=1), dim=1)
            
        #     # --- B. 准备参考轨迹的累积距离 (Meters) ---
        #     # path_dist_tensor 已经是 Meters 了: 0.0, 0.05, ..., 1.43
        #     # [B, Ref_Len]
        #     ref_dist = path_dist_tensor 
            
        #     # 2. 查找索引
        #     # indices shape: [B, N]
        #     idx_right = torch.searchsorted(ref_dist, wp_cum_dist)
        #     if Debug:
        #         print(f"[DEBUG] wp_cum_dist shape: {wp_cum_dist.shape}")
        #         print(f"[DEBUG] wp_cum_dist:\n{wp_cum_dist}")
        #         print(f"[DEBUG] ref_dist shape: {ref_dist.shape}")
        #         print(f"[DEBUG] ref_dist:\n{ref_dist}")
        #         print(f"[DEBUG] idx_right before clamping:\n{idx_right}")
        #         # pdb.set_trace()
        #     # 3. 限制索引范围，防止越界
        #     # 必须保证 idx_right >= 1 且 <= max_len-1
        #     max_ref_idx = ref_dist.shape[1] - 1
        #     idx_right = torch.clamp(idx_right, 1, max_ref_idx)
        #     idx_left = idx_right - 1
            
        #     # 4. Gather 参考距离 (x轴)
        #     # ref_dist: [B, M] -> gather dim 1 using idx [B, N]
        #     x0 = torch.gather(ref_dist, 1, idx_left)   # [B, N] (例如 0.28m)
        #     x1 = torch.gather(ref_dist, 1, idx_right)  # [B, N] (例如 0.32m)
        #     if Debug:
        #         print(f"[DEBUG] idx_right after clamping:\n{idx_right}")
        #         print(f"[DEBUG] idx_left:\n{idx_left}")
        #         print(f"[DEBUG] x0:\n{x0}")
        #         print(f"[DEBUG] x1:\n{x1}")
        #         # pdb.set_trace()
        #     # 5. Gather 参考坐标 (y轴 / 实际上是 pos)
        #     # path_pixel_raw_tensor: [B, M, 2]
        #     idx_left_2d = idx_left.unsqueeze(-1).expand(-1, -1, 2)
        #     idx_right_2d = idx_right.unsqueeze(-1).expand(-1, -1, 2)
            
        #     y0 = torch.gather(path_pixel_raw_tensor, 1, idx_left_2d)
        #     y1 = torch.gather(path_pixel_raw_tensor, 1, idx_right_2d)
            
        #     if Debug:
        #         print(f"[DEBUG] idx_left_2d:\n{idx_left_2d}")
        #         print(f"[DEBUG] idx_right_2d:\n{idx_right_2d}")
        #         print(f"[DEBUG] y0 shape: {y0.shape}, y0:\n{y0}")
        #         print(f"[DEBUG] y1 shape: {y1.shape}, y1:\n{y1}")
        #         # pdb.set_trace()
            
        #     # 6. 计算线性插值权重 alpha
        #     denom = x1 - x0
        #     denom[denom == 0] = 1e-6 
        #     # alpha = (current_dist - start_dist) / (end_dist - start_dist)
        #     alpha = (wp_cum_dist - x0) / denom        # [B, N]
        #     alpha = alpha.unsqueeze(-1)               # [B, N, 1]
            
        #     # 7. 得到基于距离的引导点
        #     gt_guide_points = y0 + alpha * (y1 - y0)  # [B, N, 2]
            
        #     if Debug:
        #         print(f"[DEBUG] alpha shape: {alpha.shape}, alpha:\n{alpha}")
        #         print(f"[DEBUG] gt_guide_points shape: {gt_guide_points.shape}, gt_guide_points:\n{gt_guide_points}")
        #         # pdb.set_trace()
            
        #     # --- D. 计算 Loss ---
        #     # ... (这部分保持不变) ...
        #     valid_mask_expanded = (~dense_mask).unsqueeze(-1).float()
        #     se_per_point = (waypoints - gt_guide_points) ** 2
        #     masked_se = se_per_point * valid_mask_expanded
        #     sum_se = torch.sum(masked_se)
            
        #     total_valid_elements = torch.sum(valid_dense_len) * 2
        #     loss_guide = sum_se / (total_valid_elements + 1e-6)
        
        # B 组：单步进展约束（防绕圈）
        dist_to_goal_seq = torch.norm(goal[:, None, :2] - waypoints[:, :, :2], dim=2)  # [B, N]
        delta_progress = dist_to_goal_seq[:, 1:] - dist_to_goal_seq[:, :-1]  # [B, N-1]
        prog_penalty = torch.relu(delta_progress + self.prog_margin)
        prog_penalty = prog_penalty.masked_fill(seg_mask, 0.0)
        lprog = torch.mean(torch.sum(prog_penalty, dim=1) / num_segments)

        # C 组：长度比约束（抑制大绕圈）
        if distance is not None:
            d_geo = distance.float().to(device).view(-1)
        else:
            d_geo = torch.norm(goal[:, :2], dim=1)
        d_geo = torch.clamp(d_geo, min=1e-3)
        len_ratio = current_path_len / d_geo
        llen = torch.mean(torch.relu(len_ratio - self.len_ratio_target) ** 2)

        # Total Loss：按 A/B/C 组合
        trajectory_loss = self.w_obs * oloss + self.w_goal * gloss + self.w_motion * mloss
        exp_mode = str(self.exp_variant).upper()
        if exp_mode in ["B", "C"]:
            trajectory_loss = trajectory_loss + self.w_prog * lprog
        if exp_mode == "C":
            trajectory_loss = trajectory_loss + self.w_len * llen
        # + self.w_guide * loss_guide

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
            # pdb.set_trace()
        
        # 2. 屏蔽超过预瞄距离(ahead_dist)的点
        # 需要把 goal_dists [B, N-1] 补齐到 [B, N] (第一点距离为0)
        zeros = torch.zeros((batch_size, 1), device=device)
        dists_full = torch.cat([zeros, goal_dists], dim=1) # [B, N]
        dists_expanded = torch.cat([dists_full, dists_full, dists_full], dim=0) # [3B, N]
        
        raw_cost_expanded[dists_expanded > ahead_dist] = 0.0
        if Debug:
            print(f"[DEBUG] raw_cost_expanded after ahead_dist masking:\n{raw_cost_expanded}")
            # pdb.set_trace()
        
        # 3. 取最大值 (判断是否碰撞) [3B, N] -> [B, 3] -> [B, 1]
        # view 成 [3, B, N] 然后 max dim=2 (along path)
        max_cost_along_path, _ = torch.max(raw_cost_expanded.view(3, batch_size, -1), dim=2) # [3, B]
        if Debug:
            print(f"[DEBUG] max_cost_along_path shape: {max_cost_along_path.shape}")
            print(f"[DEBUG] max_cost_along_path:\n{max_cost_along_path}")
            # pdb.set_trace()
        # 只要 Left/Center/Right 任意一个撞了就算撞
        is_collision = torch.any(max_cost_along_path > self.obstalce_thread, dim=0).float().unsqueeze(1) # [B, 1]
        if Debug:
            print(f"[DEBUG] is_collision shape: {is_collision.shape}")
            print(f"[DEBUG] is_collision:\n{is_collision}")
            # pdb.set_trace()
        fear_loss = torch.nn.BCELoss()(fear, is_collision)

        return trajectory_loss + 0.1 * fear_loss, trajectory_loss, self.w_motion * mloss, fear_loss, lprog, llen
    # , self.w_guide * loss_guide
    
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
            # pdb.set_trace()
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
                # plt.show(block=True)
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
            binary_obs = (grid == 0).astype(np.float32)
            dist_to_obs = ndimage.distance_transform_edt(1 - binary_obs) * self.res
            cost_free = np.exp(-3.0 * dist_to_obs) # 衰减系数可调，2.0比3.0更平缓，梯度传得更远
            dist_inside = ndimage.distance_transform_edt(binary_obs) * self.res
            cost_obs = 1.0 + 2.0 * dist_inside
            final_cost = np.where(binary_obs == 1, cost_obs, cost_free)
            final_cost = gaussian_filter(final_cost, sigma=1.0)
            out_maps.append(final_cost)
        return np.array(out_maps)[:, np.newaxis, :, :] # [B, 1, H, W]

    def collided_count(self, waypoints, cost_maps, threshold=0.8):
        """
        计算轨迹中碰撞点的数量 (Cost > threshold)
        waypoints: [B, N, 2]
        cost_maps: [B, 1, H, W]
        """
        B, N, _ = waypoints.shape
        H, W = cost_maps.shape[2], cost_maps.shape[3]
        
        # 物理坐标 -> 像素坐标
        res = 0.1
        center_row = H / 2.0
        center_col = W / 2.0
        
        px = waypoints[..., 0]  # x (meters)
        py = waypoints[..., 1]  # y (meters)
        
        row_idx = (center_row - px / res).long()
        col_idx = (center_col - py / res).long()
        
        # Clamp to valid range
        row_idx = torch.clamp(row_idx, 0, H - 1)
        col_idx = torch.clamp(col_idx, 0, W - 1)
        
        # Gather cost values
        batch_indices = torch.arange(B, device=waypoints.device).unsqueeze(1).expand(-1, N) # [B, N]
        cost_values = cost_maps[batch_indices, 0, row_idx, col_idx] # [B, N]
        
        # Count collisions
        collided_points = (cost_values > threshold).float()
        collision_count = collided_points.sum(dim=1) # [B]
        
        return collision_count

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1) # DDP 自动传入
    return parser.parse_args()

def worker_init_fn(worker_id):
    # 确保每个 worker 的 numpy 随机种子不同
    # 结合: 当前随机种子 + worker ID
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)

def train_pipeline():
    # 1. 初始化配置
    
    # -------------------------------------------------------
    # 2. 修改 DataLoader 构建函数
    # -------------------------------------------------------
    def build_dataloader(dataset, mode='train'):
        if use_ddp:
            # DDP 模式：必须使用 DistributedSampler，shuffle=False
            sampler = DistributedSampler(dataset, shuffle=(mode=='train'))
            shuffle = False 
        else:
            # 单卡模式：不使用 Sampler，直接靠 shuffle=True 乱序
            sampler = None
            shuffle = (mode == 'train')

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,        # 单卡 True, DDP False
            sampler=sampler,        # 单卡 None, DDP Sampler
            num_workers=cfg.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            multiprocessing_context='spawn'
        )
        return loader, sampler
    
    args = parse_args()
        
    # -------------------------------------------------------
    # 1. 自动环境检测 (这段必须保留！)
    # -------------------------------------------------------
    if "WORLD_SIZE" in os.environ:
        # 进入这里说明是 torchrun 启动的 (无论 nproc 是 1 还是 2)
        use_ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[INFO] DDP Mode Detected. Rank: {dist.get_rank()}, World Size: {world_size}")
    else:
        # 进入这里说明是 python 直接运行的 (单卡调试)
        use_ddp = False
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Single GPU Mode Detected. Device: {device}")

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 仅在主进程打印 Log
    if local_rank == 0:
        print(f"[INFO] DDP Initialized. Using device: {device}")
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        if os.path.exists(cfg.log_dir + '/tb'):
            shutil.rmtree(cfg.log_dir + '/tb')
        writer = SummaryWriter(log_dir=cfg.log_dir + '/tb')
    
    stage_goals = {
        0: {'target_loss': 3.2, 'collision_rate': 0.12, 'min_epochs': 200},
        1: {'target_loss': 3.23, 'collision_rate': 0.10, 'min_epochs': 220},
        2: {'target_loss': 3.42, 'collision_rate': 0.08, 'min_epochs': 200},
        3: {'target_loss': 4.89, 'collision_rate': 0.07, 'min_epochs': 300},
        4: {'target_loss': 4.1, 'collision_rate': 0.06, 'min_epochs': 260},
        5: {'target_loss': 3.2, 'collision_rate': 0.06, 'min_epochs': 260}
    }
    
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
    train_loader, train_sampler = build_dataloader(train_dataset, 'train')
    
    # 验证集
    val_dataset = CollectData(
        root_dir=cfg.data_root,
        mode='val',
        split_ratio=0.9,
        safe_dist_threshold=2.0,
        config=cfg.val_aug_config
    )
    val_loader, val_sampler = build_dataloader(val_dataset, 'val')
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- B. 模型初始化 ---
    # print("[INFO] Initializing CNN Planner...")
    # model = AutoEncoderGrid(
    #     encoder_channel=64, 
    #     max_dist=cfg.max_dist, 
    #     step_size=cfg.step_size  # 保留您的设定 0.3
    # ).to(device)
    global USING_transformer
    if USING_transformer:
        # === 修改为 Transformer ===
        print("[INFO] Initializing Transformer Planner...")
        model = TransformerPlanner(
            encoder_channel=512, # PlannerNetGrid 的输出通道
            d_model=512,         # 保持一致效率最高
            nhead=8,             # 8头注意力
            num_layers=4,        # 4层 Transformer Encoder (对于80x80地图足够了)
            max_dist=cfg.max_dist,
            step_size=cfg.step_size
        ).to(device)
    else:
        # === 保持原有 CNN 结构 ===
        print("[INFO] Initializing CNN Planner...")
        model = AutoEncoderGrid(
            encoder_channel=64, 
            max_dist=cfg.max_dist, 
            step_size=cfg.step_size  # 保留您的设定 0.3
        ).to(device)
    
    start_epoch = 0
    curr_stage = 0 
    resume_path = f"{cfg.save_dir}/last_checkpoint.pth"
    
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10
    # )
    if USING_transformer:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        from torch.optim.lr_scheduler import OneCycleLR
        # 假设总步数
        total_steps = cfg.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.1, # 前 10% 的步数用来 Warmup (从 0 升到 max_lr)
            anneal_strategy='cos', # 之后用余弦退火降下来
            div_factor=10.0,
            final_div_factor=100.0
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        curr_stage = ckpt['curr_stage']
        best_val_loss = ckpt['best_val_loss']
        train_dataset.set_stage(curr_stage)
        val_dataset.set_stage(curr_stage)
    
    if use_ddp:
        # 【关键】只有在多卡环境下才转换 SyncBatchNorm
        # 单卡或者 torchrun nproc=1 时使用 SyncBN 可能会卡死
        if world_size > 1:
            print("[INFO] Converting to SyncBatchNorm...")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # 使用 DDP 包装
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        print("[INFO] Running plain model (No DDP wrapper)")
    
    # --- C. 工具类初始化 ---
    # 使用 BatchTrajCost 替代原始 TrajCost
    batch_traj_cost = BatchTrajCost(
        gpu_id=0 if torch.cuda.is_available() else "cpu",
        w_obs=cfg.w_obs,
        w_motion=cfg.w_motion,
        w_goal=cfg.w_goal,
        w_guide=cfg.w_guide,
        obstalce_thread=cfg.fear_ahead_dist
    )
    batch_traj_cost.exp_variant = cfg.exp_variant
    batch_traj_cost.obs_topk_ratio = cfg.obs_topk_ratio
    batch_traj_cost.obs_safe_threshold = cfg.obs_safe_threshold
    batch_traj_cost.obs_mean_alpha = cfg.obs_mean_alpha
    batch_traj_cost.obs_topk_beta = cfg.obs_topk_beta
    batch_traj_cost.obs_max_gamma = cfg.obs_max_gamma
    batch_traj_cost.obs_barrier_delta = cfg.obs_barrier_delta
    batch_traj_cost.w_prog = cfg.w_prog
    batch_traj_cost.prog_margin = cfg.prog_margin
    batch_traj_cost.w_len = cfg.w_len
    batch_traj_cost.len_ratio_target = cfg.len_ratio_target

    if local_rank == 0:
        print(f"[INFO] Experiment Variant: {cfg.exp_variant}")
    map_proc = MapProcessor()

    # --- D. 训练循环 ---
    print("[INFO] Start Training Loop...")
    best_val_loss = float('inf')
     
    train_dataset.set_stage(curr_stage)
    val_dataset.set_stage(curr_stage)
    print(f"[INFO] Reset Stage to {curr_stage} based on epoch {start_epoch}")
    
    # curr_stage = 0
    # train_dataset.set_stage(curr_stage)
    # val_dataset.set_stage(curr_stage)  # 验证集也要同步！
    
    for epoch in range(start_epoch, cfg.epochs):
        # train_sampler.set_epoch(epoch)
        if use_ddp and train_sampler is not None: 
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        model.train()
        train_loss_total = 0.0
        
        # 为了保证能顺利收敛，在训练前50轮，只考虑使用引导代价（guide loss），不计算碰撞损失（oloss）和目标损失（gloss）。等引导损失稳定后，再逐渐引入其他损失。
        # pretrain_epoch = 200
        # target_epoch = stage_goals[0]['min_epochs']
        # step_size = 20
        # if epoch < pretrain_epoch:
        #     # 阶段 1: 预训练，无引导
        #     batch_traj_cost.w_obs = cfg.w_obs
        #     batch_traj_cost.w_goal = cfg.w_goal
        #     batch_traj_cost.w_motion = 0
        #     batch_traj_cost.w_guide = cfg.w_guide * 10
        # elif epoch < target_epoch:
        #     # 阶段 2: 阶梯式上升
        #     # 去掉了那个奇怪的 "-20"
        #     # 计算总共有多少个台阶
        #     total_steps = (target_epoch - pretrain_epoch) // step_size
        #     # 防止除以零 (如果总轮数差不足20轮，直接设为满权重或做特殊处理)
        #     if total_steps < 1:
        #         total_steps = 1   
        #     # 当前在第几个台阶
        #     current_step = (epoch - pretrain_epoch) // step_size
        #     # 计算 alpha (0.0 -> 1.0)
        #     alpha = min(current_step / total_steps, 1.0)
        #     batch_traj_cost.w_obs = cfg.w_obs
        #     batch_traj_cost.w_goal = cfg.w_goal
        #     batch_traj_cost.w_motion = cfg.w_motion * alpha
        #     start_w_guide = cfg.w_guide * 10
        #     end_w_guide = cfg.w_guide
        #     batch_traj_cost.w_guide = start_w_guide + (end_w_guide - start_w_guide) * alpha
        #     # batch_traj_cost.w_guide = cfg.w_guide
        # else:
        #     # 阶段 3: 稳定阶段
        #     batch_traj_cost.w_obs = cfg.w_obs
        #     batch_traj_cost.w_goal = cfg.w_goal
        #     batch_traj_cost.w_motion = cfg.w_motion
        #     batch_traj_cost.w_guide = cfg.w_guide
        
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}, Stage {curr_stage}")
        for batch_idx, (map_tensor, goal_tensor, dist_tensor, path_pixel_raw_tensor, path_dist_tensor) in enumerate(pbar):
            # map_tensor: [B, 1, 80, 80] (1=Free)
            global Debug
            if Debug:
                # 获取当前 Batch 有多少个样本 (通常是 2)
                current_batch_size = map_tensor.shape[0]
                
                # --- 修正点 1: 遍历当前 Batch 的样本数，而不是 batch_idx ---
                for sample_idx in range(current_batch_size):
                    # 1. 转换图像
                    raw_map = map_tensor[sample_idx, 0].detach().cpu().numpy()

                    # 打印极值，确认数据是否正常
                    print(f"[Debug Vis] Batch {batch_idx} Sample {sample_idx}: Map Range [{raw_map.min():.2f}, {raw_map.max():.2f}]")

                    # 2. 获取目标点 (像素坐标)
                    goal_col = int(goal_tensor[sample_idx, 0].item())
                    goal_row = int(goal_tensor[sample_idx, 1].item())

                    # 3. 使用 matplotlib 画图
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(raw_map, cmap='gray', origin='upper')

                    # 画终点 (红色星号)
                    ax.plot(goal_col, goal_row, 'r*', markersize=15, label='Goal', markeredgecolor='yellow', markeredgewidth=1.5)

                    # 画起点 (绿色圆点) - 假设地图中心是起点
                    center_x, center_y = 40, 40  # 80x80的一半
                    ax.plot(center_x, center_y, 'go', markersize=10, label='Start', markeredgecolor='white', markeredgewidth=1.5)

                    ax.set_xlabel('Column (pixels)', fontsize=12)
                    ax.set_ylabel('Row (pixels)', fontsize=12)
                    ax.set_title(f'Debug View - Batch {batch_idx} Sample {sample_idx}', fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')

                    plt.tight_layout()
                    # plt.show(block=True)
                    plt.close()

                    print(f"    Goal Pixel: ({goal_col}, {goal_row})")
                    print("    Press Any Key to continue (close window)...")
                
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
                    # plt.show(block=True)
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
            if USING_transformer:
                preds, pred_fear, mask = model(net_input, goal_norm, real_dist=dist_tensor.to(device))
            else:
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
                    # plt.show(block=True)
                    plt.close()
                    
            # --- 诊断日志：有效长度与终点距离 ---
            if batch_idx % 50 == 0:
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
                    
                    if local_rank == 0:
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
            
            # 注意，这里忽略的scale,其实实际存储的地图是81*81,但是后面统一resize成80,在求距离时，并没有更改，但是缩放比例接近1：1,这个地方先不改了
            # --- 计算 Loss (Batch) ---
            # loss, l_traj, l_motion, l_fear, l_guide = batch_traj_cost.CostofTraj_Batch(
            loss, l_traj, l_motion, l_fear, lprog, llen = batch_traj_cost.CostofTraj_Batch(
                waypoints=waypoints,
                goal=goal_meters,
                fear=pred_fear,
                log_step=epoch,
                ahead_dist=cfg.fear_ahead_dist,
                batch_maps=net_input,  # 传入与网络输入一致的 Batch Map
                sub_step_size=cfg.sub_step_size,
                mask=mask,
                distance=dist_tensor.to(device),
                path_pixel_raw_tensor=path_pixel_raw_tensor.to(device),
                path_dist_tensor=path_dist_tensor.to(device)
            )
            
            loss.backward()
            if local_rank == 0:
                writer.add_scalar('Loss/Train_Total', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/Train_Traj', l_traj.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/Train_Motion', l_motion.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/Train_Fear', l_fear.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/Train_Lprog', lprog.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/Train_Llen', llen.item(), epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('Loss/Train_Guide', l_guide.item(), epoch * len(train_loader) + batch_idx)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if USING_transformer:
                try:
                    scheduler.step()
                except Exception as e:
                    # 如果步数超了，就保持最后的 LR
                    pass
            
            train_loss_total += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'traj': l_traj.item()})
            
            # 可视化 (每 50 batch 保存一张)
            if batch_idx % 50 == 0 and local_rank == 0:
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
                    sub_step_size=cfg.sub_step_size,  # <--- 新增，确保截断准确
                    curr_stage=curr_stage  # <--- 新增，显示当前阶段信息
                )

        # --- E. 验证循环 ---
        total_samples = 0
        collided_samples = 0
        
        model.eval()
        val_loss_total = 0.0
        val_lprog_total = 0.0
        val_llen_total = 0.0
        with torch.no_grad():
            for batch_idx, (map_tensor, goal_tensor, distance_tensor, path_pixel_raw_tensor, path_dist_tensor) in enumerate(val_loader):
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
                
                # preds, pred_fear, mask = model(net_input, goal_norm)
                
                if USING_transformer:
                    preds, pred_fear, mask = model(net_input, goal_norm, real_dist=distance_tensor.to(device))
                else:
                    preds, pred_fear, mask = model(net_input, goal_norm)
                
                waypoints = batch_traj_cost.opt.TrajGeneratorFromPFreeRot(
                    preds, step=cfg.sub_step_size, mask=mask
                )
                
                # 计算碰撞数量，对于避障很重要
                collided_count = map_proc.collided_count(waypoints, net_input)
                collided_samples += (collided_count > 0).sum().item()
                total_samples += waypoints.shape[0]
                
                # loss, _, __, ___, ____ = batch_traj_cost.CostofTraj_Batch(
                loss, _, __, ___, lprog_val, llen_val = batch_traj_cost.CostofTraj_Batch(
                    waypoints=waypoints,
                    goal=goal_meters,
                    fear=pred_fear,
                    log_step=epoch,
                    ahead_dist=cfg.fear_ahead_dist,
                    batch_maps=net_input,
                    sub_step_size=cfg.sub_step_size,
                    mask=mask,
                    distance=distance_tensor.to(device),
                    path_pixel_raw_tensor=path_pixel_raw_tensor.to(device),
                    path_dist_tensor=path_dist_tensor.to(device)
                )
                val_loss_total += loss.item()
                val_lprog_total += lprog_val.item()
                val_llen_total += llen_val.item()
                
                if batch_idx % 100 == 0 and local_rank == 0:
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
                        sub_step_size=cfg.sub_step_size,  # <--- 新增
                        curr_stage=curr_stage  # <--- 新增，显示当前阶段信息
                    )
        
        # --- F. 统计与保存 ---
        # avg_train_loss = train_loss_total / len(train_loader)
        # avg_val_loss = val_loss_total / len(val_loader)
        
        metrics_tensor = torch.tensor([
            train_loss_total, 
            val_loss_total, 
            collided_samples, 
            total_samples,
            val_lprog_total,
            val_llen_total,
        ], dtype=torch.float32, device=device)
        
        if use_ddp:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            # 平均 Loss = 总 Loss / (Steps * 卡数)
            avg_train_loss = metrics_tensor[0].item() / (len(train_loader) * world_size)
            avg_val_loss = metrics_tensor[1].item() / (len(val_loader) * world_size)
        else:
            # 单卡模式不需要 reduce，直接除以 steps 即可
            avg_train_loss = metrics_tensor[0].item() / len(train_loader)
            avg_val_loss = metrics_tensor[1].item() / len(val_loader)
        
        global_train_loss = metrics_tensor[0].item()
        global_val_loss = metrics_tensor[1].item()
        global_collided = metrics_tensor[2].item()
        global_total = metrics_tensor[3].item()
        global_val_lprog = metrics_tensor[4].item()
        global_val_llen = metrics_tensor[5].item()
        
        # world_size = dist.get_world_size()
        
        if use_ddp:
            world_size = dist.get_world_size()
        else:
            world_size = 1
        
        avg_train_loss = global_train_loss / (len(train_loader) * world_size)
        avg_val_loss = global_val_loss / (len(val_loader) * world_size)
        avg_val_lprog = global_val_lprog / (len(val_loader) * world_size)
        avg_val_llen = global_val_llen / (len(val_loader) * world_size)
        val_collision_rate = global_collided / global_total if global_total > 0 else 0.0
        # [修改点 1]：在所有 Rank 上都计算碰撞率，不要放在 if local_rank == 0 里
        # 假设 total_samples 和 collided_samples 在之前的步骤中已经做过 all_reduce (汇总) 或者即使是局部的也足以作为判断依据
        # val_collision_rate = collided_samples / total_samples if total_samples > 0 else 0.0
    
        # [修改点 2]：只让 Rank 0 负责打印日志和写 TensorBoard
        if local_rank == 0:
            writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
            writer.add_scalar('Loss/Val_Lprog', avg_val_lprog, epoch)
            writer.add_scalar('Loss/Val_Llen', avg_val_llen, epoch)
            writer.add_scalar('Metrics/Val_CollisionRate', val_collision_rate, epoch)
    
        # [修改点 3]：阶段判断逻辑必须在主干道上（不要缩进到 local_rank==0 里）
        # 这样所有显卡都会同时切换到下一个 Stage
        min_total_epochs = sum([stage_goals[s]['min_epochs'] for s in range(curr_stage + 1)])
        print(f"avg_val_loss is: {avg_val_loss}, val_collision_rate is: {val_collision_rate}, epoch is: {epoch}")
        print(f"stage th, loss is: {stage_goals[curr_stage]['target_loss']}, collision rate is: {stage_goals[curr_stage]['collision_rate']}, min epoch is: {min_total_epochs}")
        if avg_val_loss < stage_goals[curr_stage]['target_loss'] and val_collision_rate < stage_goals[curr_stage]['collision_rate'] and epoch >= min_total_epochs:
            if local_rank == 0: # 只有 Rank 0 打印文字，避免刷屏
                print(f">>> Stage {curr_stage} Passed! Moving to next stage.")
            
            curr_stage += 1
            
            if curr_stage > max(stage_goals.keys()):
                if local_rank == 0:
                    print(">>> All stages passed! Training complete.")
                curr_stage = max(stage_goals.keys())
            else:
                # 关键：所有 Rank 都要更新 Dataset
                train_dataset.set_stage(curr_stage)
                val_dataset.set_stage(curr_stage)
                if local_rank == 0:
                    print(f"[System] DataLoaders rebuilt for Stage {curr_stage}")
                train_loader, train_sampler = build_dataloader(train_dataset, 'train')
                val_loader, val_sampler = build_dataloader(val_dataset, 'val')
                
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = cfg.lr * (0.9 ** curr_stage)
                    
                if not USING_transformer:
                # 只有在非 Transformer (ReduceLROnPlateau) 模式下才手动改 LR
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = cfg.lr * (0.9 ** curr_stage)
                else:
                    if local_rank == 0:
                        print(f"[Info] Stage {curr_stage} transition. OneCycleLR is still managing the curve.")
        
        if not USING_transformer:
            scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if local_rank == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if use_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'curr_stage': curr_stage,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, f"{cfg.save_dir}/best_model.pth")
            # torch.save(model.state_dict(), f"{cfg.save_dir}/best_model.pt")
            print(">>> Best Model Saved!")
            
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if use_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'curr_stage': curr_stage,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, f"{cfg.save_dir}/last_checkpoint.pth")
            #  torch.save(model.state_dict(), f"{cfg.save_dir}/epoch_{epoch+1}.pt")
    
    
    if local_rank == 0:
        writer.close()
    
    if use_ddp:
        dist.destroy_process_group()
    
    
def visualize_debug(cost_maps, waypoints, preds, mask, goals, epoch, idx, save_dir, res=0.1, sub_step_size=0.2, curr_stage=0):
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
    plt.savefig(f"{save_dir}/debug_{curr_stage}_{epoch}_{idx}.png", dpi=100)
    plt.close()

if __name__ == "__main__":
    train_pipeline()