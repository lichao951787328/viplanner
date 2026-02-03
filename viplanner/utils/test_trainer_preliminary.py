import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# === 导入模块 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(f"[INFO] 已将父目录加入搜索路径: {parent_dir}")
try:
    from plannernet.autoencoder_myself_cubic import AutoEncoderGrid
    from traj_cost_opt.traj_cost_myself_cubic import TrajCost
except ImportError as e:
    print(f"【错误】导入失败: {e}")
    print("请确保 autoencoder_grid.py, traj_cost_myself.py 在当前目录下")
    exit()

def create_dummy_npy_map(filename="costmap.npy", size=80):
    """如果没找到地图，生成一个带障碍物的假地图"""
    if os.path.exists(filename):
        return
    print(f"[INFO] 生成测试用地图: {filename}")
    # 0 为障碍物，1 为空地 (对应 Occupancy Grid 的习惯)
    # 但 TrajCost 里通常 CostMap: 0 是空地，1 是障碍物?
    # 查看 SimpleCostMapWrapper: 它直接读取 npy。
    # 假设: npy 存的是 Cost (0~1, 1=障碍物)
    map_data = np.zeros((size, size), dtype=np.float32)
    
    # 画几个障碍物 (Cost = 1.0)
    # 障碍物 1: 面前的横墙
    map_data[30:35, 20:60] = 1.0 
    # 障碍物 2: 右上角的块
    map_data[50:70, 50:70] = 1.0
    
    np.save(filename, map_data)


def find_valid_goal(traj_cost, min_dist=2.0, max_dist=4.0):
    cost_map_np = traj_cost.cost_map.cost_array.cpu().numpy() # [H, W]
    
    # 获取地图尺寸
    H, W = cost_map_np.shape
    
    for i in range(1000):
        # 1. 随机生成物理坐标
        rand_x = np.random.uniform(-max_dist, max_dist)
        rand_y = np.random.uniform(-max_dist, max_dist)
        
        # 2. 距离过滤 (圆环区域)
        dist = np.sqrt(rand_x**2 + rand_y**2)
        if dist < min_dist or dist > max_dist:
            continue
        
        # 3. 坐标转换 (关键！尽量使用 traj_cost 内部的方法以确保统一)
        # 如果 traj_cost 没有公开 Pos2Ind，这里保留你的逻辑，但请务必确认符号
        # 建议打印几个点验证一下：例如 (1,0) 是不是在前方
        
        # 假设你的逻辑是对的 (X向上为负Row, Y向左为负Col? 需确认)
        # 更好的做法是：如果类里有 self.Pos2Ind(x, y)，请直接用
        resolution = traj_cost.cost_map.cfg.general.resolution
        c_row, c_col = traj_cost.cost_map.center_index
        
        # 使用 round 而不是直接 int 截断，提高精度
        r_idx = int(np.round(c_row - (rand_x / resolution)))
        c_idx = int(np.round(c_col - (rand_y / resolution)))
        
        # 4. 边界检查
        if 0 <= r_idx < H and 0 <= c_idx < W:
            # 5. 安全检查 (不仅仅是中心点，最好周围一圈都安全)
            # 简单版：要求 Cost 极低 (0.00 代表完全无障碍)
            # 假设障碍物是 1.0，膨胀区是 0.0-1.0
            if cost_map_np[r_idx, c_idx] <= 0.01: 
                # [可选] 检查该点周围 3x3 区域是否也都安全，防止紧贴墙壁
                # sub_map = cost_map_np[max(0, r_idx-1):r_idx+2, max(0, c_idx-1):c_idx+2]
                # if np.max(sub_map) > 0.05: continue 

                return torch.tensor([[rand_x, rand_y, 0.0]], device=traj_cost.device)
    
    print("[WARN] 未找到合适终点，使用默认前方点")
    return torch.tensor([[2.5, 0.0, 0.0]], device=traj_cost.device)


def test_integration():
    print("\n" + "="*30 + " 全流程集成测试 " + "="*30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备地图
    NPY_MAP_NAME = "/home/eai/VLN/viplanner/rotated_out/carla/sample_01713/costmap.npy"
    if os.path.exists(NPY_MAP_NAME):
        check_data = np.load(NPY_MAP_NAME)
        print(f"\n[DATA CHECK] 地图数据统计:")
        print(f"  - Shape: {check_data.shape}")
        print(f"  - Min: {check_data.min()}")
        print(f"  - Max: {check_data.max()}")
        print(f"  - Mean: {check_data.mean()}")
        
        # 简单的自动判断建议
        if check_data.max() > 1.1:
            print("[SUGGESTION] 数据范围 > 1，强烈建议归一化到 [0, 1]！")
        else:
            print("[SUGGESTION] 数据范围看起来已经是 [0, 1] 了，可能不需要除以255。")
    
    # 2. 初始化 Cost Function
    # 注意: TrajCost 内部会加载地图
    
    # fear_ahead_dist: float = 2.5
    # "fear lookahead distance"
    # w_obs: float = 0.25
    # w_height: float = 1.0
    # w_motion: float = 1.5
    # w_goal: float = 4.0
    
    traj_cost = TrajCost(
        gpu_id=0 if torch.cuda.is_available() else "cpu",
        log_data=False,
        w_obs=0.5,      # 降低障碍物权重 - 环境太拥挤，需要让网络敢于探索
        w_motion=1.5,   # 保持运动权重
        w_goal=8.0,     # 目标权重 - 必须向目标前进
        obstalce_thread=2
    )
    # 加载地图 (SimpleCostMapWrapper)
    # 这里 root_path 传当前目录 ".", map_name 传文件名
    traj_cost.SetMap(".", NPY_MAP_NAME)
    
    # 在指定位置添加障碍物区域
    # 获取 cost_map 的 tensor
    cost_map_tensor = traj_cost.cost_map.cost_array  # [H, W]
    H, W = cost_map_tensor.shape

    # 定义障碍物区域 (物理坐标系)
    # 例如：在前方右侧添加一个 2m x 2m 的障碍物
    obstacle_center_x = 2.0  # 前方2米
    obstacle_center_y = 2  # 左侧1.5米
    obstacle_width = 2.0  # 宽度2米
    obstacle_height = 2.0  # 高度2米

    # 转换到像素坐标
    resolution = traj_cost.cost_map.cfg.general.resolution
    c_row, c_col = traj_cost.cost_map.center_index

    # 计算障碍物区域的像素范围
    obs_row_start = int(np.round(c_row - (obstacle_center_x + obstacle_height/2) / resolution))
    obs_row_end = int(np.round(c_row - (obstacle_center_x - obstacle_height/2) / resolution))
    obs_col_start = int(np.round(c_col - (obstacle_center_y + obstacle_width/2) / resolution))
    obs_col_end = int(np.round(c_col - (obstacle_center_y - obstacle_width/2) / resolution))

    # 边界检查
    obs_row_start = max(0, min(obs_row_start, H-1))
    obs_row_end = max(0, min(obs_row_end, H-1))
    obs_col_start = max(0, min(obs_col_start, W-1))
    obs_col_end = max(0, min(obs_col_end, W-1))

    # 添加障碍物 (设置为高cost值，1.0表示完全障碍)
    cost_map_tensor[obs_row_start:obs_row_end+1, obs_col_start:obs_col_end+1] = 1.0

    print(f"[INFO] 已在物理坐标 ({obstacle_center_x:.1f}, {obstacle_center_y:.1f}) 处添加障碍物")
    print(f"       像素范围: Row[{obs_row_start}:{obs_row_end}], Col[{obs_col_start}:{obs_col_end}]")
    
    print(f"traj_cost.cost_map.cost_array.shape: {traj_cost.cost_map.cost_array.shape}")
    
    # 添加代价地图统计信息
    costmap_array = cost_map_tensor
    # costmap_array = traj_cost.cost_map.cost_array
    print(f"\n[COSTMAP STATISTICS]")
    print(f"  - Min: {costmap_array.min().item():.4f}")
    print(f"  - Max: {costmap_array.max().item():.4f}")
    print(f"  - Mean: {costmap_array.mean().item():.4f}")
    print(f"  - Median: {costmap_array.median().item():.4f}")
    print(f"  - Std: {costmap_array.std().item():.4f}")
    print(f"  - Pixels >0.5: {(costmap_array > 0.5).sum().item()} / {costmap_array.numel()}")
    print(f"  - Pixels >0.75: {(costmap_array > 0.75).sum().item()} / {costmap_array.numel()}")
    print()
    
    # 3. 初始化模型
    # 假设地图分辨率 0.1, 80x80 -> 8m x 8m, max_dist 设为 5.0m
    model = AutoEncoderGrid(encoder_channel=64, max_dist=8.0, step_size=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
    # 4. 准备输入数据
    # 输入: 地图 Tensor [B, 1, 80, 80]
    # TrajCost 里的 cost_map.cost_array 是 [H, W]
    # 我们需要把它变成网络输入的形状
    map_tensor = traj_cost.cost_map.cost_array.clone().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    # === 新增：强制 Resize 到 80x80 ===
    # 无论原始地图多大，都缩放到模型训练时的标准尺寸
    # 注意：align_corners=False 对于空间对齐通常更好
    if map_tensor.shape[-1] != 80 or map_tensor.shape[-2] != 80:
        print(f"[INFO] 检测到地图尺寸为 {map_tensor.shape[-2:]}，正在缩放至 (80, 80)...")
        map_tensor = torch.nn.functional.interpolate(
            map_tensor, 
            size=(80, 80), 
            mode='bilinear', 
            align_corners=False
        )
    map_tensor = map_tensor.to(device)
    
    odom = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    goal = torch.tensor([[3.5, 2.0]], dtype=torch.float32, device=device)
    
    dummy_fear_pred = torch.tensor([[0.2]], device=device, requires_grad=True)

    print("\n[INFO] 开始 Overfit 训练 (1000 Steps)...")
    model.train()
    
    losses = []
    step_density = 0.1
    
    # 添加更详细的loss分解信息
    print("\n训练进度:")
    print("-" * 60)
    
    for i in range(10001):  # 增加到1000步
        optimizer.zero_grad()
        
        preds, pred_fear, mask = model(map_tensor, goal)
        
        # A. 插值生成密集轨迹 包含起点
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(
            preds,
            step=step_density,  # 插值步长
            mask=mask
        )
        
        # 2. 获取实际生成的长度 包含起止点和终点
        B, N, D = waypoints.shape

        # 3. 直接根据长度生成 xs，确保一一对应
        # 假设 xs 代表路径长度索引，步长为 step_density
        # 注意：要确保 device 一致
        xs = torch.arange(0, N, device=device, dtype=waypoints.dtype).unsqueeze(0) * step_density

        # 4. 这里的 valid_t 计算逻辑保持不变
        valid_t = (~mask).sum(dim=1).float()  # [B]
            
        # C. 生成密集 Mask
        # 如果时间 t > valid_t，则为无效 (True)
        dense_mask = xs > valid_t.unsqueeze(1)  # [B, N_dense]
        
        loss = traj_cost.CostofTraj(
            waypoints=waypoints,  # 传入插值轨迹 [B, N, 2]
            goal=goal,
            fear=pred_fear,
            log_step=i,
            ahead_dist=2.0,
            step=step_density,
            mask=dense_mask,  # 密集轨迹的mask [B, N]
            num_keypoints=32  # 指定关键点数量用于motion loss
        )
        # + w_ref * ref_loss
        
        loss.backward()
        
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss)  # 学习率调度
        
        losses.append(loss.item())
        
        if i % 50 == 0:
            # 计算梯度范数用于监控
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"Step {i:4d}: Loss={loss.item():.4f} | "
                  f"Pred[{preds.min().item():+.2f},{preds.max().item():+.2f}] | "
                  f"Grad={total_norm:.2f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # 每200步打印一次详细的预测点信息
        if i % 200 == 0 and i > 0:
            # 打印loss分量
            if hasattr(traj_cost, '_last_loss_components'):
                lc = traj_cost._last_loss_components
                print(f"\n  → Loss分解 (Step {i}):")
                print(f"     Obs={lc['obs']:.4f} | Motion={lc['motion']:.4f} | "
                      f"Goal={lc['goal']:.4f} | Fear={lc['fear']:.4f} | Total={lc['total']:.4f}")
                
                # 添加goal loss的子项
                if 'goal_dict' in lc:
                    gd = lc['goal_dict']
                    if 'direction' in gd:
                        print(f"     Goal细分: endpoint={gd['endpoint']:.4f}, avg_dist={gd['avg']:.4f}, direction={gd['direction']:.4f}")
                    else:
                        print(f"     Goal细分: endpoint={gd['endpoint']:.4f}, avg_dist={gd['avg']:.4f}")
                
                # 添加加权后的实际贡献
                print(f"     加权贡献: Obs={traj_cost.w_obs * lc['obs']:.4f}, "
                      f"Motion={traj_cost.w_motion * lc['motion']:.4f}, "
                      f"Goal={traj_cost.w_goal * lc['goal']:.4f}")
            
            print(f"\n  → 关键点统计 (Step {i}):")
            print(f"     前3个点: {preds[0, :3, :].detach().cpu().numpy()}")
            print(f"     后3个点: {preds[0, -3:, :].detach().cpu().numpy()}")
            print(f"     目标点: {goal[0].cpu().numpy()}\n")
        
        # 早停：如果预测范围过大，说明训练发散
        if abs(preds.max().item()) > 20.0 or abs(preds.min().item()) > 20.0:
            print(f"\n[WARNING] 预测范围过大 (发散)，在Step {i}处停止训练")
            print(f"当前范围: [{preds.min().item():.2f}, {preds.max().item():.2f}]")
            break

    print("[SUCCESS] 训练完成！")

    # === 5. 可视化结果 ===
    with torch.no_grad():
        preds, _, mask = model(map_tensor, goal)
        preds_2d = preds[..., 0:2]
        
        # === DEBUG: 打印网络预测的原始关键点 ===
        print("\n[DEBUG] 网络预测的原始关键点 (preds):")
        print(f"Shape: {preds.shape}")
        print(f"前5个点:\n{preds[0, :5].cpu().numpy()}")
        print(f"后5个点:\n{preds[0, -5:].cpu().numpy()}")
        print(f"统计: Min={preds.min().item():.3f}, Max={preds.max().item():.3f}, Mean={preds.mean().item():.3f}")
        
        goal_pos = goal[:, :2].unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).expand_as(preds_2d)
        preds_2d_fixed = torch.where(mask_expanded, goal_pos, preds_2d)
        
        # 生成最终轨迹 (使用三次样条)
        final_traj = traj_cost.opt.TrajGeneratorFromPFreeRot(
            preds,
            step=step_density,  # 插值步长
            mask=mask
        )
        
        # === DEBUG: 打印插值后的轨迹 ===
        print("\n[DEBUG] 插值后的轨迹 (final_traj):")
        print(f"Shape: {final_traj.shape}")
        print(f"前5个点:\n{final_traj[0, :5].cpu().numpy()}")
        print(f"后5个点:\n{final_traj[0, -5:].cpu().numpy()}")
        
    # 转换到 numpy
    traj_np = final_traj[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    cost_map_np = traj_cost.cost_map.cost_array.cpu().numpy()
    
    # 坐标转换: 物理 -> 像素 (为了在 imshow 上画图)
    # 使用 traj_cost.cost_map.Pos2Ind
    # Pos2Ind 接收 [B, N, 2] or [N, 2]
    traj_tensor_flat = final_traj[0, :, :2]
    goal_tensor_flat = goal[0, :2].unsqueeze(0)
    
    traj_px = traj_cost.cost_map.Pos2Ind(traj_tensor_flat).cpu().numpy()  # [N, 2] (Row, Col)
    goal_px = traj_cost.cost_map.Pos2Ind(goal_tensor_flat).cpu().numpy()  # [1, 2]
    
    center_r, center_c = traj_cost.cost_map.center_index
    
    
    # # === [DEBUG] 强制构造一条完美直线 ===
    # # 构造从 Start 指向 Goal 的 10 个点
    # perfect_line = torch.linspace(0, 1, 10).unsqueeze(1).repeat(1, 2).to(device) # [10, 2] (0~1)
    # # 线性插值：Start + t * (Goal - Start)
    # start_ts = torch.tensor([[0.0, 0.0]], device=device) # 假设原点
    # goal_ts = goal[:, :2] # 假设 Goal
    
    # # 简单的线性插值生成路径点
    # dummy_preds = start_ts + perfect_line.unsqueeze(0) * (goal_ts - start_ts).unsqueeze(1) # [1, 10, 2]
    
    # # 放入 TrajGenerator
    # dummy_traj = traj_cost.opt.TrajGeneratorFromPFreeRot(dummy_preds, step=0.1, start_vel=start_vel_constrained, goal_vel=goal_vel)
    
    # # 画出来
    # traj_px_dummy = traj_cost.cost_map.Pos2Ind(dummy_traj[0, :, :2]).cpu().numpy()
    
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cost_map_np, cmap='viridis', origin='upper')
    
    # === 可视化改进: 显示所有预测的关键点 ===
    preds_px = traj_cost.cost_map.Pos2Ind(preds_2d_fixed[0]).cpu().numpy()
    
    # 画出所有预测的关键点（32个点）
    plt.scatter(preds_px[:, 1], preds_px[:, 0], 
                c='yellow', s=100, edgecolors='black', zorder=10, 
                marker='o', label='Predicted Keypoints (32 points)')
    
    # 特别标注起点和终点
    plt.scatter(preds_px[0, 1], preds_px[0, 0], 
                c='lime', s=150, edgecolors='black', zorder=11, 
                marker='o', label='First Keypoint')
    plt.scatter(preds_px[-1, 1], preds_px[-1, 0], 
                c='orange', s=150, edgecolors='black', zorder=11, 
                marker='s', label='Last Keypoint')
    
    # print("\n[DEBUG] 关键路径点 (Physical坐标):")
    # print(f"  第1个点: {preds_2d_fixed[0, 0].cpu().numpy()}")
    # print(f"  第8个点: {preds_2d_fixed[0, 7].cpu().numpy()}")
    # print(f"  第16个点: {preds_2d_fixed[0, 15].cpu().numpy()}")
    # print(f"  第24个点: {preds_2d_fixed[0, 23].cpu().numpy()}")
    # print(f"  最后一个点: {preds_2d_fixed[0, -1].cpu().numpy()}")
    # print(f"  目标点: {goal[0].cpu().numpy()}")
    
    # 画白色的线用于验证坐标系
    # plt.plot(traj_px_dummy[:, 1], traj_px_dummy[:, 0], 'w--', linewidth=2, label='Perfect Line Check')
    
    # 画轨迹 (x=Col, y=Row)
    plt.plot(traj_px[:, 1], traj_px[:, 0], 'r-', linewidth=2, label='Optimized Path')
    plt.scatter(traj_px[:, 1], traj_px[:, 0], c='r', s=10)
    
    # 画起点 (中心)
    plt.plot(center_c, center_r, 'go', markersize=10, label='Start')
    
    # 画终点
    plt.plot(goal_px[0, 1], goal_px[0, 0], 'b*', markersize=15, label='Goal')
    # 画起点方向箭头
    odom_vel = odom[0, 2:4].cpu().numpy()
    odom_dir = odom_vel / (np.linalg.norm(odom_vel) + 1e-6) * 8  # 缩放箭头长度
    plt.arrow(center_c, center_r, -odom_dir[1], -odom_dir[0], color='g', head_width=2, head_length=3, length_includes_head=True)

    # 画终点方向箭头
    # goal_vel_np = goal_vel[0].cpu().numpy()
    # goal_dir = goal_vel_np / (np.linalg.norm(goal_vel_np) + 1e-6) * 8
    # plt.arrow(goal_px[0, 1], goal_px[0, 0], -goal_dir[1], -goal_dir[0], color='b', head_width=2, head_length=3, length_includes_head=True)
    plt.legend()
    plt.title(f"Navigation Result\nGoal: ({goal_np[0]:.1f}, {goal_np[1]:.1f})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_integration()