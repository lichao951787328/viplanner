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
    from plannernet.autoencoder_myself import AutoEncoderGrid
    from traj_cost_opt.traj_cost_myself import TrajCost
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


# def find_valid_goal(traj_cost, min_dist=2.0, max_dist=4.0):
#     """
#     在 CostMap 中寻找一个低 Cost 的点作为 Goal
#     """
#     cost_map = traj_cost.cost_map.cost_array.cpu().numpy() # [H, W]
#     resolution = traj_cost.cost_map.cfg.general.resolution
#     c_row, c_col = traj_cost.cost_map.center_index
    
#     H, W = cost_map.shape
    
#     # 随机采样寻找
#     for _ in range(1000):
#         # 随机生成物理坐标 (局部系)
#         # 假设地图范围大概是 [-4, 4] 米
#         rand_x = np.random.uniform(-max_dist, max_dist)
#         rand_y = np.random.uniform(-max_dist, max_dist)
        
#         # 距离过滤
#         dist = np.sqrt(rand_x**2 + rand_y**2)
#         if dist < min_dist or dist > max_dist:
#             continue
            
#         # 物理坐标 -> 像素坐标
#         # Pos2Ind 逻辑: row = c_row - (x / res)
#         # 注意 TrajCost 里 Pos2Ind 的输入是 (x, y)
#         r_idx = int(c_row - (rand_x / resolution))
#         c_idx = int(c_col - (rand_y / resolution))
        
#         # 边界检查
#         if 0 <= r_idx < H and 0 <= c_idx < W:
#             # 检查 Cost (要求是安全区域)
#             if cost_map[r_idx, c_idx] < 0.02: # 假设阈值 0.1
#                 print(f"[INFO] 找到有效终点: ({rand_x:.2f}, {rand_y:.2f}), Cost={cost_map[r_idx, c_idx]:.2f}")
#                 return torch.tensor([[rand_x, rand_y, 0.0]], device=traj_cost.device)
    
#     print("[WARN] 未找到合适终点，使用默认值")
#     return torch.tensor([[3.0, 0.0, 0.0]], device=traj_cost.device)

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
    traj_cost = TrajCost(
        gpu_id=0 if torch.cuda.is_available() else "cpu",
        log_data=False,
        w_obs=0.1, 
        w_motion=5.0, 
        w_goal=1.0,
        obstalce_thread=0.5
    )
    # 加载地图 (SimpleCostMapWrapper)
    # 这里 root_path 传当前目录 ".", map_name 传文件名
    traj_cost.SetMap(".", NPY_MAP_NAME)
    
    # 3. 初始化模型
    # 假设地图分辨率 0.1, 80x80 -> 8m x 8m, max_dist 设为 5.0m
    model = AutoEncoderGrid(encoder_channel=64, max_dist=5.0, step_size=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"traj_cost.cost_map.cost_array.shape: {traj_cost.cost_map.cost_array.shape}")
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
    
    # 目标: 自动寻找
    # goal = find_valid_goal(traj_cost).to(device) # [1, 3]
    # goal_vel = torch.zeros((1, 2), dtype=torch.float32, device=device)
    # 随机生成一个终点速度 (vx, vy), 范围 [-1.0, 1.0] m/s
    # if torch.isnan(goal).any():
    #     print("[Error] Generated Goal contains NaN!")
    #     return # 跳过这一
    
    goal = torch.tensor([[-3.5, 1.0, 0.0]], dtype=torch.float32, device=device)
    goal_vel = torch.tensor([[0.1, 0.0]], dtype=torch.float32, device=device)
    goal_np = goal[0, :2].cpu().numpy()
    
    print(f"[INFO] 随机生成终点速度: vx={goal_vel[0,0]:.2f}, vy={goal_vel[0,1]:.2f}")
    
    # Odom: 假设在原点, 且由速度
    # [x, y, vx, vy] -> [0, 0, 1.0, 0.0] (假设初始向前运动)
    odom = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    
    # === 可视化 CostMap、起点、终点及其方向 ===
    import matplotlib.patches as patches

    cost_map_np = traj_cost.cost_map.cost_array.cpu().numpy()
    center_r, center_c = traj_cost.cost_map.center_index

    # 起点像素坐标
    start_px = np.array([center_r, center_c])
    # 起点方向（速度方向）
    odom_vel = odom[0, 2:4].cpu().numpy()
    odom_dir = odom_vel / (np.linalg.norm(odom_vel) + 1e-6) * 8  # 缩放箭头长度

    # 终点像素坐标
    goal_tensor_flat = goal[0, :2].unsqueeze(0)
    goal_px = traj_cost.cost_map.Pos2Ind(goal_tensor_flat).cpu().numpy()[0]
    # 终点方向（目标速度方向）
    goal_vel_np = goal_vel[0].cpu().numpy()
    goal_dir = goal_vel_np / (np.linalg.norm(goal_vel_np) + 1e-6) * 8

    plt.figure(figsize=(7, 7))
    plt.imshow(cost_map_np, cmap='viridis', origin='upper')
    plt.plot(center_c, center_r, 'go', markersize=10, label='Start')
    plt.plot(goal_px[1], goal_px[0], 'b*', markersize=15, label='Goal')

    # 起点方向箭头
    plt.arrow(center_c, center_r, -odom_dir[1], -odom_dir[0], color='g', head_width=2, head_length=3, length_includes_head=True)
    # 终点方向箭头
    plt.arrow(goal_px[1], goal_px[0], -goal_dir[1], -goal_dir[0], color='b', head_width=2, head_length=3, length_includes_head=True)

    plt.legend()
    plt.title("CostMap with Start/Goal and Directions")
    plt.tight_layout()
    plt.show()
    
    # Fear: 模拟一个 Fear 输入 (或者由网络预测，但在 Cost 计算中它作为 Label 或 Input)
    # 这里 CostofTraj 接收 fear，实际上是用于计算 BCE Loss
    # 我们先给一个 Dummy Fear Prediction (假设网络认为目前很安全)
    dummy_fear_pred = torch.tensor([[0.2]], device=device, requires_grad=True)

    print("\n[INFO] 开始 Overfit 训练 (50 Steps)...")
    model.train()
    
    losses = []
    
    for i in range(101):
        optimizer.zero_grad()
        
        # 1. 网络推理
        # preds: [B, K, 3] (关键点)
        # c: [B, 1] (预测的 Fear)
        # mask: [B, K]
        start_vel = odom[:, 2:4]
        vel_norm = torch.norm(start_vel, dim=1, keepdim=True) + 1e-6
        start_vel_direction = start_vel / vel_norm
        start_vel_constrained = start_vel_direction * 0.2
        
        preds, pred_fear, mask = model(map_tensor, goal)
        preds_2d = preds[..., 0:2]  # Shape: [B, K, 2]
        # 2. 生成密集轨迹 (Trajectory Generation)
        # 使用 TrajOpt (三次样条)
        # 注意: TrajCost.opt 就是 TrajOpt 实例
        # step=0.1 (生成更密集的点用于 Cost 计算)
        
        goal_pos = goal[:, :2].unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).expand_as(preds_2d)
        preds_2d = torch.where(mask_expanded, goal_pos, preds_2d)
        step_density = 0.2
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(
            preds_2d,
            step=0.2,  # 插值步长
            start_vel=start_vel_constrained,  # 考虑初始速度
            goal_vel=None
        )
        
        # 2. 获取实际生成的长度
        B, N, D = waypoints.shape

        # 3. 直接根据长度生成 xs，确保一一对应
        # 假设 xs 代表路径长度索引，步长为 step_density
        # 注意：要确保 device 一致
        xs = torch.arange(0, N, device=device, dtype=waypoints.dtype).unsqueeze(0) * step_density 
        # xs shape: [1, N]

        # 4. 这里的 valid_t 计算逻辑保持不变
        valid_t = (~mask).sum(dim=1).float() # [B]
            
        # C. 生成密集 Mask
        # 如果时间 t > valid_t，则为无效 (True)
        dense_mask = xs > valid_t.unsqueeze(1) # [B, N_dense]
        
        # 3. 计算 Cost (作为 Loss)
        # 注意: CostofTraj 内部会计算 Collision, Goal, Motion Loss
        # 并且会计算 Fear Label 并与 pred_fear 做 BCE Loss
        # 取 goal 的前两个，以及 goal_vel，组成一个四维 goal_new
        goal_new = torch.cat([goal[:, :2], goal_vel], dim=1)  # [B, 4]
        
        # B, K, _ = preds_2d.shape
        
        # # 生成 0 ~ 1 的系数 [B, K]
        # t_steps = torch.linspace(0, 1, K, device=device).unsqueeze(0).repeat(B, 1)
        
        # # 起点 (0,0)
        # start_pos = torch.zeros((B, 1, 2), device=device)
        # # 终点
        # end_pos = goal[:, :2].unsqueeze(1) # [B, 1, 2]
        
        # # 理想的直线点位置: Start + t * (End - Start)
        # # [B, K, 2]
        # perfect_points = start_pos + t_steps.unsqueeze(-1) * (end_pos - start_pos)
        
        # # 2. 计算当前预测点与理想点的距离 (L2 Loss)
        # # 注意：只计算有效点 (mask 为 False 的部分)
        # # 但因为我们也把无效点填成 Goal 了，而 perfect_points 的尾部也是 Goal，
        # # 所以直接计算所有点的距离通常也没问题，或者用 mask 过滤更严谨。
        
        # ref_loss = torch.nn.functional.mse_loss(preds_2d, perfect_points)
        
        # # ... (Loss 加权) ...
        
        # # 刚开始训练时，给 ref_loss 一个巨大的权重，强迫它变直
        # w_ref = 10.0
        
        loss = traj_cost.CostofTraj(
            waypoints=waypoints,
            odom=odom,
            goal=goal_new,
            fear=pred_fear, # 传入网络预测的 Fear
            log_step=i,
            ahead_dist=2.0,
            mask=dense_mask
        )
        # + w_ref * ref_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f"Step {i}: Total Loss = {loss.item():.4f}")

    print("[SUCCESS] 训练完成！")

    # === 5. 可视化结果 ===
    with torch.no_grad():
        preds, _, _ = model(map_tensor, goal)
        preds_2d = preds[..., 0:2]
        
        goal_pos = goal[:, :2].unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).expand_as(preds_2d)
        preds_2d_fixed = torch.where(mask_expanded, goal_pos, preds_2d)
        
        # 生成最终轨迹
        final_traj = traj_cost.opt.TrajGeneratorFromPFreeRot(preds_2d_fixed, step=0.1, start_vel=odom[:, 2:4], goal_vel=None)
        
    # 转换到 numpy
    traj_np = final_traj[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    cost_map_np = traj_cost.cost_map.cost_array.cpu().numpy()
    
    # 坐标转换: 物理 -> 像素 (为了在 imshow 上画图)
    # 使用 traj_cost.cost_map.Pos2Ind
    # Pos2Ind 接收 [B, N, 2] or [N, 2]
    traj_tensor_flat = final_traj[0, :, :2]
    goal_tensor_flat = goal[0, :2].unsqueeze(0)
    
    traj_px = traj_cost.cost_map.Pos2Ind(traj_tensor_flat).cpu().numpy() # [N, 2] (Row, Col)
    goal_px = traj_cost.cost_map.Pos2Ind(goal_tensor_flat).cpu().numpy() # [1, 2]
    
    center_r, center_c = traj_cost.cost_map.center_index
    
    
    # === [DEBUG] 强制构造一条完美直线 ===
    # 构造从 Start 指向 Goal 的 10 个点
    perfect_line = torch.linspace(0, 1, 10).unsqueeze(1).repeat(1, 2).to(device) # [10, 2] (0~1)
    # 线性插值：Start + t * (Goal - Start)
    start_ts = torch.tensor([[0.0, 0.0]], device=device) # 假设原点
    goal_ts = goal[:, :2] # 假设 Goal
    
    # 简单的线性插值生成路径点
    dummy_preds = start_ts + perfect_line.unsqueeze(0) * (goal_ts - start_ts).unsqueeze(1) # [1, 10, 2]
    
    # 放入 TrajGenerator
    dummy_traj = traj_cost.opt.TrajGeneratorFromPFreeRot(dummy_preds, step=0.1, start_vel=start_vel_constrained, goal_vel=goal_vel)
    
    # 画出来
    traj_px_dummy = traj_cost.cost_map.Pos2Ind(dummy_traj[0, :, :2]).cpu().numpy()
    
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cost_map_np, cmap='viridis', origin='upper')
    
    preds_px = traj_cost.cost_map.Pos2Ind(preds_2d_fixed[0]).cpu().numpy()
    plt.scatter(preds_px[:, 1], preds_px[:, 0], c='yellow', s=50, edgecolors='black', zorder=10, label='Raw Keypoints')
    print("\n[DEBUG] Raw Keypoints (Physical):")
    print(preds_2d_fixed[0, :5].cpu().numpy())
    print("...")
    print(preds_2d_fixed[0, -5:].cpu().numpy())
    
    # 画白色的线用于验证坐标系
    plt.plot(traj_px_dummy[:, 1], traj_px_dummy[:, 0], 'w--', linewidth=2, label='Perfect Line Check')
    
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
    goal_vel_np = goal_vel[0].cpu().numpy()
    goal_dir = goal_vel_np / (np.linalg.norm(goal_vel_np) + 1e-6) * 8
    plt.arrow(goal_px[0, 1], goal_px[0, 0], -goal_dir[1], -goal_dir[0], color='b', head_width=2, head_length=3, length_includes_head=True)
    plt.legend()
    plt.title(f"Navigation Result\nGoal: ({goal_np[0]:.1f}, {goal_np[1]:.1f})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_integration()