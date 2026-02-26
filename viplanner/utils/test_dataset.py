import os
import glob
import numpy as np
import torch
import pypose as pp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 假设你的 Dataset 文件名为 dataset_myself_.py
from dataset_myself_ import PlannerData

# --- 1. 模拟配置 ---
class MockDataCfg:
    def __init__(self):
        self.real_world_data = False
        self.map_resolution = 0.1  # 假设 0.1m/pixel
        self.local_map_size_pixels = (80, 80)  # [请修改] 你的NPY图片实际尺寸
        self.local_map_size = 8.0
        
        # 增强配置
        self.enable_map_noise = True
        self.enable_ghosting = True
        self.enable_blur = True
        self.ghost_prob = 0.75
        self.blur_prob = 0.75
        self.dropout_prob = 0.0  # 既然是局部图，可能不需要再模拟dropout了

# --- 2. 姿态解析函数 ---
def parse_camera_pose(txt_path):
    """
    解析 /home/.../camera_pose.txt
    格式: 6行 float (x, y, z, yaw, pitch, roll) 单位: deg
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    # 过滤注释行，提取数字
    values = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
            
    if len(values) < 6:
        print(f"Warning: {txt_path} data incomplete.")
        return pp.identity_se3(1)

    x, y, z = values[0], values[1], values[2]
    yaw_deg, pitch_deg, roll_deg = values[3], values[4], values[5]
    
    # --- 角度转旋转矩阵/四元数 ---
    # 通常顺序是 Z-Y-X (Yaw, Pitch, Roll)
    # 输入 pypose 需要弧度
    euler = torch.tensor([
        np.deg2rad(yaw_deg), 
        np.deg2rad(pitch_deg), 
        np.deg2rad(roll_deg)
    ])
    
    # 创建 pypose SE3 对象
    # translation: [x, y, z]
    # rotation: 使用 euler2SO3 转换
    trans = torch.tensor([x, y, z])
    rot = pp.euler2SO3(euler) # 默认顺序是 ZYX
    
    # 组合成 SE3: [x, y, z, qx, qy, qz, qw]
    pose = pp.SE3(torch.cat([trans, rot.tensor()], dim=0)).unsqueeze(0)
    return pose

# --- 3. 主测试逻辑 ---
def main():
    # 根目录
    base_dir = "/home/lichao/viplanner/rotated_out/carla"
    
    # 搜索所有 sample 文件夹
    sample_dirs = sorted(glob.glob(os.path.join(base_dir, "sample_*")))
    
    map_files = []
    odom_list = []
    
    print("Loading data metadata...")
    for s_dir in sample_dirs:
        npy_path = os.path.join(s_dir, "sem_mask.npy")
        pose_path = os.path.join(s_dir, "camera_pose.txt")
        
        if os.path.exists(npy_path) and os.path.exists(pose_path):
            map_files.append(npy_path)
            # 解析 Pose
            pose = parse_camera_pose(pose_path)
            odom_list.append(pose)
            
    if len(map_files) == 0:
        print("No valid data found!")
        return
    
    # 拼接 Odom
    odom_data = torch.cat(odom_list, dim=0)  # (N, 7)
    
    # 伪造 Goal (因为你没提供 goal 文件)
    # 假设目标都在机器人前方 2 米处 (Local Frame: x=2, y=0)
    goal_data = torch.tensor([[2.0, 0.0, 0.0]]).repeat(len(map_files), 1)
    
    # Augment flag
    pair_augment = np.zeros(len(map_files), dtype=bool)
    
    print(f"Loaded {len(map_files)} samples.")
    
    # 初始化 Dataset
    cfg = MockDataCfg()
    dataset = PlannerData(cfg, transform=None)
    dataset.update_buffers(map_files, odom_data, goal_data, pair_augment)
    
    # 随机取样测试
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        data = dataset[idx]
        
        img_tensor = data[0]  # (1, H, W)
        img_np = img_tensor.squeeze().numpy()
        start_pose = data[2]  # Odom
        goal_local = data[3]
        
        print(f"Sample {idx}:")
        print(f"  Odom (World): {start_pose.translation().tolist()}")
        print(f"  Goal (Local): {goal_local.tolist()}")
        
        # 画图
        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(img_np, cmap='gray', origin='upper')  # 假设 .npy 存储时 origin 在上
        
        # 画中心点(机器人)
        h, w = img_np.shape
        cy, cx = h // 2, w // 2
        ax.plot(cx, cy, 'ro', label='Map Center')
        
        # start_local_x, start_local_y = data
        # start_py = cy - (start_local_x / res)
        # start_px = cx - (start_local_y / res)
        # ax.plot(start_px, start_py, 'go', label='Jittered Start')
        
        # 画目标点 (将 Local Goal 投影回像素)
        # x = forward (up), y = left (left) -> 对应 dataset 里的逆运算
        res = cfg.map_resolution
        gx_local, gy_local = goal_local[0], goal_local[1]
        
        # 逆变换: pixel = center - local / res
        # 注意方向：_pixel_to_local_se3 里定义了 local_x = (center_y - py) * res
        # 所以 py = center_y - local_x / res
        target_py = cy - (gx_local / res)
        target_px = cx - (gy_local / res)
        
        ax.plot(target_px, target_py, 'bx', markersize=10, label='Gen Goal')
        sample_dir = sample_dirs[idx]
        sample_name = os.path.basename(sample_dir)
        ax.set_title(f"{sample_name} (idx={idx})")
        ax.legend()
        
    plt.show()

if __name__ == "__main__":
    main()