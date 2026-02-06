'''
Author: lichao951787328 951787328@qq.com
Date: 2026-02-03 13:44:15
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-02-06 15:35:19
FilePath: /viplanner/viplanner/utils/trainer_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 步骤
# 1. 创建模型保存目录
# 2. 设置模型保存路径
# 3. 加载数据，固定加载一些以及随机化加载，包括障碍物的随机生成，挡住起点与终点的中间部分，保证避障性能，随机覆盖终点，保证泛化能力，即使将终点覆盖，也能保证走向区域终点的方向，终点就在原地图的障碍里。起点与终点的距离对数据进行分类
# 4. 设置数据生成器、数据加载器、路径显示器。
# 5. 加载模型，若无则创建新模型
# 6. 设置优化器、学习率调度器
# 7. 分层训练设计，主要针对障碍物与起止点的连线关系，障碍物大小，位置等
# 8. 训练循环，包含前向传播、损失计算、反向传播、优化步骤
# 9. 定期保存模型与可视化结果

# model_file_dir = "/home/eai/VLN/viplanner/viplanner_model_myself"
# model_file_path = model_file_dir + "/test_model.pt"
# 读取数据、筛选数据，这一步是读取定死的数据
# 随机数据生成器的步骤包括：随机边缘（边缘粗糙化，随机胖瘦）、随机鬼影、噪声模糊
# 随机终点生成，按比例配备更远的终点，保证路径的多样性（）保证终点在可通行区域内。--第一步
# 障碍物随机生成（大小随机、位置随机、但不覆盖终点，膨胀算法（生成簇状、不规则物体）），保证避障能力。--第二步
# 覆盖终点的随机障碍物生成，保证泛化能力。此时不再检查终点--第三步
# 这三部分数据也是按照课程学习的方式来，
# 第一步，从开始的占比较大，后面逐步占比减少。这其中根据距离远近还有一个内部占比设计，依旧保留着近距离数据由多变少，远距离数据由少变多的设计。
# 第二步，逐步增加占比
# 第三步，逐步增加占比

# 第一步概率0.9-0.2,内部近距离概率0.9-0.2，远距离概率0.1-0.8
# 第二步概率0.1-0.4
# 第三步概率0.0-0.4
# 训练500轮，每5轮调整一次各部分概率
# 第一部分的概率每5轮降低0.007，内部近距离概率每5轮降低0.007，远距离概率每5轮增加0.007
# 第二部分的概率每5轮增加0.003
# 第三部分的概率每5轮增加0.004
# 每隔100轮记录一次模型与可视化结果  

import os
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset


class CollectData(Dataset):
    # 
    def __init__(self,
                 root_dir="/home/eai/VLN/viplanner/rotated_out/carla",
                 mode='train',               # 'train' 或 'val'
                 split_ratio=0.8,            # 80% 的底图用于训练，20% 用于验证
                 safe_dist_threshold=3.0,    # 起点安全距离阈值1
                 config=None):
        """
        Args:
            mode: 'train' -> 使用前 80% 地图，生成大量数据
                  'val'   -> 使用后 20% 地图，生成少量数据
        """
        self.mode = mode
        self.root_path = Path(root_dir)
        
        # --- 1. 配置参数 (保持原有的增强配置) ---
        self.cfg = config if config is not None else {
            "enable_map_noise": True,
            "enable_ghosting": True,
            "enable_blur": True,
            "ghost_prob": 0.8,         # 鬼影概率，某个地图增强时会有鬼影的概率
            "blur_prob": 0.8           # 模糊概率
        }

        # --- 2. 课程学习状态 ---
        # 默认从最简单的阶段开始
        self.current_epoch = 0
        self.probs = {
            'step1': 1.0, 'step2': 0.0, 'step3': 0.0, # 任务阶段概率
            'p_near': 0.9, 'p_far': 0.1               # 距离概率
        }

        self.maps = []         # 存储地图图像
        self.map_indices = []  # 存储预计算的索引

        # --- 3. 扫描并分割底图 (核心逻辑) ---
        print(f"[INFO] Scanning subdirectories in {self.root_path}...")
        subdirs = sorted(list(self.root_path.glob("sample_[0-9][0-9][0-9][0-9][0-9]")))
        
        if len(subdirs) == 0:
            raise ValueError(f"No 'sample_XXXXX' folders found in {self.root_path}")

        # **关键步骤：固定种子打乱，保证每次运行 train 和 val 分到的地图是一样的**
        # 这样不会出现某张图这次在 train，下次跑代码跑到 val 去了
        random.seed(42) 
        random.shuffle(subdirs)

        # 计算分割点
        split_point = int(len(subdirs) * split_ratio)

        # 根据模式选择底图子集
        if self.mode == 'train':
            selected_subdirs = subdirs[:split_point]
            print(f"[Dataset] Mode: TRAIN. Using {len(selected_subdirs)} maps (First {split_ratio*100}%).")
        else: # 'val' 或 'test'
            selected_subdirs = subdirs[split_point:]
            print(f"[Dataset] Mode: VAL. Using {len(selected_subdirs)} maps (Last {(1-split_ratio)*100:.1f}%).")

        # --- 4. 加载地图与预计算 ---
        valid_count = 0
        for subdir in tqdm(selected_subdirs, desc=f"Loading Maps ({self.mode})"):
            npy_path = subdir / "sem_mask.npy"
            if not npy_path.exists(): continue
                
            try:
                # 加载并检查
                raw_mask = np.load(npy_path)
                if len(raw_mask.shape) > 2: raw_mask = raw_mask.squeeze()
                
                h, w = raw_mask.shape
                cx, cy = w // 2, h // 2
                
                if raw_mask[cy, cx] == 0: continue # 中心障碍，跳过

                mask_uint8 = raw_mask.astype(np.uint8)
                dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
                
                # 保证终点安全栅格数量
                robot_radius = 5.0
                
                if dist_map[cy, cx] >= safe_dist_threshold:
                    self.maps.append(mask_uint8)
                    
                    # --- 预计算 Near/Far 索引 (优化速度) ---
                    # 只需要算距离，不再算角度
                    y_coords, x_coords = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                    r_limit = 0.3 * w
                    
                    is_free = (mask_uint8 == 1) & (dist_map >= robot_radius)
                    
                    self.map_indices.append({
                        'near': np.argwhere(is_free & (dist_from_center <= r_limit)),
                        'far':  np.argwhere(is_free & (dist_from_center > r_limit)),
                        'center': (cx, cy)
                    })
                    valid_count += 1
            except Exception as e:
                print(f"[WARN] Failed to load {npy_path}: {e}")

        if valid_count == 0:
            raise RuntimeError(f"No valid maps found for mode {self.mode}!")

    def __len__(self):
        """
        控制生成数据的数量
        """
        if self.mode == 'train':
            # 训练集：每张底图重复利用 20 次 (随机生成不同障碍和终点)
            # 这样一个 epoch 会比较长，数据覆盖面广
            return len(self.maps) * 6
        else:
            # 验证集：每张底图只跑 1 次，或者固定数量
            # 数量少，只为了快速看指标
            return int(len(self.maps))  # 或者直接 return len(self.maps)

    def __getitem__(self, index):
        # 1. 基础数据获取
        # 验证集如果想要固定结果，可以在这里 seed(index)，但通常为了测试泛化，也可以保持随机
        if self.mode == 'val':
            # 这样验证集每次 getitem 生成的障碍物是一样的，方便对比模型优劣
            np.random.seed(index)
            
        map_idx = index % len(self.maps)
        base_map = self.maps[map_idx].copy()
        map_meta = self.map_indices[map_idx]
        
        # 2. 确定起点 (中心)
        h, w = base_map.shape
        start_pos = (w // 2, h // 2)

        # 3. 采样终点 (使用之前优化过的逻辑)
        # 注意，此处返回的是像素/栅格坐标，不是实际的米坐标
        goal_pos = self._sample_goal_optimized(map_meta)
        
        # 4. 课程学习：生成障碍物
        # 这里的逻辑和训练一模一样，只是验证集数量少
        rand_step = np.random.rand()
        final_map = base_map.copy()

        if rand_step < self.probs['step1']:
            pass
        elif rand_step < self.probs['step1'] + self.probs['step2']:
            final_map = self._add_obstacles(base_map, start_pos, goal_pos, density='low', targeted=False)
        else:
            final_map = self._add_obstacles(base_map, start_pos, goal_pos, density='high', targeted=True)

        # 5. 图像增强 (通常验证集不加噪声，或者加少量噪声，取决于你想测什么)
        # 如果你想测“模型在真实噪声下的表现”，则 mode=='val' 也可以为 True
        # 这里默认：训练加噪，验证保持干净以便观察路径规划逻辑
        if self.mode == 'train':
            final_map = self._augment_map(final_map)

        # 6. 格式转换
        # map_tensor = torch.from_numpy(final_map).unsqueeze(0)
        map_tensor = torch.from_numpy(final_map).float().unsqueeze(0)
        goal_tensor = torch.tensor(goal_pos, dtype=torch.float32)

        # 连通性检查失败回退
        if not self._is_reachable(final_map, start_pos, goal_pos):
            final_map = base_map
            map_tensor = torch.from_numpy(final_map).float().unsqueeze(0)
        return map_tensor, goal_tensor

    # ---------------- 以下辅助函数保持不变 ----------------
    def update_curriculum(self, epoch, new_probs):
        self.current_epoch = epoch
        self.probs = new_probs

    def _sample_goal_optimized(self, map_meta):
        # 简单的 Near/Far 采样，不涉及 FOV
        p_far = self.probs.get('p_far', 0.1)
        
        target_pool = map_meta['near']
        # 如果随机到远距离，且远距离有空地
        if len(map_meta['far']) > 0 and np.random.rand() < p_far:
            target_pool = map_meta['far']
        
        if len(target_pool) == 0:
            cx, cy = map_meta['center']
            return (cx), (cy)

        idx = np.random.randint(len(target_pool))
        goal_y, goal_x = target_pool[idx]
        return (goal_x), (goal_y)

    # 输出依然是 0-1 二值图，1 表示可通行
    def _add_obstacles(self, map_data, start, goal, density='low', targeted=False):
        h, w = map_data.shape
        temp_map = map_data.copy()
        
        # 1. 确定目标数量
        if density == 'low':
            target_num_obs = np.random.randint(5, 10)
        else:
            target_num_obs = np.random.randint(20, 30)

        added_count = 0       # 已成功添加的数量
        attempts = 0          # 当前尝试次数
        max_attempts = target_num_obs * 2.5  # 设定最大重试次数 (比如目标数量的5倍)

        # 2. 改为 while 循环
        while added_count < target_num_obs * 0.6 and attempts < max_attempts:
            attempts += 1
            
            # 3. 决定是否进行针对性阻断 (逻辑改为依赖 added_count)
            # 意思：前 50% 成功的障碍物，要是阻断型的
            is_blocking = targeted and (added_count < target_num_obs * 0.5)
            
            if is_blocking:
                t = np.random.uniform(0.2, 0.8) # 0.2-0.8 避开两端，减少碰撞概率
                center_x = start[0] + t * (goal[0] - start[0])
                center_y = start[1] + t * (goal[1] - start[1])
                jitter = min(w, h) * 0.1
                cx = int(center_x + np.random.uniform(-jitter, jitter))
                cy = int(center_y + np.random.uniform(-jitter, jitter))
                obs_type = np.random.choice(['rect', 'line'])
            else:
                cx = np.random.randint(0, w)
                cy = np.random.randint(0, h)
                obs_type = np.random.choice(['circle', 'rect', 'line'])

            # --- 绘制障碍物掩码 (逻辑不变) ---
            obs_mask = np.zeros_like(temp_map)
            
            if obs_type == 'circle':
                r = np.random.randint(5, 15)
                cv2.circle(obs_mask, (cx, cy), r, 1, -1)
            elif obs_type == 'rect':
                scale = 1.5 if is_blocking else 1.0
                rw = int(np.random.randint(8, 20) * scale)
                rh = int(np.random.randint(8, 20) * scale)
                top_left = (max(0, cx - rw//2), max(0, cy - rh//2))
                bottom_right = (min(w, cx + rw//2), min(h, cy + rh//2))
                cv2.rectangle(obs_mask, top_left, bottom_right, 1, -1)
            elif obs_type == 'line':
                if is_blocking:
                    self._draw_perpendicular_wall(obs_mask, start, goal, (cx, cy), w, h)
                else:
                    ex, ey = np.random.randint(0, w), np.random.randint(0, h)
                    cv2.line(obs_mask, (cx, cy), (ex, ey), 1, thickness=5)

            # 转换：obs_mask (1=Obs) -> current_obs_layer (0=Obs)
            current_obs_layer = 1 - obs_mask 

            # --- 关键修改 ---
            # 如果碰撞了，直接进入下一次循环 (不增加 added_count)
            if self._check_collision(current_obs_layer, start, goal):
                continue
            
            # 如果没碰撞，才合并进去，并增加计数
            temp_map = np.minimum(temp_map, current_obs_layer)
            added_count += 1
            
        return temp_map
    
    def _check_collision(self, new_obs_mask, start, goal):
        """
        利用掩码相交法检查碰撞。
        Args:
            new_obs_mask: 当前生成的新障碍物掩码 (要求: 1=障碍, 0=空地)
            start: (x, y)
            goal: (x, y)
        """
        # 1. 创建一张全黑的安全区域图
        safety_mask = np.zeros_like(new_obs_mask)
        
        # 2. 定义膨胀半径 (安全距离)
        radius = 6 
        
        # 3. 在图上把起点和终点画成实心圆 (值为1)
        # 注意要转成 int 坐标
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        
        cv2.circle(safety_mask, (sx, sy), radius, 1, -1)
        cv2.circle(safety_mask, (gx, gy), radius, 1, -1)
        
        # 4. 核心逻辑：检查两个掩码是否有重叠
        # new_obs_mask == 1 的地方是障碍
        # safety_mask  == 1 的地方是不能碰的禁区
        # 只要这俩图相乘(或逻辑与)的结果里有 1，就说明撞了
        
        # 方法 A:使用 numpy 逻辑与 (推荐，代码短)
        is_overlap = np.any((new_obs_mask == 0) & (safety_mask == 1))
        return is_overlap
        
    def _is_reachable(self, map_data, start, goal):
        h, w = map_data.shape
        
        # 1. 制作基础图：0=障碍(黑), 255=路(白)
        check_map = (map_data * 255).astype(np.uint8) 
        
        # --- 新增步骤：障碍物膨胀 (即：自由区域腐蚀) ---
        # 目的：确保路径有一定宽度，不是紧贴墙壁的
        # 膨胀半径 3 像素 -> 核大小 = 2*3 + 1 = 7
        kernel_size = 7 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 对白色区域(路)进行腐蚀，黑色区域(障碍)就会变大
        check_map = cv2.erode(check_map, kernel)
        
        # -------------------------------------------
        
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        
        # 边界检查 (防止坐标越界)
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return False

        # 如果膨胀后，起点或终点被障碍物吞噬了，说明起/终点离墙太近，不可达
        if check_map[sy, sx] == 0 or check_map[gy, gx] == 0: 
            return False

        # 准备 FloodFill
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # 开始漫水填充，填充值为 128
        cv2.floodFill(check_map, mask, (sx, sy), 128)
        
        # 检查终点是否被染成了 128
        return check_map[gy, gx] == 128

    def _draw_perpendicular_wall(self, mask, start, goal, center, w, h):
        cx, cy = center
        # 1. 计算路径向量
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # 2. 计算垂直向量 (-dy, dx) 并归一化
        length = np.sqrt(dx*dx + dy*dy) + 1e-6
        perp_dx = -dy / length
        perp_dy = dx / length
        
        # 3. 随机墙长度
        wall_len = np.random.randint(20, 50)
        
        # 4. 计算墙的两个端点
        p1 = (int(cx + perp_dx * wall_len / 2), int(cy + perp_dy * wall_len / 2))
        p2 = (int(cx - perp_dx * wall_len / 2), int(cy - perp_dy * wall_len / 2))
        
        # 5. 绘制 (在 mask 上画 1)
        cv2.line(mask, p1, p2, 1, thickness=4)

    # --- 修复后的增强函数 ---
    def _apply_roughness_and_morph(self, image, roughness_prob, morph_prob):
        res = image.copy()
        if random.random() < roughness_prob:
            k = random.choice([3, 5])
            res = cv2.GaussianBlur(res, (k, k), 0)
            noise = np.random.randn(*res.shape).astype(np.float32) * 0.1
            res = (res + noise > 0.5).astype(np.float32)
        
        if random.random() < morph_prob:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if random.random() > 0.5:
                res = cv2.dilate(res, kernel)
            else:
                res = cv2.erode(res, kernel)
        return res

    def _apply_ghosting(self, image, ghost_prob):
        if random.random() > ghost_prob:
            return image
        h, w = image.shape
        res = image.copy()
        for _ in range(random.randint(1, 5)):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            # 在空地 (值接近1) 上加噪点 (变成0，即障碍)
            # 原代码逻辑：if image < 0.1 (障碍上) 加障碍？ 
            # 通常 Ghosting 是在空地上出现假障碍。
            # 这里假设：地图 1=Free, 0=Obs.
            # 我们想在 Free 区域加 Obs (0)。
            if image[cy, cx] > 0.9: # 如果是空地
                # 画黑色圆 (0) 代表障碍
                # 注意：OpenCV circle color 应该是 0
                cv2.circle(res, (cx, cy), random.randint(1, 3), 0.0, -1)
        return res

    def _apply_blur(self, image, blur_prob):
        """
        新增：模拟概率栅格地图的模糊效果
        """
        if random.random() > blur_prob:
            return image
        
        # 使用高斯模糊让 0/1 边界变得平滑 (0.0~1.0)
        k = random.choice([3, 5, 7])
        res = cv2.GaussianBlur(image, (k, k), 0)
        
        # 这里不需要二值化，保留灰度值模拟概率
        return res

    def _augment_map(self, grid_map: np.ndarray) -> np.ndarray:
        # 这里的 input grid_map 已经是 0(obs)~1(free) 的 float32 数据
        augmented_map = grid_map.copy()
        
        # 使用 self.cfg 替代 self._cfg
        if self.cfg.get("enable_map_noise", False):
            augmented_map = self._apply_roughness_and_morph(
                augmented_map,
                roughness_prob=0.5,
                morph_prob=0.5
            )

        if self.cfg.get("enable_ghosting", False):
            augmented_map = self._apply_ghosting(
                augmented_map, 
                ghost_prob=self.cfg.get("ghost_prob", 0.3)
            )

        if self.cfg.get("enable_blur", False):
            augmented_map = self._apply_blur(
                augmented_map, 
                blur_prob=self.cfg.get("blur_prob", 0.3)
            )

        augmented_map = np.clip(augmented_map, 0.0, 1.0)
        return (augmented_map > 0.5).astype(np.float32)


# import shutil
# import matplotlib.pyplot as plt    
from torch.utils.data import DataLoader


def test_standard_dataloader_flow():
    # 1. 配置路径
    real_data_root = "/home/eai/VLN/viplanner/rotated_out/carla"
    output_dir = "./output_dataloader_test"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化 Dataset
    print("[1/4] Initializing Dataset...")
    dataset = CollectData(
        root_dir=real_data_root,
        mode='train',          
        split_ratio=1.0,       
        safe_dist_threshold=2.0, 
        config={
            "enable_map_noise": True,
            "enable_ghosting": True, 
            "ghost_prob": 0.5,
            "blur_prob": 0.5
        }
    )
    
    # 强制设置高难度，方便观察
    dataset.probs = {'step1': 0.0, 'step2': 0.0, 'step3': 1.0, 'p_near': 0.2, 'p_far': 0.8}

    # --- 关键点：这里体现了 __len__ 的作用 ---
    # 假设你有 100 张底图，train模式下 __len__ 返回 100 * 10 = 1000
    total_len = len(dataset)
    print(f"[2/4] Dataset declared length: {total_len} samples.")

    # 3. 初始化 DataLoader (这是PyTorch的标准加载器)
    # batch_size=1: 每次取1张图
    # shuffle=True: 打乱顺序 (0 ~ total_len-1 的索引被打乱)
    # num_workers=0: 单进程，方便调试
    print("[3/4] Initializing DataLoader...")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    print(f"[4/4] Starting Simulation Loop (Taking first 10 batches)...")
    
    # --- 4. 模拟训练循环 (Step 8 in your comments) ---
    # 这里 enumerate 会自动调用 __len__ 确定边界，并自动调用 __getitem__
    for i, (map_tensor, goal_tensor) in enumerate(train_loader):
        
        # 为了测试，我们只看前 10 个 batch 就停
        if i >= 40:
            print("--- Reached 10 samples, stopping test ---")
            break

        # --- 以下是可视化逻辑 ---
        
        # map_tensor 来自 DataLoader，会多一个 Batch 维度
        # Shape: [1, 1, 80, 80] -> squeeze -> [80, 80]
        grid_map = map_tensor.squeeze().numpy()
        
        # goal_tensor Shape: [1, 2] -> squeeze -> [2] (x, y)
        goal_x, goal_y = goal_tensor.squeeze().numpy()

        # 还原图片 (0-255)
        img_display = (grid_map * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        
        # 放大 5 倍方便查看
        scale = 5
        h, w = grid_map.shape
        img_large = cv2.resize(img_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        # 绘制起点 (中心)
        start_x, start_y = w // 2, h // 2
        cv2.circle(img_large, (start_x * scale, start_y * scale), 2 * scale, (0, 255, 0), -1) 
        
        # 绘制终点
        cv2.circle(img_large, (int(goal_x * scale), int(goal_y * scale)), 2 * scale, (0, 0, 255), -1)
        
        # 绘制连线
        cv2.line(img_large, (start_x * scale, start_y * scale), (int(goal_x * scale), int(goal_y * scale)), (255, 0, 0), 1)

        filename = f"{output_dir}/loader_sample_{i:02d}.png"
        cv2.imwrite(filename, img_large)
        print(f"Batch {i}: Saved {filename}")

if __name__ == "__main__":
    test_standard_dataloader_flow()