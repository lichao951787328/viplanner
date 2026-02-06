import torch
import torch.nn as nn
import numpy as np
import math
from skimage import graph # 用于计算最短路

# 假设 PlannerNetGrid 已经导入
try:
    from plannernet.PlannerNet_myself_cubic import PlannerNetGrid
except ImportError:
    pass


class AutoEncoderGrid(nn.Module):
    def __init__(self, encoder_channel=64, max_dist=8.0, step_size=0.5):
        super().__init__()
        
        # 1. 地图参数 (你需要根据实际情况调整)
        # 假设地图是 80x80，覆盖范围是 20米 x 20米 (即中心向各方向延伸10米)
        # 那么分辨率 = 20 / 80 = 0.25 米/像素
        self.map_size = 80
        self.map_res = 0.1
        self.max_dist = max_dist

        # 2. Encoder
        self.encoder = PlannerNetGrid(in_channels=1, small_input=True)
        
        # 3. Decoder
        self.decoder = DecoderGridDynamic(
            in_channels=512,
            goal_channels=encoder_channel,
            input_feature_size=10,
            max_dist=max_dist,
            step_size=step_size
        )

    def forward(self, x: torch.Tensor, goal_norm: torch.Tensor):
        """
        x: [Batch, 1, 80, 80] - 占用地图/Costmap (假设 >0.5 为障碍)
        goal_norm: [Batch, 2] - 目标点 (米), 坐标系需与地图一致
        """
        goal_meters = goal_norm * self.max_dist
        # --- 步骤 1: 在线计算实际路径长度 (CPU操作) ---
        # 这一步只在推理时并没有梯度，不会影响反向传播
        with torch.no_grad():
            real_dist = self._compute_geodesic_distance(x, goal_meters)
        
        # --- 步骤 2: 正常的网络推理 ---
        # Encoder
        features = self.encoder(x)
        
        # Decoder (传入计算好的 real_dist)
        out_path, cost, mask = self.decoder(features, goal_norm, real_dist)
        
        return out_path, cost, mask

    def _compute_geodesic_distance(self, map_tensor, goals):
        """
        计算从地图中心到目标的实际避障距离
        """
        batch_size = map_tensor.shape[0]
        distances = []
        
        # 转 CPU numpy
        maps_np = map_tensor.squeeze(1).detach().cpu().numpy()
        goals_np = goals.detach().cpu().numpy()
        
        # 地图中心索引 (假设机器人总是在中心)
        center_idx = (self.map_size // 2, self.map_size // 2)

        for i in range(batch_size):
            grid = maps_np[i]  # 80x80
            g_x, g_y = goals_np[i]  # meters
            
            # 计算像素坐标
            row_idx = int(round(center_idx[0] - (g_x / self.map_res)))
            col_idx = int(round(center_idx[1] - (g_y / self.map_res)))
            raw_target = (row_idx, col_idx)
            target_node = (np.clip(row_idx, 0, self.map_size - 1),
                           np.clip(col_idx, 0, self.map_size - 1))

            # 欧氏距离 (保底用)
            euclidean = np.linalg.norm([g_x, g_y])
            
            if not (0 <= raw_target[0] < self.map_size and 0 <= raw_target[1] < self.map_size):
                distances.append(euclidean)
                continue

            # 构建 Cost Grid (0=空, 1=障碍)
            # 假设输入 map_tensor 中 1 是障碍，0 是空
            # skimage 需要: 1=通行代价低, 1000=通行代价高
            cost_grid = np.ones_like(grid)
            # 稍微膨胀一下障碍物判定阈值
            cost_grid[grid > 0.5] = 1000.0 

            try:
                # 检查起点终点是否在障碍物里
                if cost_grid[center_idx] > 100 or cost_grid[target_node] > 100:
                    dist = euclidean * 1.5  # 如果在墙里，给个惩罚
                else:
                    # 计算最小代价路径
                    indices, weight = graph.route_through_array(
                        cost_grid, start=center_idx, end=target_node, 
                        fully_connected=True, geometric=True
                    )
                    # weight 是像素距离，转回米
                    dist = weight * self.map_res
                    
                    # 如果算出来的距离大得离谱(穿墙了)，回退
                    if dist > self.max_dist * 3:
                        dist = euclidean * 1.5
            except:
                # 寻路失败 (不可达)
                dist = euclidean * 1.5
            
            distances.append(dist)

        return torch.tensor(distances, device=map_tensor.device, dtype=torch.float32)


class DecoderGridDynamic(nn.Module):
    def __init__(self, in_channels, goal_channels,
                 input_feature_size=10,
                 max_dist=10.0,
                 step_size=0.25):
        super().__init__()
        
        self.step_size = step_size
        
        self.max_dist = max_dist
        
        # === 自动计算 Max K ===
        # 例如 10.0 / 0.5 = 20 个点
        # 这是是不严谨的强制设置为10米，如果按0.5米步长，最大只能20个点。20个点能满足大部分场景。地图场景为8*8米。且机器人起点为中心点。
        self.max_k = int(math.ceil(10.0 / step_size))
        print(f"[INFO] 动态 Decoder 初始化: Max Dist={self.max_dist}m, Step={step_size}m, Max K={self.max_k}")
        self.relu = nn.ReLU(inplace=True)
        self.state_embedding = nn.Linear(2, goal_channels)
        self.sigmoid = nn.Sigmoid()

        # === 卷积层 ===
        # 保持 10x10 的空间尺寸 (padding=1)
        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        # === 自动计算 Flatten 维度 ===
        flatten_dim = self._get_flatten_dim(input_feature_size, in_channels + goal_channels)

        # === 全连接层 ===
        self.fc1 = nn.Linear(flatten_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # 输出层：输出固定数量的点 (max_k)
        # 修改：输出维度从 max_k * 4 (x,y,vx,vy) 改为 max_k * 2 (x,y)
        self.fc3 = nn.Linear(512, self.max_k * 2)

        # Cost 分支
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def _get_flatten_dim(self, size, channels):
        dummy = torch.zeros(1, channels, size, size)
        with torch.no_grad():
            x = self.conv2(self.conv1(dummy))
            return x.view(1, -1).size(1)
        
    # 输入归一化后的目标点坐标 goal_norm [B,2]，范围大约在 [-1,1]
    def forward(self, x, goal_norm, real_dist=None):
        """
        real_dist: [Batch] 由 Encoder 计算好的实际测地线距离 (Meters)
        """
        # state_input = goal  # [B, 2]
        state_encoded = self.state_embedding(goal_norm)  # [B, goal_channels]
        H, W = x.shape[2], x.shape[3]
        state_map = state_encoded[:, :, None, None].expand(-1, -1, H, W)
        
        # 拼接 + 卷积
        x = torch.cat((x, state_map), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        
        # MLP
        f = self.relu(self.fc1(x))
        
        # 预测路径 (全量预测)
        # 修改：输出维度从 [B, max_k, 4] 改为 [B, max_k, 2] (只有位置，无速度)
        out_path = self.fc3(self.relu(self.fc2(f)))
        out_path = out_path.reshape(-1, self.max_k, 2)
        
        # === 添加输出约束，防止梯度爆炸 ===
        # 使用tanh将输出限制在合理范围内，然后缩放到实际需要的范围
        # tanh输出[-1, 1]，缩放到[-max_dist, +max_dist]
        out_path = torch.tanh(out_path) * self.max_dist
        
        # 预测成本
        c = self.sigmoid(self.frc2(self.relu(self.frc1(f))))
        
        # --- 这里的逻辑变了 ---
        if real_dist is None:
            # 如果没传，退化为欧氏距离
            dist_to_use = torch.norm(goal_norm, p=2, dim=1) * self.max_dist
        else:
            # 使用传入的 A* 距离
            dist_to_use = real_dist

        # 计算需要的点数 (添加 10% 的余量)
        num_valid_points = torch.ceil((dist_to_use / self.step_size) * 1.1).long()
        # 截断到 max_k
        num_valid_points = num_valid_points.clamp(min=1, max=self.max_k)
        
        # 生成 Mask
        batch_size = x.size(0)
        indices = torch.arange(self.max_k, device=x.device).unsqueeze(0).expand(batch_size, -1)
        # padding_mask: True 表示是 padding 点，需要被屏蔽
        padding_mask = indices >= num_valid_points.unsqueeze(1)

        return out_path, c, padding_mask