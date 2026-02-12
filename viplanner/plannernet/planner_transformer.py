import torch
import torch.nn as nn
import math
import numpy as np
from skimage import graph 

# 假设 PlannerNetGrid 在 plannernet 文件夹下
try:
    from plannernet.PlannerNet_myself_cubic import PlannerNetGrid
except ImportError:
    pass

class TransformerPlanner(nn.Module):
    def __init__(self, 
                 encoder_channel=512, # ResNet backbone 输出通道数
                 d_model=512,         # Transformer 内部维度
                 nhead=8,             # 注意力头数
                 num_layers=4,        # Transformer 层数
                 max_dist=8.0, 
                 step_size=0.5):
        super().__init__()
        
        # 1. 参数设置
        self.map_size = 80
        self.map_res = 0.1
        self.max_dist = max_dist
        self.step_size = step_size
        self.max_k = int(math.ceil(10.0 / step_size)) # 预测点数
        
        # 2. CNN Backbone (提取视觉特征)
        # 输出 [Batch, 512, 10, 10]
        self.backbone = PlannerNetGrid(in_channels=1, small_input=True)
        
        # 3. Transformer 组件
        self.d_model = d_model
        
        # 目标点 Embedding: 将 (x,y) 映射到 d_model 维度
        self.goal_embedding = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 如果 Backbone 输出维度和 Transformer 不一致，需要投影 (这里默认都是512)
        if encoder_channel != d_model:
            self.input_proj = nn.Linear(encoder_channel, d_model)
        else:
            self.input_proj = nn.Identity()

        # 位置编码 (Positional Encoding)
        # 序列长度 = 10*10 (Map) + 1 (Goal) = 101
        self.pos_embed = nn.Parameter(torch.randn(1, 101, d_model) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出头 (Prediction Heads)
        # 路径预测头
        self.path_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, self.max_k * 2) # 输出 (x, y)
        )
        
        # 碰撞风险/Fear 预测头
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, goal_norm: torch.Tensor, real_dist=None):
        """
        Args:
            x: [Batch, 1, 80, 80] 地图
            goal_norm: [Batch, 2] 归一化目标点
            real_dist: [Batch] A* 真实距离 (可选)
        """
        B = x.shape[0]
        
        # --- 1. CNN 提取特征 ---
        # features: [B, 512, 10, 10]
        features = self.backbone(x)
        
        # --- 2. 构建 Transformer 输入序列 ---
        # Flatten map features: [B, 512, 10, 10] -> [B, 512, 100] -> [B, 100, 512]
        map_tokens = features.flatten(2).transpose(1, 2)
        map_tokens = self.input_proj(map_tokens)
        
        # Embed Goal: [B, 2] -> [B, 1, 512]
        goal_token = self.goal_embedding(goal_norm).unsqueeze(1)
        
        # 拼接: [Goal, Map_Patch_1, ..., Map_Patch_100]
        # Shape: [B, 101, 512]
        sequence = torch.cat([goal_token, map_tokens], dim=1)
        
        # 加上位置编码
        sequence = sequence + self.pos_embed
        
        # --- 3. Transformer 推理 ---
        # out_sequence: [B, 101, 512]
        out_sequence = self.transformer(sequence)
        
        # 取出第一个 token (Goal Token) 用于预测
        # 因为它通过 Attention 机制“看”过了整个地图
        cls_token = out_sequence[:, 0, :] # [B, 512]
        
        # --- 4. 解码输出 ---
        # 路径 [B, max_k * 2]
        flat_path = self.path_head(cls_token)
        out_path = flat_path.view(B, self.max_k, 2)
        
        # 约束输出范围 (与原 Decoder 保持一致)
        out_path = torch.tanh(out_path) * self.max_dist
        
        # 预测 Fear/Cost
        cost = self.cost_head(cls_token)
        
        # --- 5. 生成 Mask (与原代码逻辑完全一致) ---
        if real_dist is None:
            # 推理时如果没有 real_dist，用欧氏距离估计
            goal_meters = goal_norm * self.max_dist
            dist_to_use = torch.norm(goal_meters, p=2, dim=1)
            # 可以在这里加入 self._compute_geodesic_distance 的逻辑(如果需要)
        else:
            dist_to_use = real_dist

        num_valid_points = torch.ceil((dist_to_use / self.step_size) * 1.1).long()
        num_valid_points = num_valid_points.clamp(min=1, max=self.max_k)
        
        indices = torch.arange(self.max_k, device=x.device).unsqueeze(0).expand(B, -1)
        padding_mask = indices >= num_valid_points.unsqueeze(1)
        
        return out_path, cost, padding_mask

    # 保持这个辅助函数用于计算 loss 时的 mask
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