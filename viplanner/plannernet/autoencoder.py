# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Any

import torch
import torch.nn as nn
import math

# visual-imperative-planner
from .PlannerNet import PlannerNet
from .rgb_encoder import PRE_TRAIN_POSSIBLE, RGBEncoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        # 使用 PlannerNet 作为编码器，结构为 ResNet-18 的变体 ([2,2,2,2]对应各层Block数量)
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        # 定义解码器，输入特征图通道数512，encoder_channel用于之前的版本兼容或特定参数，k是输出路径点数量
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        # 输入 x 通常是单通道深度图 (N, 1, H, W)，这里将其扩展为 3 通道以适配 ResNet 输入
        x = x.expand(-1, 3, -1, -1)
        # 编码提取特征
        x = self.encoder(x)
        # 解码，融合目标 goal，输出路径 x 和 碰撞/成本分数 c
        x, c = self.decoder(x, goal)
        return x, 
# === 双模态自动编码器 (Depth + Semantic) ===
class DualAutoEncoder(nn.Module):
    def __init__(
        self,
        train_cfg: Any,
        m2f_cfg=None,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        # 1. 深度图编码器：始终使用 PlannerNet (ResNet-18)
        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem and PRE_TRAIN_POSSIBLE:
            # # 2. 语义/RGB编码器：
            # 如果配置启用了RGB，且允许预训练语义模型，则使用 RGBEncoder (基于Mask2Former/Detectron2)
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            # 否则也使用标准的 PlannerNet
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])

        # 3. 解码器选择：根据配置选择标准解码器或小型解码器 (DecoderS)
        # 输入通道是 1024 (因为两个编码器输出各512，拼接后为1024)
        if train_cfg.decoder_small:
            self.decoder = DecoderS(1024, train_cfg.in_channel, train_cfg.knodes)
        else:
            self.decoder = Decoder(1024, train_cfg.in_channel, train_cfg.knodes)
        return

    def forward(self, x_depth: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        # encode depth # 编码深度图
        x_depth = x_depth.expand(-1, 3, -1, -1)
        x_depth = self.encoder_depth(x_depth)
        # encode sem # 编码语义图
        x_sem = self.encoder_sem(x_sem)
        # concat # 特征拼接：在通道维度拼接 (N, 512, H, W) -> (N, 1024, H, W)
        x = torch.cat((x_depth, x_sem), dim=1)  # x.size = (N, 1024, 12, 20)
        # decode # 解码生成路径和成本
        x, c = self.decoder(x, goal)
        return x, c


# # === 标准解码器 ===
class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k # 输出路径点的数量
        self.relu = nn.ReLU(inplace=True)
        # 将 goal (x,y,z 3维) 映射到 goal_channels 维度
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        # # 卷积层：融合图像特征和Goal特征
        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels), # 输入通道 = 图像特征通道 + Goal通道
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        # 全连接层用于生成路径点
        self.fc1 = nn.Linear(256 * 128, 1024) # 输入维度取决于上一层卷积输出的尺寸 (需根据输入图像大小计算)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3) # 输出 k 个点，每个点 (x, y, z)

        # # 分支全连接层用于生成成本/碰撞概率
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding # 1. 处理 Goal
        goal = self.fg(goal[:, 0:3])
        # 将 Goal 向量扩展为与特征图 x 相同的空间尺寸 (H, W)，以便进行 Concat
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim # 2. 拼接 特征图 与 Goal
        x = torch.cat((x, goal), dim=1)
        # compute x # 3. 卷积处理
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        # # 4. 共享全连接层
        f = self.relu(self.fc1(x))
        # 5. 路径预测分支
        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)
        # 6. 成本/碰撞预测分支
        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c

# === 小型解码器 (DecoderS) ===
# 结构类似 Decoder，但卷积层更多、通道数更少，全连接层参数更少，用于轻量化部署
class DecoderS(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels),
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(64 * 48, 256)  # --> in that setting 33 million parameters
        self.fc2 = nn.Linear(256, k * 3)

        self.frc1 = nn.Linear(256, 1)

    # # ... (初始化定义了4层卷积，逐步降维) ...
    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)  # x.size = (N, 1024+16, 12, 20)
        # compute x
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)  --> (N, 512, 10, 18),
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)  --> (N, 256, 8, 16)
        x = self.relu(self.conv3(x))  # size = (N, 512, x.H/90, x.W/90)  --> (N, 128, 6, 14)
        x = self.relu(self.conv4(x))  # size = (N, 512, x.H/120, x.W/120) --> (N, 64, 4, 12)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.fc2(f)
        x = x.reshape(-1, self.k, 3)

        c = self.sigmoid(self.frc1(f))

        return x, c


# 基于transformer的decoder，动态调整路径点数量
class DynamicScalePlanner(nn.Module):
    def __init__(self, step_size=0.4, max_dist=10.0):
        super().__init__()
        self.step_size = step_size  # 空间分辨率，每0.4米一个点
        self.max_tokens = int(math.ceil(max_dist / step_size)) # 最大Token数 = 25
        
        # --- Encoder (沿用之前的 CNN+Transformer) ---
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.pos_embedding = nn.Parameter(torch.randn(1, 225, 64))

        # --- Decoder ---
        # 1. 距离编码: 类似于正弦位置编码，但代表的是"第i个0.4米"
        self.dist_embedding = nn.Embedding(self.max_tokens, 64)
        
        # 2. 目标编码: 把目标点(x,y)融合进Query，告诉网络方向
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # 4. 输出头
        self.regressor = nn.Linear(64, 2)

    def forward(self, map_input, goal_pos):
        """
        map_input: [B, 1, 120, 120]
        goal_pos:  [B, 2] (局部坐标)
        """
        batch_size = map_input.size(0)
        device = map_input.device

        # === 1. Encoder (提取地图特征) ===
        features = self.cnn_backbone(map_input)
        memory = features.view(batch_size, 64, -1).permute(0, 2, 1)
        memory = self.encoder(memory + self.pos_embedding)

        # === 2. 构建 Dynamic Mask ===
        # 计算目标距离
        dist_to_goal = torch.norm(goal_pos, dim=1) # [B]
        
        # 计算每个样本需要多少个点 (Ceil操作)
        # 比如 dist=2.0, step=0.4 -> num_valid=5
        num_valid_tokens = torch.ceil(dist_to_goal / self.step_size).long().clamp(min=1, max=self.max_tokens)
        
        # 生成 Mask: [B, max_tokens]
        # True 表示要被 mask 掉 (Padding)，False 表示有效
        indices = torch.arange(self.max_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_key_padding_mask = indices >= num_valid_tokens.unsqueeze(1)

        # === 3. 构建 Queries ===
        # Query 由两部分组成：
        # Part A: "我是第几个点？" (距离信息)
        dist_queries = self.dist_embedding(torch.arange(self.max_tokens, device=device)) # [25, 64]
        dist_queries = dist_queries.unsqueeze(0).expand(batch_size, -1, -1) # [B, 25, 64]
        
        # Part B: "目标在哪？" (方向信息)
        goal_feat = self.goal_encoder(goal_pos).unsqueeze(1) # [B, 1, 64]
        
        # 最终 Query = 距离编码 + 目标方向编码
        # 这样 Query[i] 就理解为："请预测向着Goal方向，距离起点 i*step 米处的坐标"
        queries = dist_queries + goal_feat

        # === 4. Decoder 推理 ===
        # 传入 tgt_key_padding_mask，让 Attention 忽略那些超出终点的 Query
        out_tokens = self.decoder(
            tgt=queries, 
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # === 5. 回归坐标 ===
        waypoints = self.regressor(out_tokens) # [B, 25, 2]
        
        return waypoints, tgt_key_padding_mask

# 和上一步对应的损失函数
# def compute_loss(pred_path, mask, costmap, goal_pos):
#     """
#     pred_path: [B, 25, 2]
#     mask: [B, 25] (True为无效点)
#     costmap: [B, H, W]
#     """
#     # 1. 碰撞/穿越成本 (Traversability Cost)
#     # 采样 Costmap 上的值。只取 mask 为 False 的有效点。
#     # 这里的 sample_cost 是一个双线性插值函数
#     path_costs = sample_cost_from_map(costmap, pred_path) # [B, 25]
    
#     # 关键：把无效点的 Cost 设为 0
#     path_costs[mask] = 0.0
    
#     # 对有效点求平均或求和
#     valid_counts = (~mask).sum(dim=1)
#     traversability_loss = path_costs.sum(dim=1) / valid_counts
    
#     # 2. 平滑性 Loss (Smoothness)
#     # 计算 ||P_i - P_{i-1}|| 的变化
#     # 同样需要利用 mask，只计算有效段的平滑度
    
#     # 3. 目标点 Loss (Goal Loss)
#     # 我们只希望路径的"最后一个有效点"接近 Goal
#     # 获取每个 Batch 最后一个有效点的索引
#     last_valid_idx = (~mask).sum(dim=1) - 1
#     # 提取最后一个点
#     end_points = pred_path[torch.arange(len(pred_path)), last_valid_idx]
#     goal_loss = torch.nn.functional.mse_loss(end_points, goal_pos)
    
#     return traversability_loss.mean() + goal_loss

# EoF
