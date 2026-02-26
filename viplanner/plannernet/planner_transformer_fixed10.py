import torch
import torch.nn as nn
import math

try:
    from plannernet.PlannerNet_myself_cubic import PlannerNetGrid
except ImportError:
    pass

class TransformerPlanner(nn.Module):
    def __init__(self, 
                 encoder_channel=512, # ResNet backbone 输出通道数 (ResNet18 layer4 是 512)
                 d_model=512,         # Transformer 内部维度 (建议保持 512)
                 nhead=8,             # 注意力头数
                 num_layers=4,        # 层数 (4层对于这种规模的地图足够了)
                 max_dist=8.0, 
                 step_size=0.5):
        super().__init__()
        
        # 1. 参数设置
        self.map_size = 80
        self.map_res = 0.1
        self.max_dist = max_dist
        self.step_size = step_size
        self.max_k = 6 # 固定输出6个路径点（不依赖A*长度）
        
        # 2. CNN Backbone
        # 输出 [Batch, 512, 10, 10]
        self.backbone = PlannerNetGrid(in_channels=1, small_input=True)
        
        # 3. Transformer 组件
        self.d_model = d_model
        
        # Goal Embedding
        self.goal_embedding = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Projection: 如果 CNN 通道数与 Transformer 维度不同，则进行投影
        if encoder_channel != d_model:
            self.input_proj = nn.Linear(encoder_channel, d_model)
        else:
            self.input_proj = nn.Identity()

        # Positional Encoding
        # 序列长度 = 100 (10x10 Map) + 1 (Goal) = 101
        self.pos_embed = nn.Parameter(torch.randn(1, 101, d_model) * 0.02)
        
        # Transformer Encoder
        # 使用 batch_first=True 符合直觉 [Batch, Seq, Dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True,
            norm_first=True # 【建议】Pre-Norm 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Heads
        self.path_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, self.max_k * 2) 
        )
        
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 【建议】初始化输出头，使初始轨迹接近 0，加速收敛
        with torch.no_grad():
            self.path_head[-1].weight.mul_(0.1)
            self.path_head[-1].bias.zero_()

    # 后期可以考虑增加一个分支，直接预测路径长度（Distance），用于生成动态 Mask：
    # 方案 C：预测一个标量 Distance（需要改一点模型）
    # 原理：在 Transformer 的 cls_token 上再加一个全连接层 dist_head，让它回归一个标量：predicted_total_length。
    # 推理时：先跑一遍模型得到 pred_len，然后根据 pred_len 生成 Mask，再取路径点。
    # 优点：显式建模了路径长度。
    def forward(self, x: torch.Tensor, goal_norm: torch.Tensor, real_dist=None):
        B = x.shape[0]
        
        # 1. CNN Feature: [B, 512, 10, 10]
        features = self.backbone(x)
        
        # 2. Prepare Sequence
        # [B, 512, 100] -> [B, 100, 512]
        map_tokens = features.flatten(2).transpose(1, 2)
        map_tokens = self.input_proj(map_tokens)
        
        # Goal Token: [B, 1, 512]
        goal_token = self.goal_embedding(goal_norm).unsqueeze(1)
        
        # Concat: [B, 101, 512]
        sequence = torch.cat([goal_token, map_tokens], dim=1)
        
        # Add Positional Encoding
        sequence = sequence + self.pos_embed
        
        # 3. Transformer Forward
        out_sequence = self.transformer(sequence)
        
        # 4. Use the Goal Token (index 0) for prediction
        # Goal token 此时已经聚合了全局地图信息
        cls_token = out_sequence[:, 0, :] 
        
        # 5. Prediction
        flat_path = self.path_head(cls_token)
        out_path = flat_path.view(B, self.max_k, 2)
        
        # Limit Output Range
        out_path = torch.tanh(out_path) * self.max_dist
        
        cost = self.cost_head(cls_token)
        
        return out_path, cost, None