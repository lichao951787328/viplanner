# autoencoder_grid.py
import torch
import torch.nn as nn
import math

# 导入你的 PlannerNet
try:
    from plannernet.PlannerNet_myself import PlannerNetGrid
except ImportError:
    print("[Error] 请先将 PlannerNetGrid 类添加到 PlannerNet_myself.py 文件中！")
    PlannerNetGrid = None

class AutoEncoderGrid(nn.Module):
    def __init__(self, encoder_channel=64, max_dist=10.0, step_size=0.5):
        super().__init__()
        
        # 1. 初始化 Encoder
        # in_channels=1 (占用栅格), small_input=True (保留 10x10 特征图)
        self.encoder = PlannerNetGrid(in_channels=1, small_input=True)
        
        # 2. 初始化 Decoder (动态版)
        # 注意：这里我们不再传 k，而是传 max_dist 和 step_size
        # Decoder 内部会自动计算出 k = max_dist / step_size
        self.decoder = DecoderGridDynamic(
            in_channels=512, 
            goal_channels=encoder_channel, 
            input_feature_size=10,
            max_dist=max_dist,
            step_size=step_size
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        # x: [Batch, 1, 80, 80]
        # goal: [Batch, 3]
        
        # Encoder 推理
        x = self.encoder(x) # -> [Batch, 512, 10, 10]
        
        # Decoder 推理
        # 返回三个值: path, cost, mask
        x, c, mask = self.decoder(x, goal)
        
        return x, c, mask

class DecoderGridDynamic(nn.Module):
    def __init__(self, in_channels, goal_channels, 
                 input_feature_size=10, 
                 max_dist=10.0, 
                 step_size=0.5):
        super().__init__()
        
        self.step_size = step_size
        self.max_dist = max_dist
        
        # === 自动计算 Max K ===
        # 例如 10.0 / 0.5 = 20 个点
        self.max_k = int(math.ceil(max_dist / step_size))
        print(f"[INFO] 动态 Decoder 初始化: Max Dist={max_dist}m, Step={step_size}m, Max K={self.max_k}")

        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
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
        self.fc3 = nn.Linear(512, self.max_k * 3)

        # Cost 分支
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def _get_flatten_dim(self, size, channels):
        dummy = torch.zeros(1, channels, size, size)
        with torch.no_grad():
            x = self.conv2(self.conv1(dummy))
            return x.view(1, -1).size(1)

    def forward(self, x, goal):
        # --- 1. 常规网络前向传播 ---
        # 目标编码
        goal_encoded = self.fg(goal[:, 0:3])
        goal_map = goal_encoded[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        
        # 拼接 + 卷积
        x = torch.cat((x, goal_map), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        
        # MLP
        f = self.relu(self.fc1(x))
        
        # 预测路径 (全量预测)
        out_path = self.fc3(self.relu(self.fc2(f)))
        out_path = out_path.reshape(-1, self.max_k, 3)
        
        # 预测成本
        c = self.sigmoid(self.frc2(self.relu(self.frc1(f))))

        # --- 2. 动态掩码计算 (Dynamic Masking) ---
        # 计算每个样本目标的实际距离
        dist_to_goal = torch.norm(goal[:, :2], p=2, dim=1) # [Batch]
        
        # 计算实际需要的点数 (向上取整)
        num_valid_points = torch.ceil(dist_to_goal / self.step_size).long().clamp(min=1, max=self.max_k)
        
        # 生成掩码 [Batch, max_k]
        # True = 无效点 (Padding)
        batch_size = x.size(0)
        indices = torch.arange(self.max_k, device=x.device).unsqueeze(0).expand(batch_size, -1)
        padding_mask = indices >= num_valid_points.unsqueeze(1)

        return out_path, c, padding_mask