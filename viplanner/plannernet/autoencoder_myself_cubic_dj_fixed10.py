import torch
import torch.nn as nn
import math

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
        # --- 步骤 1: 正常的网络推理 ---
        # Encoder
        features = self.encoder(x)
        
        # fixed10版本：固定输出10个点，不依赖A*长度
        out_path, cost, mask = self.decoder(features, goal_norm, real_dist=None)
        
        return out_path, cost, mask


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
        self.max_k = 10
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
        
        return out_path, c, None