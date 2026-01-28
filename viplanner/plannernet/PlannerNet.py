# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn

# 辅助函数：定义 3x3 卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

# 辅助函数：定义 1x1 卷积 (用于调整通道数)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# === ResNet 的基本残差块 (BasicBlock) ===
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道倍数，BasicBlock为1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1 # 第一层卷积
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # 下采样层 (如果输入输出尺寸或通道不匹配，用于调整 identity 路径)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差相加
        out = self.relu(out)

        return out

# === 主网络结构 ===
class PlannerNet(nn.Module):
    def __init__(
        self,
        layers,
        block=BasicBlock,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        # 目的：迅速缩小图像尺寸，减少后续计算量。操作：conv1 (7x7, 步长2)：图片长宽减半，通道数变 64。maxpool (3x3, 步长2)：图片长宽再减半。效果：如果输入是 $120 \times 120$，这一套下来变成了 $30 \times 30$。虽然丢了细节，但大概轮廓还在。
        # 初始层：7x7 卷积，步长2 (快速降低分辨率)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4个残差层 (Layer 1-4)，对应 ResNet 的四个阶段
        # layers 参数通常是 [2, 2, 2, 2] 对应 ResNet-18
        # 精细提取特征。因为步长是 1，这里图像大小不变 ($30 \times 30$)。它负责看清“纹理、小边缘”。
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 下采样。长宽变 $15 \times 15$，通道翻倍。开始聚合特征，比如把几条线组合成一个“角”。
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        # 下采样。长宽变 $8 \times 8$（近似），通道翻倍。识别更复杂的形状，比如“障碍物块”。
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        # 目的：下采样。长宽变 $4 \times 4$（近似），通道翻倍。
        # 语义：这时候的每一个像素点，代表了原图中很大一块区域（比如 32x32 的范围）。它不再表示“颜色”，而是表示“这里是否可以通过”、“这里是不是终点方向”这种高度抽象的概念。
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

# 构建层的辅助函数 # ... 负责堆叠 BasicBlock 并处理步长/空洞卷积 ...
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()] # 前向传播流程
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 注意：这里没有 Global Average Pooling 和 FC 层，直接输出特征图
        return x

    def forward(self, x):
        return self._forward_impl(x)


class HybridCNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. CNN 前端：负责提取几何特征并降维
        # 输入 120x120 -> 输出 15x15 (降采样8倍)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),  # 60x60
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 30x30
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 15x15
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        # 2. Transformer 部分
        # 序列长度 = 15*15 = 225
        self.hidden_dim = 64
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3) # 3层足够了
        
        # 位置编码 (Positional Encoding) - 这里的简单实现是用可学习参数
        self.pos_embedding = nn.Parameter(torch.randn(1, 225, 64))
        
        # 3. 输出头
        self.head = nn.Sequential(
            nn.Linear(64 * 225, 256),
            nn.ReLU(),
            nn.Linear(256, 20) # 假设输出10个点的(x,y)
        )

    def forward(self, x):
        # x: [Batch, 1, 120, 120]
        
        # 1. CNN 提取特征
        features = self.cnn_backbone(x) # [Batch, 64, 15, 15]
        
        # 2. 变换维度以适应 Transformer
        # 变成 [Batch, 225, 64] (Batch, Sequence, Feature)
        b, c, h, w = features.shape
        tokens = features.view(b, c, -1).permute(0, 2, 1)
        
        # 加上位置编码
        tokens = tokens + self.pos_embedding
        
        # 3. Transformer 全局推理
        out_tokens = self.transformer(tokens)
        
        # 4. 展平并输出
        flat = out_tokens.flatten(1)
        path = self.head(flat)
        return path
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 输入: [Batch, 1, 120, 120]
        self.features = nn.Sequential(
            # 第一层: 提取基础几何特征 (120 -> 60)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 第二层: 组合特征 (60 -> 30)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第三层: 提取高层通路特征 (30 -> 15)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第四层 (可选): 如果觉得15x15太大，可以再卷一次 (15 -> 7)
            # 或者保持 15x15 的特征图进入全连接层，保留更多空间感
        )
        
        # 此时特征图大小为 [Batch, 64, 15, 15]
        # Flatten后大小为 64 * 15 * 15 = 14400
        
        self.planner_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # 假设输出是 N 个路径点的 (x,y) 坐标
            nn.Linear(128, 2 * 10) # 例如输出 10 个点
        )

    def forward(self, x):
        # x shape: [Batch, 1, 120, 120]
        feat = self.features(x)
        out = self.planner_head(feat)
        return out

