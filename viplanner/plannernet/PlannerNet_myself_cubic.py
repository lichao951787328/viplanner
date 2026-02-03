import torch
import torch.nn as nn

# === 辅助函数 (保持不变) ===
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# === BasicBlock (保持不变) ===
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)  # 建议加上 BN，虽然原代码没写，但ResNet标准是有BN的
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if hasattr(self, 'bn1'): out = self.bn1(out)  # 兼容原代码可能没有BN的情况
        out = self.relu(out)
        out = self.conv2(out)
        if hasattr(self, 'bn2'): out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# === 适配 80x80 栅格地图的 PlannerNet ===
class PlannerNetGrid(nn.Module):
    def __init__(
        self,
        layers=[2, 2, 2, 2],  # 对应 ResNet-18
        in_channels=1,        # 默认为 1，适应占用栅格地图
        small_input=True      # 针对 80x80 开启此选项，避免下采样过快
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.small_input = small_input
        
        # --- 1. 输入层 (Stem) ---
        if self.small_input:
            # 针对 80x80 的优化 Stem：
            # 使用 3x3 卷积，步长为 1，去掉 MaxPool
            # 输出: 80x80 -> 80x80
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # 原始 ImageNet Stem：
            # 使用 7x7 卷积，步长为 2，加上 MaxPool
            # 输出: 80x80 -> 40x40 -> 20x20 (下采样太快，不推荐)
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # --- 2. 残差层 (Layers) ---
        # Layer 1: 64通道 (Stride 1) -> 80x80 (如果 small_input=True)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        
        # Layer 2: 128通道 (Stride 2) -> 40x40
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        
        # Layer 3: 256通道 (Stride 2) -> 20x20
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        
        # Layer 4: 512通道 (Stride 2) -> 10x10
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, 1, 80, 80]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if not self.small_input:
            x = self.maxpool(x)

        x = self.layer1(x)  # 80x80
        x = self.layer2(x)  # 40x40
        x = self.layer3(x)  # 20x20
        x = self.layer4(x)  # 10x10
        
        # 输出特征图: [Batch, 512, 10, 10]
        # 相比原始版本的 3x3，这里保留了 10x10 的空间特征，更适合路径规划
        return x

# === 使用示例 ===
if __name__ == "__main__":
    # 1. 创建模型 (输入通道=1, 开启小图模式)
    model = PlannerNetGrid(in_channels=1, small_input=True)
    
    # 2. 模拟一个 Batch 的 80x80 栅格地图
    # [Batch=2, Channel=1, Height=80, Width=80]
    dummy_input = torch.randn(2, 1, 80, 80)
    
    # 3. 前向传播
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")   # torch.Size([2, 1, 80, 80])
    print(f"Output Shape: {output.shape}")       # torch.Size([2, 512, 10, 10])
    
    # 4. 计算展平后的特征维度 (供 Decoder 使用)
    flatten_dim = output.shape[1] * output.shape[2] * output.shape[3]
    print(f"Flattened Feature Dim: {flatten_dim}") # 512 * 10 * 10 = 51200