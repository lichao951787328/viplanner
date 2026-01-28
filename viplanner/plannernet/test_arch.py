import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
# from .autoencoder import AutoEncoder, DualAutoEncoder
# 导入你的模型
try:
    # 注意：这里假设你在 autoencoder_grid.py 里把类名改成了 AutoEncoderGrid
    # 并且 AutoEncoderGrid 内部使用的是 DecoderGridDynamic
    from autoencoder_myself import AutoEncoderGrid 
except ImportError:
    print("【错误】找不到 autoencoder_grid.py，请确保文件在当前目录下。")
    exit()

# === 核心补充：带掩码的 Loss 计算函数 ===
def compute_masked_loss(pred_path, gt_path, mask):
    """
    pred_path: [B, K, 3] (预测路径)
    gt_path:   [B, K, 3] (Ground Truth，虽然是随机生成的，但也需要对齐)
    mask:      [B, K]    (True 表示该点是无效 Padding，不应计算 Loss)
    """
    # 1. 计算所有点的 MSE (均方误差): [B, K, 3]
    raw_loss = (pred_path - gt_path) ** 2
    
    # 2. 扩展 mask 维度以匹配 loss: [B, K] -> [B, K, 1]
    mask_expanded = mask.unsqueeze(-1)
    
    # 3. 关键：将无效点 (mask=True) 的 Loss 强制置为 0
    # 这样网络就不会被迫去拟合那些“不存在”的点
    raw_loss = raw_loss.masked_fill(mask_expanded, 0.0)
    
    # 4. 计算平均 Loss
    # 分母应该是“有效数据的总数”，而不是 Batch * K * 3
    # (~mask) 选出的是 False (即有效点) 的位置
    num_valid_elements = (~mask).sum() * 3 
    
    # 加一个极小值 1e-6 防止除以零
    final_loss = raw_loss.sum() / (num_valid_elements + 1e-6)
    
    return final_loss

def check_model_structure():
    print("\n" + "="*20 + " 1. 静态结构检查 (Dry Run) " + "="*20)
    
    # 定义配置
    BATCH_SIZE = 4
    INPUT_H, INPUT_W = 80, 80
    MAX_DIST = 10.0
    STEP_SIZE = 0.5
    # 计算理论最大点数: ceil(10 / 0.5) = 20
    MAX_K = int(math.ceil(MAX_DIST / STEP_SIZE)) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")

    # 实例化模型 (确保你的 AutoEncoderGrid 能接收 max_dist 参数，或者在内部写死)
    try:
        # 注意：这里传入的 k=MAX_K 主要是为了占位，实际逻辑由 DecoderGridDynamic 内部控制
        model = AutoEncoderGrid(encoder_channel=64, max_dist=MAX_DIST, step_size=STEP_SIZE).to(device)
        print("[INFO] 模型实例化成功！")
    except Exception as e:
        print(f"【严重错误】模型实例化失败: {e}")
        return None

    # 构造假数据
    dummy_input = torch.randn(BATCH_SIZE, 1, INPUT_H, INPUT_W).to(device)
    dummy_goal = torch.randn(BATCH_SIZE, 3).to(device)

    # 前向传播测试
    try:
        print("[INFO] 开始前向传播测试...")
        
        # === 修改点：这里接收 3 个返回值 ===
        out_path, out_cost, out_mask = model(dummy_input, dummy_goal)
        
        print(f"[SUCCESS] 前向传播成功！")
        print(f"   - Input Shape: {dummy_input.shape}")
        print(f"   - Output Path Shape: {out_path.shape} (预期: [{BATCH_SIZE}, {MAX_K}, 3])")
        print(f"   - Output Cost Shape: {out_cost.shape} (预期: [{BATCH_SIZE}, 1])")
        print(f"   - Output Mask Shape: {out_mask.shape} (预期: [{BATCH_SIZE}, {MAX_K}])")
        
        # 简单验证一下 Mask 是否生效
        # 计算 dummy_goal 的距离
        dists = torch.norm(dummy_goal[:, :2], dim=1)
        print(f"   - 随机生成的目标距离: {dists.detach().cpu().numpy()}")
        print(f"   - Mask 中无效点数量: {out_mask.sum(dim=1).detach().cpu().numpy()} (距离越近，无效点应该越多)")
        
    except RuntimeError as e:
        print(f"【错误】前向传播失败: {e}")
        return None
    except ValueError as e:
        print(f"【错误】返回值数量不对，请检查模型 forward 是否返回了 mask: {e}")
        return None
        
    return model

def check_overfitting_capability(model):
    print("\n" + "="*20 + " 2. 动态学习能力检查 (Overfit One Batch) " + "="*20)
    
    device = next(model.parameters()).device
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # === 准备“假”训练数据 ===
    # 1. 固定输入图像
    fixed_input = torch.randn(4, 1, 80, 80).to(device)
    
    # 2. 固定目标点 (这是 Input Goal)
    # 我们故意设置不同的距离，测试 Mask 是否有效
    # 样本0: 距离很近 (比如 1m) -> 应该有很多 Mask=True
    # 样本1: 距离很远 (比如 9m) -> 应该很少 Mask=True
    fixed_goal = torch.tensor([
        [1.0, 0.0, 0.0],  # 近
        [9.0, 0.0, 0.0],  # 远
        [3.0, 3.0, 0.0],  # 中
        [5.0, -5.0, 0.0]  # 中
    ]).to(device)
    
    # 3. 伪造 Ground Truth 路径 (这是 Label)
    # 这里的尺寸必须是 [Batch, Max_K, 3]
    # 即使对于近距离目标，GT 矩阵也得填满，但 Loss 会根据 Mask 忽略掉多余的部分
    max_k = model.decoder.max_k if hasattr(model, 'decoder') else 20
    target_path = torch.randn(4, max_k, 3).to(device)
    
    # 4. 伪造 Cost Label
    target_cost = torch.rand(4, 1).to(device)

    losses = []
    print(f"[INFO] 开始训练 50 Steps (Max K={max_k})...")
    
    for i in range(51):
        optimizer.zero_grad()
        
        # Forward: 获取预测值 + Mask
        pred_path, pred_cost, mask = model(fixed_input, fixed_goal)
        
        # Loss 1: 路径 Loss (使用 Mask!)
        loss_path = compute_masked_loss(pred_path, target_path, mask)
        
        # Loss 2: 成本 Loss (普通 MSE)
        loss_cost = nn.MSELoss()(pred_cost, target_cost)
        
        total_loss = loss_path + loss_cost
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if i % 10 == 0:
            print(f"   Step {i:02d}: Loss = {total_loss.item():.6f}")

    # 绘图
    plt.plot(losses)
    plt.title("Sanity Check: Dynamic Masked Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

    if losses[-1] < 0.1:
        print("[SUCCESS] 动态 Loss 验证通过！模型能够学会忽略 Padding 部分。")
    else:
        print("[WARNING] Loss 下降不明显，请检查 Mask 逻辑。")

if __name__ == "__main__":
    model = check_model_structure()
    if model:
        check_overfitting_capability(model)