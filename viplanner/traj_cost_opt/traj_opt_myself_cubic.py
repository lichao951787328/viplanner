# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 在 PyTorch 框架下实现轨迹插值和平滑。它利用**三次样条插值（Cubic Spline Interpolation）**将稀疏的路径点（比如神经网络预测的几个关键点）转换成平滑、连续的高分辨率轨迹。
# 由于完全使用 PyTorch 操作，这整个过程是可微分的，意味着它可以直接嵌入到神经网络的训练流程中，允许梯度反向传播。
import torch
from torch.nn.utils.rnn import pad_sequence
torch.set_default_dtype(torch.float32)

# 实现了基于 Hermite 样条的三次插值数学逻辑
class CubicSplineTorch:
    # Reference: https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    def __init__(self):
        # 初始化一个用于后续计算切线的张量（虽然在这段代码中似乎未被直接使用，可能是保留代码）
        self.init_m = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    # 计算 Hermite 基函数多项式
    def h_poly(self, t):
        # 创建指数序列 [0, 1, 2, 3]
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        # 计算 t 的幂次方: [1, t, t^2, t^3]
        # t[:, None, :] 增加维度以支持广播
        tt = t[:, None, :] ** alpha[None, :, None]
        # Hermite 样条的系数矩阵 (标准形式)
        # 对应多项式: 
        # h00 = 1 - 3t^2 + 2t^3
        # h10 = t - 2t^2 + t^3
        # h01 = 3t^2 - 2t^3
        # h11 = -t^2 + t^3
        A = torch.tensor(
            [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
            dtype=t.dtype,
            device=t.device,
        )
        return A @ tt
    # 注意这里并没有严格的考虑机器人的真实物理速度，只是假设机器人在控制点之间的运动是平滑连续的。
    # 这种假设在很多路径规划场景下是合理的，尤其是当控制点足够密集时。
    # 但是不符合轮式差速地盘和人形，人形机器人对稳定性要求很高，如果速度不连续会导致跌倒。
    def interp(self, x, y, xs):
        # x: 原始点的索引（如 0, 1, 2...）
        # y: 原始路径点坐标 (Batch, Points, Dims)
        # xs: 需要插值出的新时间点索引
        
        # 1. 计算切线 (Slope/Tangent) 'm'
        # 使用有限差分法：(后一点 - 前一点) / 距离
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        # 处理边界切线：
        # 中间点的切线取前后两段斜率的平均值 (Catmull-Rom 样条风格)
        # 起点和终点直接取第一段和最后一段的斜率
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        # 2. 确定 xs 中的每个点落在原始 x 的哪一段区间内
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        idxs = torch.clamp(idxs, max=x.shape[1] - 2)
        # 计算该段区间的长度 dx
        dx = x[:, idxs + 1] - x[:, idxs]
        # 3. 计算归一化时间 t = (当前点 - 区间起点) / 区间长度
        # 然后代入 h_poly 计算基函数值
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)
        # 4. 组合最终结果 (Hermite 插值公式)
        # p(t) = h00*p0 + h10*m0*dx + h01*p1 + h11*m1*dx
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:, :, None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:, :, None]
        return out
    
    
class TrajOpt:
    debug = False

    def __init__(self):
        self.cs_interp = CubicSplineTorch()

    def TrajGeneratorFromPFreeRotVI(self, preds, step):
        # Points is in se3
        # preds: 预测的关键点，形状通常为 [Batch, Num_Points, Dims]
        # step: 插值的步长（决定了输出轨迹的密度）
        batch_size, num_p, dims = preds.shape
        # 1. 添加原点
        # 神经网络通常只预测未来的点，不包含当前位置。
        # 这里假设机器人当前在局部坐标系的 (0,0,0)，将其拼接到预测点序列的最前面。
        points_preds = torch.cat(
            (
                torch.zeros(
                    batch_size,
                    1,
                    dims,
                    device=preds.device,
                    requires_grad=preds.requires_grad,
                ),
                preds,
            ),
            axis=1,
        )
        # 更新点数（因为加了原点，所以 +1）
        num_p = num_p + 1
        # 2. 构造时间/索引向量
        # xs: 目标插值点索引（例如 0.0, 0.1, 0.2 ... num_p-1）
        xs = torch.arange(0, num_p - 1 + step, step, device=preds.device)
        xs = xs.repeat(batch_size, 1)
        # x: 原始控制点索引（例如 0, 1, 2 ... num_p）
        x = torch.arange(num_p, device=preds.device, dtype=preds.dtype)
        x = x.repeat(batch_size, 1)
        waypoints = self.cs_interp.interp(x, points_preds, xs)

        # print(f"[DEBUG] Generated waypoints shape: {waypoints.shape}")

        if self.debug:
            import matplotlib.pyplot as plt  # for plotting

            plt.scatter(
                points_preds[0, :, 0].cpu().numpy(),
                points_preds[0, :, 1].cpu().numpy(),
                label="Samples",
                color="purple",
            )
            plt.plot(
                waypoints[0, :, 0].cpu().numpy(),
                waypoints[0, :, 1].cpu().numpy(),
                label="Interpolated curve",
                color="blue",
            )
            plt.legend()
            plt.show()

        return waypoints

    # 从预测点生成轨迹
    def TrajGeneratorFromPFreeRot(self, preds, step, mask=None):
        if mask is None:
            return self.TrajGeneratorFromPFreeRotVI(preds, step)
        
        # 1. 计算每个样本的有效长度
        # mask: True 表示无效/Padding
        valid_lens = (~mask).sum(dim=1).long()  # [Batch]
        
        batch_size = preds.shape[0]
        interpolated_trajs = []
        
        # 2. 循环处理每个样本 (Loop over Batch)
        for i in range(batch_size):
            curr_len = valid_lens[i]
            
            # 拿到当前样本的有效控制点 [Valid_N, 2]
            # 这一步非常关键：彻底剔除了尾部的垃圾数据
            curr_pred = preds[i, :curr_len, :] 
            
            # 升维成 [1, Valid_N, 2] 传给插值函数
            # 注意：这里的 step 决定了输出的密度，由于 curr_len 不同，
            # 生成出来的轨迹长度 M 也会不同！
            curr_traj = self.TrajGeneratorFromPFreeRotVI(curr_pred.unsqueeze(0), step)
            
            # 降维回 [M, 2] 并存入列表
            interpolated_trajs.append(curr_traj.squeeze(0))

        # 3. 将变长的轨迹重新 Padding 成 Batch Tensor
        # pad_sequence 会自动把短的轨迹后面补 0，对齐到最长的轨迹
        # 输出: [Batch, Max_M, 2]
        waypoints_batch = pad_sequence(interpolated_trajs, batch_first=True, padding_value=0.0)
        
        # 4. (可选) 可视化调试
        if self.debug and batch_size > 0:
            import matplotlib.pyplot as plt
            idx = 0 # 看第一个样本
            orig = preds[idx, :valid_lens[idx]].detach().cpu().numpy()
            interp = waypoints_batch[idx].detach().cpu().numpy()
            # 剔除掉 padding 的 0 (假设 trajectory 不会正好在 0,0 结束)
            # 或者简单画图不剔除也行
            plt.figure()
            plt.scatter(orig[:,0], orig[:,1], c='purple', label='Control Points')
            plt.plot(interp[:,0], interp[:,1], c='blue', label='Interp Traj')
            plt.title(f"Sample 0 (Valid Len: {valid_lens[idx]})")
            plt.show()

        return waypoints_batch

    # 简单的双线性插值
    # 这是一个备选方案，比三次样条更简单、计算更快，但平滑度较差（一阶导数不连续，生成的轨迹是折线）。
    def interpolate_waypoints(self, preds):
        shape = list(preds.shape)
        out_shape = [50, shape[2]]
        waypoints = torch.nn.functional.interpolate(
            preds.unsqueeze(1),
            size=tuple(out_shape),
            mode="bilinear",
            align_corners=True,
        )
        waypoints = waypoints.squeeze(1)

        if self.debug:
            import matplotlib.pyplot as plt  # for plotting

            plt.scatter(
                preds[0, :, 0].detach().cpu().numpy(),
                preds[0, :, 1].detach().cpu().numpy(),
                label="Samples",
                color="purple",
            )
            plt.plot(
                waypoints[0, :, 0].detach().cpu().numpy(),
                waypoints[0, :, 1].detach().cpu().numpy(),
                label="Interpolated curve",
                color="blue",
            )
            plt.legend()
            plt.show()

        return waypoints

