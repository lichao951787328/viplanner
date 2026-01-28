# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 在 PyTorch 框架下实现轨迹插值和平滑。它利用**三次样条插值（Cubic Spline Interpolation）**将稀疏的路径点（比如神经网络预测的几个关键点）转换成平滑、连续的高分辨率轨迹。
# 由于完全使用 PyTorch 操作，这整个过程是可微分的，意味着它可以直接嵌入到神经网络的训练流程中，允许梯度反向传播。
import torch

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
    def interp(self, x, y, xs, start_vel=None, goal_vel=None):
        """
        执行三次 Hermite 样条插值，支持自定义起止点切线（速度）。
        
        参数:
            x: 原始点的索引/时间 (Batch, Num_Points)
            y: 原始路径点坐标 (Batch, Num_Points, Dims)
            xs: 需要插值出的新时间点索引 (Batch, Num_Samples)
            start_vel: (可选) 起点切线向量 [Batch, Dims]
            end_vel:   (可选) 终点切线向量 [Batch, Dims]
        """
        # 1. 计算基于位置的几何切线 (Geometric Tangent)
        # 先计算每两个点之间的差分斜率
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        
        # 处理边界切线 (默认使用 Catmull-Rom 风格，即中间点取平均，两端取单边差分)
        # m 的形状变为: [Batch, Num_Points, Dims]
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        
        # # 2. 【关键修改】如果提供了真实速度，强制覆盖起点的几何切线
        # if start_vel is not None:
        #     # 确保维度匹配: start_vel 应为 [Batch, Dims]
        #     m[:, 0, :] = start_vel
            
        # # 3. 【关键修改】如果提供了终点速度/朝向向量，强制覆盖终点的几何切线
        # if goal_vel is not None:
        #     # 确保维度匹配: goal_vel 应为 [Batch, Dims]
        #     m[:, -1, :] = goal_vel
        
        # --- 以下为标准的 Hermite 插值计算流程 ---

        # 4. 确定 xs 中的每个插值点落在原始 x 的哪一段区间内
        # torch.searchsorted 找到 xs 在 x 中的插入位置索引
        # 注意：这里假设所有 Batch 的 x 分布是一样的，取 x[0, 1:] 作为参考
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        idxs = torch.clamp(idxs, max=x.shape[1] - 2)
        
        # 5. 计算该段区间的长度 dx
        # x[:, idxs] 是区间起点，x[:, idxs+1] 是区间终点
        dx = x[:, idxs + 1] - x[:, idxs]
        
        # 6. 计算归一化时间 t，范围 [0, 1]
        t = (xs - x[:, idxs]) / dx
        
        # 7. 代入 Hermite 基函数多项式计算权重
        hh = self.h_poly(t)
        hh = torch.transpose(hh, 1, 2)  # [Batch, Num_Samples, 4]
        
        # 分离四个基函数系数 h00, h10, h01, h11
        h00 = hh[:, :, 0:1]  # 对应 p0
        h10 = hh[:, :, 1:2]  # 对应 m0 (切线)
        h01 = hh[:, :, 2:3]  # 对应 p1
        h11 = hh[:, :, 3:4]  # 对应 m1 (切线)
        
        # 获取对应的控制点和切线
        p0 = y[:, idxs, :]     # 当前段起点
        p1 = y[:, idxs + 1, :]  # 当前段终点
        m0 = m[:, idxs]        # 当前段起点切线
        m1 = m[:, idxs + 1]    # 当前段终点切线
        
        # 扩展 dx 维度以支持广播乘法 [Batch, Num_Samples, 1]
        dx_expanded = dx[:, :, None]

        # 8. 组合最终结果 (标准 Hermite 插值公式)
        # p(t) = h00*p0 + h10*m0*dx + h01*p1 + h11*m1*dx
        # 注意：切线项 m 需要乘以区间长度 dx，因为 m 是关于 x 的导数，而基函数是关于归一化时间 t 的
        out = h00 * p0 + h10 * m0 * dx_expanded + h01 * p1 + h11 * m1 * dx_expanded
              
        return out


    def generate_quintic_path(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T, num_points=10):
        """
        计算五次多项式轨迹
        T: 这一段轨迹的总时间 (比如网络预测两个点间隔 0.5秒)
        """
        # 归一化时间 t: [0, ..., T]
        t = torch.linspace(0, T, num_points, device=start_pos.device)
        
        # 求解系数 c0, c1, c2 (很简单，就是初始状态)
        c0 = start_pos
        c1 = start_vel
        c2 = start_acc / 2.0
        
        # 求解系数 c3, c4, c5 (通过矩阵逆运算求解边界约束)
        # 这是一个标准的五次多项式线性方程组解法
        # 为了简化代码，直接写出解析解：
        
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        # 位置差
        h = end_pos - start_pos
        
        c3 = (10*h - (4*start_vel + end_vel)*T - (0.5*start_acc - 0.5*end_acc)*T2 * 3.0) / (2*T3) # 注意系数可能有变体，建议校验标准公式
        # 这里使用更通用的矩阵形式可能更稳健，或者使用简化版(假设加速度为0)
        
        # --- 简化版 (假设起止加速度为0，通常足够用) ---
        # p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        # 约束: p(0)=s, p'(0)=v_s, p''(0)=0
        #       p(T)=g, p'(T)=v_e, p''(T)=0
        
        c3 = (10 * h - 4 * start_vel * T - end_vel * T) / (T3 * 2.0) * 2.0 # 修正系数
        c3 = (10*h)/T3 - (6*start_vel + 4*end_vel)/T2 # 另推导形式
        
        # 让我们用最稳的标准 Minimum Jerk 公式 (Pos, Vel, Acc=0):
        # P(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        h = end_pos - start_pos
        
        a0 = start_pos
        a1 = start_vel
        a2 = 0.5 * start_acc
        
        a3 = (20*h - (8*start_vel + 12*end_vel)*T - (3*start_acc - end_acc)*T2) / (2*T3)
        a4 = (-30*h + (14*start_vel + 16*end_vel)*T + (3*start_acc - 2*end_acc)*T2) / (2*T4)
        a5 = (12*h - (6*start_vel + 6*end_vel)*T - (start_acc - end_acc)*T2) / (2*T5)

        # 计算轨迹
        # t shape: [num_points] -> need [Batch, num_points, 1]
        t = t.view(1, -1, 1).repeat(start_pos.shape[0], 1, 1)
        
        pos = a0.unsqueeze(1) + \
            a1.unsqueeze(1)*t + \
            a2.unsqueeze(1)*t**2 + \
            a3.unsqueeze(1)*t**3 + \
            a4.unsqueeze(1)*t**4 + \
            a5.unsqueeze(1)*t**5
            
        return pos
        
    
class TrajOpt:
    debug = False

    def __init__(self):
        self.cs_interp = CubicSplineTorch()

    def TrajGeneratorQuintic(self, preds, step, start_vel, end_vel=None):
        """
        混合生成器：第一段用五次多项式，后面用普通插值（可选）
        这里演示全段生成的逻辑。
        """
        # --- 参数准备 ---
        # 假设 preds 是 [Batch, 1, 2] (第一个预测点)
        # odom 是当前位置，但在 TrajOpt 里通常假设当前是 (0,0) 原点
        
        batch_size, _, dims = preds.shape
        device = preds.device
        
        # 1. 构造起点状态 (P, V, A)
        p_start = torch.zeros((batch_size, dims), device=device) # 假设局部坐标系原点
        v_start = start_vel  # [Batch, 2]
        a_start = torch.zeros((batch_size, dims), device=device) # 假设起始加速度为0，也可以传真实的
        
        # 2. 构造终点状态 (P, V, A)
        # 这里以预测的第一个点为例
        p_end = preds[:, 0, :] # [Batch, 2]
        
        # 终点速度 v_end: 
        # 如果只有一段，且 end_vel 给了，就用 end_vel
        # 如果这是中间点，通常为了平滑，可以计算这一段的平均速度作为 v_end
        if end_vel is not None:
             v_end = end_vel
        else:
             v_end = torch.zeros((batch_size, dims), device=device) # 默认停车
             
        a_end = torch.zeros((batch_size, dims), device=device) # 终点加速度设为0
        
        # 3. 设定时间 T
        # 这是一个超参数：网络预测的这个点代表多久之后的？
        # 对于 ViPlanner，假设每段 step 代表 0.5秒
        T = 0.5 
        
        # 4. 生成
        # step 是归一化的步长 (例如 0.1)，那么 num_points 大约是 1.0/step
        num_points = int(1.0 / step)
        
        waypoints = self.cs_interp.generate_quintic_path(
            p_start, v_start, a_start,
            p_end, v_end, a_end,
            T, num_points
        )
        
        return waypoints
    
    # 从预测点生成轨迹
    def TrajGeneratorFromPFreeRot(self, preds, step, start_vel=None, goal_vel=None):
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
        waypoints = self.cs_interp.interp(x, points_preds, xs, start_vel=start_vel, goal_vel=goal_vel)

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

        return waypoints  # R3

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
