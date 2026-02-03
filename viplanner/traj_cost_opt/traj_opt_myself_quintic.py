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
    
    # 针对单一段（两个点之间）的求解器
    # 它完全支持 Batch（批量）操作。输入可以是 [Batch, Dims]，利用 PyTorch 的广播机制一次性算出所有样本的轨迹。
    @staticmethod
    def generate_quintic_path(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T, num_points=10):
        """
        生成两点之间的五次多项式轨迹 (Minimum Jerk Trajectory)
        支持 Batch 计算。
        
        Args:
            start_pos: [Batch, Dims]
            start_vel: [Batch, Dims]
            start_acc: [Batch, Dims]
            end_pos:   [Batch, Dims]
            end_vel:   [Batch, Dims]
            end_acc:   [Batch, Dims]
            T:         标量或 [Batch, 1], 总时间
            num_points:生成的点数
        
        Returns:
            pos: [Batch, num_points, Dims]
            vel: [Batch, num_points, Dims]
            acc: [Batch, num_points, Dims]
        """
        device = start_pos.device
        dtype = start_pos.dtype
        
        # 1. 准备时间向量 t: [1, num_points, 1]
        # 用于广播到 [Batch, num_points, Dims]
        t = torch.linspace(0, T, num_points, device=device, dtype=dtype)
        t = t.view(1, -1, 1) 
        
        # 预计算 T 的幂次
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        # 状态差分
        h = end_pos - start_pos
        
        # 2. 计算五次多项式系数 (a0 ~ a5)
        # a0, a1, a2 由初始状态直接决定
        a0 = start_pos
        a1 = start_vel
        a2 = 0.5 * start_acc
        
        # a3, a4, a5 通过解线性方程组得到 (Minimum Jerk 标准公式)
        # 注意: 这里的维度是 [Batch, Dims]
        a3 = (20 * h - (8 * start_vel + 12 * end_vel) * T - (3 * start_acc - end_acc) * T2) / (2 * T3)
        a4 = (-30 * h + (14 * start_vel + 16 * end_vel) * T + (3 * start_acc - 2 * end_acc) * T2) / (2 * T4)
        a5 = (12 * h - (6 * start_vel + 6 * end_vel) * T - (start_acc - end_acc) * T2) / (2 * T5)
        
        # 3. 扩展系数维度以便广播: [Batch, Dims] -> [Batch, 1, Dims]
        a0 = a0.unsqueeze(1)
        a1 = a1.unsqueeze(1)
        a2 = a2.unsqueeze(1)
        a3 = a3.unsqueeze(1)
        a4 = a4.unsqueeze(1)
        a5 = a5.unsqueeze(1)
        
        # 4. 计算位置 (Pos)
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        pos = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
        
        # 5. 计算速度 (Vel) - 对时间求导
        # v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        vel = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
        
        # 6. 计算加速度 (Acc) - 对速度求导
        # a(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
        acc = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
        
        return pos, vel, acc

    # 责处理整条路径（由多个点组成序列），是实际调用的接口。它采用的是**“分段拼接”**策略。
    # waypoints 包含了位置和速度，没有加速度
    # def interp_quintic(self, waypoints, step_time=0.5, start_acc=None, goal_acc=None):
    #     """
    #     分段五次样条插值，支持自定义起点和终点的速度/加速度。
    #     Args:
    #         x: (保留接口，暂未使用)
    #         y: 路径控制点 [Batch, Num_Waypoints, Dims]
    #         step_time: 每一段的时间长度 (float)
    #         start_vel: [Batch, Dims] 起点速度
    #         start_acc: [Batch, Dims] 起点加速度
    #         end_vel:   [Batch, Dims] 终点速度 (默认为 None，即使用差分估算)
    #         end_acc:   [Batch, Dims] 终点加速度 (默认为 None，即使用差分估算)
    #     """
    #     batch_size, num_wp, dims = y.shape
    #     device = y.device
    #     # 1. 估算所有中间点的速度 (Vel) 和 加速度 (Acc)
    #     # 使用有限差分法 (Finite Difference) 配合 Catmull-Rom 策略
    #     # 计算段间向量
    #     diffs = y[:, 1:] - y[:, :-1] # [B, N-1, D]
    #     # --- A. 速度估算 ---
    #     vels = torch.zeros_like(y)
    #     # 中间点：使用中心差分 (Central Difference)
    #     vels[:, 1:-1] = (diffs[:, :-1] + diffs[:, 1:]) / 2.0 / step_time
    #     # 边界点：默认使用单边差分 (Forward/Backward Difference)
    #     vels[:, 0] = diffs[:, 0] / step_time
    #     vels[:, -1] = diffs[:, -1] / step_time
    #     # --- B. 加速度估算 ---
    #     accs = torch.zeros_like(y)
    #     # 先基于默认速度计算加速度
    #     vel_diffs = vels[:, 1:] - vels[:, :-1]
    #     accs[:, 1:-1] = (vel_diffs[:, :-1] + vel_diffs[:, 1:]) / 2.0 / step_time
    #     # 边界加速度默认为 0 (假设静止起步/停止)，或者也可以用差分
    #     accs[:, 0] = 0.0 
    #     accs[:, -1] = 0.0
    #     # 2. 【关键】应用强制约束 (如果有输入，覆盖默认估算值)
    #     # 起点约束
    #     if start_vel is not None:
    #         vels[:, 0] = start_vel
    #     if start_acc is not None:
    #         accs[:, 0] = start_acc
    #     # 终点约束 (新增)
    #     if goal_vel is not None:
    #         vels[:, -1] = goal_vel
    #     if goal_acc is not None:
    #         accs[:, -1] = goal_acc
    #     # 3. 逐段生成五次多项式
    #     all_pos_segments = []
    #     # 每一段生成的点数 (基于 step_time 和想要的密度)
    #     dt_sim = 0.1  # 仿真/插值的时间分辨率
    #     points_per_seg = int(step_time / dt_sim)
    #     # 保证至少有2个点
    #     points_per_seg = max(points_per_seg, 2)
    #     print(f"[ERROR] points_per_seg: {points_per_seg}")
    #     for i in range(num_wp - 1):
    #         # 提取当前段的边界条件
    #         p_s, p_e = y[:, i], y[:, i+1]
    #         v_s, v_e = vels[:, i], vels[:, i+1]
    #         a_s, a_e = accs[:, i], accs[:, i+1]
    #         # 生成该段轨迹
    #         seg_pos, _, _ = self.generate_quintic_path(
    #             p_s, v_s, a_s, 
    #             p_e, v_e, a_e, 
    #             T=step_time, 
    #             num_points=points_per_seg
    #         )
    #         # 拼接处理：
    #         # 如果不是最后一段，去掉该段的最后一个点，因为它是下一段的起点
    #         if i < num_wp - 2:
    #             all_pos_segments.append(seg_pos[:, :-1, :])
    #         else:
    #             # 最后一段保留所有点
    #             all_pos_segments.append(seg_pos)
    #     # 4. 拼接
    #     full_path = torch.cat(all_pos_segments, dim=1)
    #     return full_path
        
    # step_time 表示每一段的时间长度，由上层给出
    def interp_quintic(self, waypoints, step_time=0.5, start_acc=None, goal_acc=None):
        """
        分段五次样条插值 (Revised)
        
        Args:
            waypoints: [Batch, Num_Waypoints, 4] 
                       其中 waypoints[..., 0:2] 是位置 (x, y)
                       其中 waypoints[..., 2:4] 是速度 (vx, vy)
            step_time: 每一段的时间长度 (float)
            start_acc: [Batch, 2] 起点加速度 (可选，来自 Odom)
            goal_acc:  [Batch, 2] 终点加速度 (可选，通常设为 0)
        
        Returns:
            full_path: [Batch, Total_Dense_Points, 2] 生成的密集轨迹位置
        """
        batch_size, num_wp, dims = waypoints.shape
        device = waypoints.device
        
        # 1. 解包数据 (Unpack)
        # 假设 waypoints 最后一维是 4: [x, y, vx, vy]
        # 注意：这里的 dims 应该是 4
        assert dims >= 4, f"Waypoints dim must be >= 4 (pos+vel), but got {dims}"
        
        y_pos = waypoints[..., 0:2] # [B, N, 2]
        y_vel = waypoints[..., 2:4] # [B, N, 2]
        
        # 2. 估算加速度 (Estimate Acceleration)
        # 因为网络只预测了位置和速度，我们需要通过速度差分来计算加速度
        # Acc = dv / dt
        
        accs = torch.zeros_like(y_vel) # [B, N, 2]
        
        # A. 中间点加速度：使用中心差分 (Central Difference)
        # acc[i] = (vel[i+1] - vel[i-1]) / (2 * step_time)
        accs[:, 1:-1] = (y_vel[:, 2:] - y_vel[:, :-2]) / (2.0 * step_time)
        
        # B. 边界点加速度：使用单边差分
        # 起点：前向差分 (vel[1] - vel[0]) / dt
        accs[:, 0] = (y_vel[:, 1] - y_vel[:, 0]) / step_time
        # 终点：后向差分 (vel[-1] - vel[-2]) / dt
        accs[:, -1] = (y_vel[:, -1] - y_vel[:, -2]) / step_time

        # 3. 应用强制加速度约束 (Constraint Overwrite)
        # 如果从 Odom 传入了真实的当前加速度，覆盖起点
        if start_acc is not None:
            accs[:, 0] = start_acc
            
        # 如果对终点加速度有要求 (比如希望平稳停车，goal_acc=0)，覆盖终点
        if goal_acc is not None:
            accs[:, -1] = goal_acc
            
        # 4. 逐段生成五次多项式 (Segment Generation)
        all_segments = []
        
        # 定义插值密度：每段 step_time 秒，每 0.1秒 采一个点
        # 而是由“代码实现（切片拼接）”和“离散化表示”决定的
        dt_sim = 0.1 
        points_per_seg = int(step_time / dt_sim)
        points_per_seg = max(points_per_seg, 2) # 至少2个点
        
        for i in range(num_wp - 1):
            p_s, p_e = y_pos[:, i], y_pos[:, i+1]
            v_s, v_e = y_vel[:, i], y_vel[:, i+1]
            a_s, a_e = accs[:, i], accs[:, i+1]
            
            # === 修改点 1: 获取速度 ===
            # generate_quintic_path 返回 pos, vel, acc
            # 我们需要捕获 vel
            seg_pos, seg_vel, _ = self.generate_quintic_path(
                p_s, v_s, a_s, 
                p_e, v_e, a_e, 
                T=step_time, 
                num_points=points_per_seg
            )
            
            # === 修改点 2: 拼接位置和速度 ===
            # seg_pos: [B, N, 2], seg_vel: [B, N, 2]
            # 合并后: [B, N, 4]
            seg_state = torch.cat([seg_pos, seg_vel], dim=2)

            # 拼接处理：防止点重叠
            if i < num_wp - 2:
                all_segments.append(seg_state[:, :-1, :]) 
            else:
                all_segments.append(seg_state) 
        
        # 4. 合并
        full_path = torch.cat(all_segments, dim=1) # [Batch, Total_N, 4]
        
        return full_path
    
    
class TrajOpt:
    debug = False

    def __init__(self):
        self.cs_interp = CubicSplineTorch()

    # 此时输入的preds需要包含机器人预测点的速度信息
    # odom包括机器人的起点位置、起点速度、起点加速度等信息
    # 由于预测时preds不包含终点的加速信息，因此这里增加了goal_acc参数
    def TrajGeneratorQuintic(self, preds, step_time=0.5, odom=None, goal_acc=None):
        """
        使用五次样条生成轨迹，暴露所有动力学接口。
        """
        batch_size, num_p, dims = preds.shape
        device = preds.device
        
        # 1. 构造控制点序列 (原点 + 预测点)
        # 假设当前机器人在原点 (0,0)
        # start_point = torch.zeros((batch_size, 1, dims), device=device) 
        # 取 odom 的前四位作为位置和速度
        # 假设 odom 形状为 [Batch, 4]，前两位为位置，后两位为速度
        start_point = odom[:, 0:4].unsqueeze(1)
        
        waypoints = torch.cat([start_point, preds], dim=1) # [Batch, N+1, Dims]
        
        # 2. 处理默认值
        # if start_vel is None:
        #     start_vel = torch.zeros((batch_size, dims), device=device)
        
        # 注意：这里我们通常不把 end_vel 默认为 0，而是传 None 进去
        # 让 interp_quintic 内部决定是使用差分估算还是强制为0
        # 如果你希望此处显式默认为0，可以在这里赋值，但建议保持 None 以便灵活性
            
        # 3. 调用五次插值
        full_trajectory = self.cs_interp.interp_quintic(
            waypoints=waypoints, 
            step_time=step_time,
            start_acc=odom[:, 4:6],  # 取 odom 的后两位作为起点加速度
            goal_acc=None   # <--- 传入接口
        )
        
        return full_trajectory
    
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

