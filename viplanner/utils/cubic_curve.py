'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-28 16:54:44
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-28 17:09:57
FilePath: /viplanner/viplanner/utils/cubic_curve.py
Description: 三次Hermite样条可视化，包含自定义速度开关与原地转弯演示
'''
import numpy as np
import matplotlib.pyplot as plt

def cubic_hermite_spline(P0, P1, M0, M1, num_points=100):
    """
    计算三次Hermite样条曲线 (核心数学公式保持不变)
    P0, P1: 起点和终点的坐标 [x, y]
    M0, M1: 起点和终点的切线向量（速度向量） [vx, vy]
    """
    t = np.linspace(0, 1, num_points)
    
    # 三次Hermite基函数
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    # 计算曲线上的点
    x = h00*P0[0] + h10*M0[0] + h01*P1[0] + h11*M1[0]
    y = h00*P0[1] + h10*M0[1] + h01*P1[1] + h11*M1[1]
    
    return x, y

def plot_scenario(ax, p_start, p_end, angle_start=0, angle_end=0, scale=5.0, 
                  use_custom_vel=True, title=""):
    """
    绘制单个场景
    
    参数:
        scale: 【关键参数】切线向量的模长。
               - 值越大，曲线惯性越大，转弯半径越大。
               - 值越小 (如 0.1)，曲线转弯半径越小，接近原地转弯。
        use_custom_vel: 【新增选项】
               - True: 强制使用给定的 angle 和 scale 作为起止速度。
               - False: 忽略 angle，根据 P_start 指向 P_end 的向量自动生成速度（模拟无约束）。
    """
    
    if use_custom_vel:
        # --- 选项 A: 给定起止速度 (考虑机器人的 Yaw 角) ---
        # 将角度转换为单位向量，并乘以 scale (模长)
        rad_start = np.deg2rad(angle_start)
        rad_end   = np.deg2rad(angle_end)
        
        m_start = np.array([np.cos(rad_start), np.sin(rad_start)]) * scale
        m_end   = np.array([np.cos(rad_end),   np.sin(rad_end)])   * scale
        
        mode_str = f"Custom Vel (Scale={scale})"
    else:
        # --- 选项 B: 不给定速度 (自动计算/无约束) ---
        # 这里使用一种简单的启发式：速度 = (终点 - 起点)，这会产生一条直线轨迹
        # 在实际规划器中，这通常由 Catmull-Rom 或其它插值算法根据前后点决定
        direction = p_end - p_start
        m_start = direction
        m_end   = direction
        
        mode_str = "Auto Vel (Unconstrained)"

    # 计算轨迹
    x, y = cubic_hermite_spline(p_start, p_end, m_start, m_end)
    
    # --- 绘图部分 ---
    ax.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    
    # 画起点(绿)和终点(红)
    ax.plot(p_start[0], p_start[1], 'go', markersize=8, label='Start')
    ax.plot(p_end[0],   p_end[1],   'ro', markersize=8, label='Goal')
    
    # 画速度向量箭头 (Quiver)
    # 这里的箭头长度直观反映了 scale 的大小
    ax.quiver(p_start[0], p_start[1], m_start[0], m_start[1], 
              angles='xy', scale_units='xy', scale=1, color='g', width=0.015, alpha=0.6, label='Vel Vector')
    ax.quiver(p_end[0],   p_end[1],   m_end[0],   m_end[1],   
              angles='xy', scale_units='xy', scale=1, color='r', width=0.015, alpha=0.6)
    
    ax.set_title(f"{title}\n[{mode_str}]")
    ax.grid(True)
    ax.axis('equal')

# --- 主程序 ---
fig, axs = plt.subplots(1, 4, figsize=(24, 5))

# 场景 1: 不给定速度 (use_custom_vel=False)
# 效果：忽略角度，直接连线，类似于 A* 出来的折线效果（如果是多段的话）
plot_scenario(axs[0], np.array([0,0]), np.array([5,5]), 
              use_custom_vel=False, 
              title="1. No Velocity Given")

# 场景 2: 给定速度，正常 Scale (use_custom_vel=True, scale=5.0)
# 效果：平滑的大半径转弯，适合高速行驶
plot_scenario(axs[1], np.array([0,0]), np.array([5,5]), 
              angle_start=0, angle_end=90, 
              scale=5.0, use_custom_vel=True, 
              title="2. Custom Vel (High Speed)")

# =======================================================
# 下面两个场景展示【原地转弯】的关键点
# =======================================================

# 场景 3: “大肚子”回环 (Scale 很大)
# 这是一个困难场景：终点在起点的侧后方，但车头都朝右
plot_scenario(axs[2], np.array([0,0]), np.array([-2,2]), 
              angle_start=0, angle_end=0, 
              scale=5.0, use_custom_vel=True, 
              title="3. Overshoot (Large Scale)")

# 场景 4: 【类似于原地转弯】 (Scale 很小)
# 修改点：将 scale 从 5.0 改为 0.5
# 效果：曲线在起点立刻转向，不再向前冲，实现了极小半径的变道
plot_scenario(axs[3], np.array([0,0]), np.array([-2,2]), 
              angle_start=0, angle_end=0, 
              scale=1, use_custom_vel=True, 
              title="4. Turn-in-Place (Small Scale)")

plt.legend()
plt.tight_layout()
plt.show()