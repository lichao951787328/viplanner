'''
Author: lichao951787328
Description: 五次Hermite样条可视化 (Quintic)，仅考虑起止速度 (加速度默认为0)
'''
import numpy as np
import matplotlib.pyplot as plt

def quintic_hermite_spline(P0, P1, V0, V1, num_points=100):
    """
    计算五次Hermite样条曲线
    
    参数:
    P0, P1: 起点和终点坐标 [x, y]
    V0, V1: 起点和终点速度向量 [vx, vy]
    
    注意: 
    为了满足"不考虑起止加速度"的要求，我们在数学公式中
    将起点加速度 A0 和终点加速度 A1 强制设为 [0, 0]。
    """
    t = np.linspace(0, 1, num_points)
    
    # 预计算 t 的幂次
    t2 = t**2
    t3 = t**3
    t4 = t**4
    t5 = t**5

    # --- 五次 Hermite 基函数 (加速度为0时的简化版) ---
    # 标准公式中包含加速度项，但因为 A0=A1=0，相关项被消去了。
    # 剩下的项仅与 位置(P) 和 速度(V) 有关。
    
    # 1. 位置 P0 的系数: h00 = 1 - 10t^3 + 15t^4 - 6t^5
    h_p0 = 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5
    
    # 2. 速度 V0 的系数: h10 = t - 6t^3 + 8t^4 - 3t^5
    h_v0 = t - 6.0*t3 + 8.0*t4 - 3.0*t5
    
    # 3. 加速度 A0 的系数 (已忽略，因为 A0=0)
    
    # 4. 位置 P1 的系数: h01 = 10t^3 - 15t^4 + 6t^5
    h_p1 = 10.0*t3 - 15.0*t4 + 6.0*t5
    
    # 5. 速度 V1 的系数: h11 = -4t^3 + 7t^4 - 3t^5
    h_v1 = -4.0*t3 + 7.0*t4 - 3.0*t5
    
    # 6. 加速度 A1 的系数 (已忽略，因为 A1=0)
    
    # 计算曲线上的坐标
    x = h_p0*P0[0] + h_v0*V0[0] + h_p1*P1[0] + h_v1*V1[0]
    y = h_p0*P0[1] + h_v0*V0[1] + h_p1*P1[1] + h_v1*V1[1]
    
    return x, y

def plot_scenario(ax, p_start, p_end, angle_start=0, angle_end=0, scale=5.0, 
                  use_custom_vel=True, title=""):
    """
    绘制单个场景
    """
    
    if use_custom_vel:
        # 将角度转换为速度向量
        rad_start = np.deg2rad(angle_start)
        rad_end   = np.deg2rad(angle_end)
        
        m_start = np.array([np.cos(rad_start), np.sin(rad_start)]) * scale
        m_end   = np.array([np.cos(rad_end),   np.sin(rad_end)])   * scale
        
        mode_str = f"Custom Vel (Scale={scale})"
    else:
        # 自动模式：直接指向终点
        direction = p_end - p_start
        m_start = direction
        m_end   = direction
        mode_str = "Auto Vel (Unconstrained)"

    # --- 这里调用五次样条函数 ---
    x, y = quintic_hermite_spline(p_start, p_end, m_start, m_end)
    
    # --- 绘图部分 ---
    ax.plot(x, y, 'b-', linewidth=2, label='Quintic Traj')
    
    # 画起点(绿)和终点(红)
    ax.plot(p_start[0], p_start[1], 'go', markersize=8, label='Start')
    ax.plot(p_end[0],   p_end[1],   'ro', markersize=8, label='Goal')
    
    # 画速度向量箭头
    ax.quiver(p_start[0], p_start[1], m_start[0], m_start[1], 
              angles='xy', scale_units='xy', scale=1, color='g', width=0.015, alpha=0.6, label='Vel Vector')
    ax.quiver(p_end[0],   p_end[1],   m_end[0],   m_end[1],   
              angles='xy', scale_units='xy', scale=1, color='r', width=0.015, alpha=0.6)
    
    ax.set_title(f"{title}\n[{mode_str}]")
    ax.grid(True)
    ax.axis('equal')

# --- 主程序 ---
if __name__ == "__main__":
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    # 场景 1: 不给定速度
    plot_scenario(axs[0], np.array([0,0]), np.array([5,5]), 
                  use_custom_vel=False, 
                  title="1. Quintic - No Vel Given")

    # 场景 2: 给定速度，正常 Scale
    # 五次样条比三次样条看起来更平滑，起步时更"柔和"
    plot_scenario(axs[1], np.array([0,0]), np.array([5,5]), 
                  angle_start=0, angle_end=90, 
                  scale=5.0, use_custom_vel=True, 
                  title="2. Quintic - High Speed")

    # 场景 3: “大肚子”回环
    plot_scenario(axs[2], np.array([0,0]), np.array([-2,2]), 
                  angle_start=0, angle_end=0, 
                  scale=5.0, use_custom_vel=True, 
                  title="3. Quintic - Overshoot")

    # 场景 4: 原地转弯测试 (小 Scale)
    plot_scenario(axs[3], np.array([0,0]), np.array([-2,2]), 
                  angle_start=0, angle_end=0, 
                  scale=1.0, use_custom_vel=True, 
                  title="4. Quintic - Turn-in-Place")

    plt.legend()
    plt.tight_layout()
    plt.show()