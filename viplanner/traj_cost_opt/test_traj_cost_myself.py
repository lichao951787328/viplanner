'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-23 17:29:40
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-26 21:00:07
FilePath: /viplanner/viplanner/traj_cost_opt/test_traj_cost_myself.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import unittest
import torch
import numpy as np
import os
import shutil
import pypose as pp
import torch.nn.functional as F

from viplanner.traj_cost_opt.traj_cost_myself import TrajCost

def main():
    npy_path = "/home/eai/VLN/viplanner/rotated_out/carla/sample_01713"
    TrajCost_obj = TrajCost(   # 创建轨迹成本计算器实例
                0,
                log_data=False,
                w_obs=0.25,  # 障碍物权重
                w_goal=4.0,  # 目标权重
                w_motion=1.5,  # 运动成本权重
                obstalce_thread=0.02,  # 障碍物阈值
            )
    TrajCost_obj.SetMap(npy_path, "costmap.npy")  # 设置代价地图
    # 生成两组测试轨迹点，每组为5个二维点
    traj1 = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
    traj2 = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.float32)
    # 合并为一个 (2, 5, 2) 的数组
    test_trajs = np.stack([traj1, traj2], axis=0)
    # 转为torch张量并放到正确设备
    test_trajs_tensor = torch.from_numpy(test_trajs).to(TrajCost_obj.device)
    waypoints = torch.cat([test_trajs_tensor, torch.zeros(test_trajs_tensor.shape[0], test_trajs_tensor.shape[1], 1, device=TrajCost_obj.device)], dim=-1)
    # odom, goal 都是四维的，分别是x,y,vx,vy
    # odom = torch.tensor([[0.0, 0.0, 2, 0.0]], device=TrajCost_obj.device)
    # goal = torch.tensor([[4.0, 1.0, 0.01, 0.0]], device=TrajCost_obj.device)
    odom = torch.tensor([
        [0.0, 0.0, 4.0, -1.0],
        [0.0, 0.0, 0.0, 4.0]
    ], device=TrajCost_obj.device)
    goal = torch.tensor([
        [4.0, 1.0, 0.0, 0.0],  # Case 1 Goal: (4,0)
        [1.0, 4.0, 0.0, 0.0]   # Case 2 Goal: (0,4)
    ], device=TrajCost_obj.device)
    fear = torch.tensor([[0.8], [0.1]], device=TrajCost_obj.device)
    log_step = 0
    ahead_dist = 2.5
    dataset = "train"
    TrajCost_obj.CostofTraj(
        waypoints=waypoints,
        odom=odom,
        goal=goal,
        fear=fear,
        log_step=log_step,
        ahead_dist=ahead_dist,
        dataset=dataset
    )
    
    
if __name__ == "__main__":
    main()