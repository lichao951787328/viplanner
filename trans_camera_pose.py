'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-13 19:32:44
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-13 19:34:53
FilePath: /viplanner/trans_camera_pose.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import numpy as np
from glob import glob

# 根目录
root_dir = '/home/eai/VLN/viplanner/rotated_out/carla'
output_txt = 'all_camera_poses.txt'

with open(output_txt, 'w') as f_out:
    for sample_dir in sorted(glob(os.path.join(root_dir, 'sample_*'))):
        pose_path = os.path.join(sample_dir, 'camera_pose.npy')
        if os.path.exists(pose_path):
            pose = np.load(pose_path, allow_pickle=True)
            # 位置
            position = pose[:3, 3]
            # 朝向（旋转矩阵）
            rotation = pose[:3, :3].flatten()
            f_out.write(f"{sample_dir}: pos={position.tolist()}, rot={rotation.tolist()}\n")
        else:
            f_out.write(f"{sample_dir}: camera_pose.npy not found\n")

print(f"所有相机位姿已保存到 {output_txt}")