'''
Author: lichao951787328 951787328@qq.com
Date: 2026-01-12 19:45:38
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-12 19:45:41
FilePath: /viplanner/draw_samplepoints.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pickle
import torch
import matplotlib.pyplot as plt

file_path = "rotated_out/viewpoints_seed1_samples8.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

# 转换为 numpy 方便画图
points = data.cpu().numpy()

x = points[:, 0]
y = points[:, 1]

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c='red', marker='x', label='Camera Position')

# 标注序号
for i, (px, py) in enumerate(zip(x, y)):
    plt.text(px, py, str(i), fontsize=9)

plt.title("Viewpoint Samples Distribution (Top-Down)")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.grid(True)
plt.axis('equal') # 保持比例一致
plt.legend()
plt.savefig("viewpoints_viz.png")
print("分布图已保存为 viewpoints_viz.png")