#!/usr/bin/env python3
"""
数据质量检查工具 (Data Quality Dashboard) - 修复版

功能：
遍历采样文件夹，为每个样本生成一张 'check_dashboard.jpg'。
在一张图上同时显示：
1. Top-down RGB (如果有)
2. 高度图 (Height Map)
3. 语义图 (Semantic Mask)
4. 叠加对比图 (Overlay) -> 最关键的检查项
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import sys

def normalize_height(height_map):
    """将高度图归一化到 0-1 以便可视化"""
    h = height_map.copy()
    mask_valid = np.isfinite(h)
    if not mask_valid.any():
        return np.zeros_like(h)
    
    h_min = h[mask_valid].min()
    h_max = h[mask_valid].max()
    
    if h_max > h_min:
        h[mask_valid] = (h[mask_valid] - h_min) / (h_max - h_min)
    else:
        h[mask_valid] = 0.5
        
    h[~mask_valid] = 0
    return h

def create_dashboard(sample_dir, save_path):
    height_path = sample_dir / "height.npy"
    sem_path = sample_dir / "sem_mask.npy"
    rgb_path = sample_dir / "top_down_rgb.png"
    
    if not height_path.exists() or not sem_path.exists():
        return False
    
    # 1. 加载数据
    height_map = np.load(height_path)
    sem_mask = np.load(sem_path)
    
    has_rgb = rgb_path.exists()
    rgb_img = None
    if has_rgb:
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is not None:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # 2. 准备绘图
    fig_cols = 4 if has_rgb else 3
    fig, axes = plt.subplots(1, fig_cols, figsize=(5 * fig_cols, 5))
    
    # --- 图 1: RGB 原图 ---
    ax_idx = 0
    if has_rgb:
        axes[ax_idx].imshow(rgb_img)
        axes[ax_idx].set_title("1. RGB Image (Ground Truth)")
        axes[ax_idx].axis('off')
        ax_idx += 1
        
    # --- 图 2: 高度图 ---
    h_viz = normalize_height(height_map)
    im = axes[ax_idx].imshow(h_viz, cmap='magma')
    axes[ax_idx].set_title("2. Height Map")
    axes[ax_idx].axis('off')
    plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
    ax_idx += 1
    
    # --- 图 3: 语义掩码 ---
    sem_viz = np.zeros((sem_mask.shape[0], sem_mask.shape[1], 3))
    sem_viz[sem_mask == 1] = [0, 1, 0] # Green
    sem_viz[sem_mask == 0] = [0.2, 0.2, 0.2] 
    
    axes[ax_idx].imshow(sem_viz)
    axes[ax_idx].set_title("3. Semantic Mask\n(Green=Go, Grey=No)")
    axes[ax_idx].axis('off')
    ax_idx += 1
    
    # --- 图 4: 叠加检查 (Overlay) ---
    # [关键修复] 处理尺寸不匹配问题
    if has_rgb:
        base_img = rgb_img.copy() / 255.0
        target_h, target_w = base_img.shape[:2]
        
        # 检查是否需要缩放 mask
        if sem_mask.shape != (target_h, target_w):
            # 使用最近邻插值放大 mask，保证类别标签不模糊 (0还是0, 1还是1)
            # cv2.resize 接受 (width, height)
            sem_mask_for_overlay = cv2.resize(
                sem_mask.astype(np.uint8), 
                (target_w, target_h), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            sem_mask_for_overlay = sem_mask
            
    else:
        # 没有RGB时，底图就是高度图，尺寸天然匹配
        base_img = plt.cm.magma(h_viz)[:, :, :3]
        sem_mask_for_overlay = sem_mask
        
    overlay = base_img.copy()
    
    # 红色覆盖不可通行区域
    mask_obstacle = (sem_mask_for_overlay == 0)
    # 确保 mask 也是 boolean 且维度匹配
    if mask_obstacle.shape != overlay.shape[:2]:
         # 双重保险
         mask_obstacle = cv2.resize(mask_obstacle.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay[mask_obstacle] = overlay[mask_obstacle] * 0.7 + np.array([1, 0, 0]) * 0.3
    
    # 绿色覆盖可通行区域
    mask_traversable = (sem_mask_for_overlay == 1)
    if mask_traversable.shape != overlay.shape[:2]:
         mask_traversable = cv2.resize(mask_traversable.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
         
    overlay[mask_traversable] = overlay[mask_traversable] * 0.7 + np.array([0, 1, 0]) * 0.3
    
    axes[ax_idx].imshow(overlay)
    axes[ax_idx].set_title("4. Overlay Check\n(Does mask align with objects?)")
    axes[ax_idx].axis('off')
    
    # 保存
    plt.tight_layout()
    output_filename = sample_dir / "check_dashboard.jpg"
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/eai/VLN/viplanner/rotated_out")
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    if not root_path.exists():
        print(f"[ERROR] Root path does not exist: {root_path}")
        return

    sample_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    print(f"Generating Quality Dashboards for {len(sample_dirs)} samples...")
    
    count = 0
    for idx, d in enumerate(sample_dirs):
        print(f"[{idx+1}/{len(sample_dirs)}] Processing {d.name}...", end="\r")
        try:
            if create_dashboard(d, d / "check_dashboard.jpg"):
                count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed on {d.name}: {e}")
            
    print(f"\nDone! Saved {count} dashboards.")
    print(f"Check the 'check_dashboard.jpg' inside each sample folder.")

if __name__ == "__main__":
    main()